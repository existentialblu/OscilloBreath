import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyedflib
from pathlib import Path
from datetime import datetime
import webbrowser

def parse_date_from_filename(filename):
    """Extract date from ResMed filename format YYYYMMDD_HHMMSS_BRP.edf"""
    try:
        date_str = filename.split('_')[0]
        date = datetime.strptime(date_str, '%Y%m%d')
        return date
    except:
        return None

def find_edf_files(folder_path):
    """Find all .edf files in folder and subdirectories, sort by date"""
    folder = Path(folder_path)
    edf_files = list(folder.glob('**/*_BRP.edf'))

    valid_files = []
    for f in edf_files:
        date = parse_date_from_filename(f.stem)
        if date:
            valid_files.append((date, f))

    valid_files.sort(key=lambda x: x[0])
    return valid_files

def read_flow_data(edf_path):
    """Extract flow rate data from EDF file"""
    try:
        with pyedflib.EdfReader(str(edf_path)) as f:
            signal_labels = f.getSignalLabels()

            flow_keywords = ['flow', 'Flow', 'FLOW']
            flow_idx = None

            for i, label in enumerate(signal_labels):
                if any(keyword in label for keyword in flow_keywords):
                    flow_idx = i
                    break

            if flow_idx is None:
                return None, None, None

            flow = f.readSignal(flow_idx)
            sample_rate = f.getSampleFrequency(flow_idx)
            duration_hours = len(flow) / sample_rate / 3600

            return flow, sample_rate, duration_hours
    except Exception as e:
        print(f"  Error reading {edf_path.name}: {e}")
        return None, None, None

def calculate_suffering_score(flow, sample_rate):
    """Calculate the primary suffering metric: derivative range"""
    dt = 1.0 / sample_rate
    flow_derivative = np.gradient(flow, dt)

    # Primary metric: derivative range (violence of changes)
    deriv_range = np.ptp(flow_derivative)

    # Secondary metrics for context
    deriv_std = np.std(flow_derivative)
    deriv_95th = np.percentile(np.abs(flow_derivative), 95)

    return {
        'deriv_range': deriv_range,
        'deriv_std': deriv_std,
        'deriv_95th': deriv_95th
    }

def smooth_data(values, window_size=7):
    """Apply moving average smoothing"""
    if len(values) < window_size:
        window_size = max(3, len(values) // 2)

    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    pad_size = len(values) - len(smoothed)
    left_pad = pad_size // 2
    right_pad = pad_size - left_pad

    smoothed = np.concatenate([
        np.full(left_pad, smoothed[0]),
        smoothed,
        np.full(right_pad, smoothed[-1])
    ])

    return smoothed

def create_suffering_plot(dates, metrics_list):
    """Create focused plot showing suffering score over time"""

    # Extract derivative range (primary suffering metric)
    deriv_ranges = [m['deriv_range'] for m in metrics_list]
    deriv_stds = [m['deriv_std'] for m in metrics_list]
    deriv_95ths = [m['deriv_95th'] for m in metrics_list]

    # Calculate statistics for annotations
    mean_suffering = np.mean(deriv_ranges)
    min_suffering = np.min(deriv_ranges)
    max_suffering = np.max(deriv_ranges)
    recent_avg = np.mean(deriv_ranges[-30:]) if len(deriv_ranges) >= 30 else np.mean(deriv_ranges)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Suffering Score: dFlow/dt Range (Lower is Better)',
            'Supporting Metrics'
        ),
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35]
    )

    # Main suffering score plot
    smoothed = smooth_data(deriv_ranges)

    # Raw points
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=deriv_ranges,
            mode='markers',
            marker=dict(
                size=6,
                color=deriv_ranges,
                colorscale='RdYlGn_r',  # Red = high suffering, Green = low
                showscale=True,
                colorbar=dict(
                    title="Suffering<br>Score",
                    y=0.775,
                    len=0.5
                ),
                opacity=0.6
            ),
            name='Nightly Score',
            hovertemplate='%{x|%Y-%m-%d}<br>Suffering: %{y:.2f} L/s²<extra></extra>'
        ),
        row=1, col=1
    )

    # Smoothed trend
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=smoothed,
            mode='lines',
            line=dict(color='darkred', width=3),
            name='7-day Trend',
            hovertemplate='%{x|%Y-%m-%d}<br>Trend: %{y:.2f} L/s²<extra></extra>'
        ),
        row=1, col=1
    )

    # Add reference lines
    fig.add_hline(
        y=mean_suffering,
        line=dict(color='gray', dash='dash', width=1),
        annotation_text=f"Overall Avg: {mean_suffering:.1f}",
        annotation_position="right",
        row=1, col=1
    )

    fig.add_hline(
        y=recent_avg,
        line=dict(color='green', dash='dash', width=1),
        annotation_text=f"Recent Avg: {recent_avg:.1f}",
        annotation_position="right",
        row=1, col=1
    )

    # Supporting metrics (std and 95th percentile)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=deriv_stds,
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.4),
            name='Std Dev',
            showlegend=True,
            hovertemplate='%{x|%Y-%m-%d}<br>Std: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=smooth_data(deriv_stds),
            mode='lines',
            line=dict(color='blue', width=2),
            name='Std Trend',
            showlegend=True,
            hovertemplate='%{x|%Y-%m-%d}<br>Trend: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=deriv_95ths,
            mode='markers',
            marker=dict(size=4, color='purple', opacity=0.4),
            name='95th Percentile',
            showlegend=True,
            hovertemplate='%{x|%Y-%m-%d}<br>95th: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=smooth_data(deriv_95ths),
            mode='lines',
            line=dict(color='purple', width=2),
            name='95th Trend',
            showlegend=True,
            hovertemplate='%{x|%Y-%m-%d}<br>Trend: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Derivative Range (L/s²)", row=1, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value (L/s²)", row=2, col=1)

    # Layout with stats annotation
    stats_text = (
        f"<b>Statistics:</b><br>"
        f"Best Night: {min_suffering:.1f} L/s²<br>"
        f"Worst Night: {max_suffering:.1f} L/s²<br>"
        f"Overall Average: {mean_suffering:.1f} L/s²<br>"
        f"Recent 30-day Avg: {recent_avg:.1f} L/s²<br>"
        f"<br>"
        f"<b>Improvement:</b> {((mean_suffering - recent_avg) / mean_suffering * 100):.1f}%"
    )

    fig.update_layout(
        title="OscilloBreath Suffering Tracker - Derivative Range Over Time",
        height=1000,
        hovermode='closest',
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref='paper',
                yref='paper',
                text=stats_text,
                showarrow=False,
                align='left',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=10)
            )
        ]
    )

    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'suffering_tracker',
            'height': 1000,
            'width': 1600,
            'scale': 2
        },
        'displaylogo': False
    }

    return fig, config

def main():
    print("=" * 70)
    print("OscilloBreath - Suffering Tracker")
    print("Derivative Range: The Violence of Your Breathing")
    print("=" * 70)
    print()

    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    folder_path = filedialog.askdirectory(
        title="Select folder containing EDF files"
    )
    root.destroy()

    if not folder_path:
        print("No folder selected. Exiting.")
        return

    print(f"Scanning folder: {folder_path}")
    print()

    edf_files = find_edf_files(folder_path)

    if not edf_files:
        print("No valid EDF files found")
        return

    print(f"Found {len(edf_files)} EDF files")
    print()

    dates = []
    metrics_list = []
    processed_count = 0

    for date, filepath in edf_files:
        print(f"Processing {filepath.name}...")

        flow, sample_rate, duration_hours = read_flow_data(filepath)

        if flow is None:
            print(f"  Skipped (read error)")
            continue

        if duration_hours < 0.5:
            print(f"  Skipped (only {duration_hours*60:.1f} minutes)")
            continue

        metrics = calculate_suffering_score(flow, sample_rate)

        dates.append(date)
        metrics_list.append(metrics)
        processed_count += 1

        print(f"  ✓ Suffering score: {metrics['deriv_range']:.2f}")

    print()
    print("=" * 70)
    print(f"Successfully processed {processed_count} nights")
    print("=" * 70)
    print()

    if processed_count == 0:
        print("No data to plot!")
        return

    print("Creating suffering tracker visualization...")
    fig, config = create_suffering_plot(dates, metrics_list)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "suffering_tracker.html"
    fig.write_html(str(output_file), config=config)

    print()
    print("=" * 70)
    print(f"Suffering tracker complete!")
    print(f"Saved to: {output_file}")
    print("=" * 70)
    print()
    print("Opening in browser...")

    webbrowser.open(f'file://{output_file.absolute()}')

if __name__ == "__main__":
    main()
