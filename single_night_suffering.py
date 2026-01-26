import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyedflib
from tkinter import Tk, filedialog
from pathlib import Path
import webbrowser

def select_edf_file():
    """Open file picker to select an EDF file"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select EDF file to analyze",
        filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def read_flow_data(edf_path):
    """Extract flow rate data from EDF file"""
    print(f"Reading {Path(edf_path).name}...")

    with pyedflib.EdfReader(edf_path) as f:
        signal_labels = f.getSignalLabels()
        print(f"Available signals: {signal_labels}")

        flow_keywords = ['flow', 'Flow', 'FLOW']
        flow_idx = None

        for i, label in enumerate(signal_labels):
            if any(keyword in label for keyword in flow_keywords):
                flow_idx = i
                print(f"Found flow signal: {label}")
                break

        if flow_idx is None:
            print("Available signals:")
            for i, label in enumerate(signal_labels):
                print(f"  [{i}] {label}")
            raise ValueError("Could not find flow signal. Please check signal names above.")

        flow = f.readSignal(flow_idx)
        sample_rate = f.getSampleFrequency(flow_idx)

        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(flow)/sample_rate/3600:.2f} hours")

    return flow, sample_rate

def analyze_suffering_windows(flow, sample_rate, window_minutes=5):
    """
    Analyze derivative range in sliding windows to see when suffering occurred
    """
    print(f"Analyzing suffering in {window_minutes}-minute windows...")

    # Calculate derivative
    dt = 1.0 / sample_rate
    flow_derivative = np.gradient(flow, dt)

    # Overall metrics
    overall_deriv_range = np.ptp(flow_derivative)
    overall_deriv_std = np.std(flow_derivative)

    # Window analysis
    window_samples = int(window_minutes * 60 * sample_rate)
    hop_samples = window_samples // 2  # 50% overlap

    num_windows = (len(flow) - window_samples) // hop_samples + 1

    window_data = {
        'deriv_range': [],
        'deriv_std': [],
        'deriv_mean': [],
        'deriv_95th': [],
        'flow_std': [],
        'window_start_time': [],
        'window_center_time': []
    }

    for i in range(num_windows):
        start_idx = i * hop_samples
        end_idx = start_idx + window_samples

        if end_idx > len(flow):
            break

        window_flow = flow[start_idx:end_idx]
        window_deriv = flow_derivative[start_idx:end_idx]

        # Calculate window metrics
        window_data['deriv_range'].append(np.ptp(window_deriv))
        window_data['deriv_std'].append(np.std(window_deriv))
        window_data['deriv_mean'].append(np.mean(np.abs(window_deriv)))
        window_data['deriv_95th'].append(np.percentile(np.abs(window_deriv), 95))
        window_data['flow_std'].append(np.std(window_flow))

        start_time = start_idx / sample_rate / 60  # minutes
        center_time = (start_idx + window_samples // 2) / sample_rate / 60
        window_data['window_start_time'].append(start_time)
        window_data['window_center_time'].append(center_time)

    # Convert to arrays
    for key in window_data:
        window_data[key] = np.array(window_data[key])

    # Find worst and best periods
    worst_idx = np.argmax(window_data['deriv_range'])
    best_idx = np.argmin(window_data['deriv_range'])

    worst_time = window_data['window_start_time'][worst_idx]
    best_time = window_data['window_start_time'][best_idx]

    stats = {
        'overall_deriv_range': overall_deriv_range,
        'overall_deriv_std': overall_deriv_std,
        'worst_window_range': window_data['deriv_range'][worst_idx],
        'worst_time_minutes': worst_time,
        'best_window_range': window_data['deriv_range'][best_idx],
        'best_time_minutes': best_time,
        'mean_window_range': np.mean(window_data['deriv_range']),
        'median_window_range': np.median(window_data['deriv_range'])
    }

    return flow_derivative, window_data, stats

def create_single_night_plot(flow, sample_rate, flow_derivative, window_data, stats, filename):
    """Create comprehensive single-night suffering analysis"""

    print("Creating visualization...")

    # Downsample for phase space (keep every Nth point)
    downsample_factor = max(1, len(flow) // 50000)
    if downsample_factor > 1:
        flow_viz = flow[::downsample_factor]
        flow_derivative_viz = flow_derivative[::downsample_factor]
        time_viz = np.arange(len(flow_viz)) * downsample_factor
    else:
        flow_viz = flow
        flow_derivative_viz = flow_derivative
        time_viz = np.arange(len(flow))

    time_minutes_viz = time_viz / sample_rate / 60
    time_minutes_full = np.arange(len(flow)) / sample_rate / 60

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Phase Space Portrait (colored by derivative range)',
            'Windowed Suffering Score Over Time',
            'Flow Rate Over Time',
            'Flow Derivative Over Time'
        ),
        vertical_spacing=0.08,
        row_heights=[0.35, 0.25, 0.2, 0.2]
    )

    # 1. Phase space colored by derivative magnitude
    fig.add_trace(
        go.Scattergl(
            x=flow_viz,
            y=flow_derivative_viz,
            mode='markers',
            marker=dict(
                size=2,
                color=np.abs(flow_derivative_viz),
                colorscale='RdYlGn_r',  # Red = high violence, Green = low
                showscale=True,
                colorbar=dict(
                    title="Violence<br>(|dFlow/dt|)",
                    x=1.15,
                    len=0.3,
                    y=0.825
                ),
                opacity=0.5
            ),
            name='Phase Space',
            hovertemplate='Flow: %{x:.2f}<br>dFlow/dt: %{y:.2f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )

    # 2. Windowed suffering score
    window_times = window_data['window_center_time']
    deriv_ranges = window_data['deriv_range']

    # Color-code windows
    fig.add_trace(
        go.Scatter(
            x=window_times,
            y=deriv_ranges,
            mode='lines+markers',
            line=dict(color='darkred', width=2),
            marker=dict(
                size=6,
                color=deriv_ranges,
                colorscale='RdYlGn_r',
                showscale=False,
                opacity=0.8
            ),
            fill='tozeroy',
            fillcolor='rgba(139, 0, 0, 0.2)',
            name='Suffering Score',
            hovertemplate='Time: %{x:.1f} min<br>Derivative Range: %{y:.2f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )

    # Add mean line
    fig.add_hline(
        y=stats['mean_window_range'],
        line=dict(color='gray', dash='dash', width=1),
        annotation_text=f"Mean: {stats['mean_window_range']:.1f}",
        annotation_position="right",
        row=2, col=1
    )

    # Mark worst period
    fig.add_trace(
        go.Scatter(
            x=[stats['worst_time_minutes']],
            y=[stats['worst_window_range']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x', line=dict(width=3)),
            name='Worst Period',
            hovertemplate=f'Worst: {stats["worst_time_minutes"]:.1f} min<extra></extra>',
            showlegend=True
        ),
        row=2, col=1
    )

    # Mark best period
    fig.add_trace(
        go.Scatter(
            x=[stats['best_time_minutes']],
            y=[stats['best_window_range']],
            mode='markers',
            marker=dict(size=15, color='green', symbol='circle', line=dict(width=3)),
            name='Best Period',
            hovertemplate=f'Best: {stats["best_time_minutes"]:.1f} min<extra></extra>',
            showlegend=True
        ),
        row=2, col=1
    )

    # 3. Flow rate
    fig.add_trace(
        go.Scattergl(
            x=time_minutes_viz,
            y=flow_viz,
            mode='lines',
            line=dict(color='blue', width=0.5),
            name='Flow',
            showlegend=False,
            hovertemplate='Time: %{x:.1f} min<br>Flow: %{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )

    # 4. Derivative
    fig.add_trace(
        go.Scattergl(
            x=time_minutes_viz,
            y=flow_derivative_viz,
            mode='lines',
            line=dict(color='darkred', width=0.5),
            name='Derivative',
            showlegend=False,
            hovertemplate='Time: %{x:.1f} min<br>dFlow/dt: %{y:.2f}<extra></extra>'
        ),
        row=4, col=1
    )

    # Update axes
    fig.update_xaxes(title_text="Flow Rate (L/s)", row=1, col=1)
    fig.update_yaxes(title_text="dFlow/dt (L/s²)", row=1, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    fig.update_yaxes(title_text="Derivative Range (L/s²)", row=2, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    fig.update_yaxes(title_text="Flow (L/s)", row=3, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=4, col=1)
    fig.update_yaxes(title_text="dFlow/dt (L/s²)", row=4, col=1)

    # Stats annotation
    stats_text = (
        f"<b>Overall Night Statistics:</b><br>"
        f"Total Derivative Range: {stats['overall_deriv_range']:.2f} L/s²<br>"
        f"Mean Window Range: {stats['mean_window_range']:.2f} L/s²<br>"
        f"Median Window Range: {stats['median_window_range']:.2f} L/s²<br>"
        f"<br>"
        f"<b>Worst Period:</b><br>"
        f"Time: {stats['worst_time_minutes']:.1f} min<br>"
        f"Range: {stats['worst_window_range']:.2f} L/s²<br>"
        f"<br>"
        f"<b>Best Period:</b><br>"
        f"Time: {stats['best_time_minutes']:.1f} min<br>"
        f"Range: {stats['best_window_range']:.2f} L/s²"
    )

    # Layout
    duration_hours = len(flow) / sample_rate / 3600
    fig.update_layout(
        title=f"Single Night Suffering Analysis: {filename}<br>" +
              f"<sub>Duration: {duration_hours:.1f} hours | " +
              f"Overall Suffering Score: {stats['overall_deriv_range']:.2f} L/s²</sub>",
        height=1400,
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
            'filename': f'{filename}_single_night_suffering',
            'height': 1400,
            'width': 1600,
            'scale': 2
        },
        'displaylogo': False
    }

    return fig, config

def main():
    print("=" * 70)
    print("OscilloBreath - Single Night Suffering Analysis")
    print("Deep dive into when suffering occurred during one night")
    print("=" * 70)
    print()

    # Select file
    edf_path = select_edf_file()
    if not edf_path:
        print("No file selected. Exiting.")
        return

    filename = Path(edf_path).stem

    # Read data
    flow, sample_rate = read_flow_data(edf_path)

    # Analyze
    print()
    flow_derivative, window_data, stats = analyze_suffering_windows(flow, sample_rate, window_minutes=5)

    print()
    print("=" * 70)
    print("SINGLE NIGHT ANALYSIS")
    print("=" * 70)
    print(f"Overall derivative range: {stats['overall_deriv_range']:.2f} L/s²")
    print(f"Mean windowed range: {stats['mean_window_range']:.2f} L/s²")
    print(f"Median windowed range: {stats['median_window_range']:.2f} L/s²")
    print()
    print(f"WORST period: {stats['worst_time_minutes']:.1f} min ({stats['worst_time_minutes']/60:.1f} hours)")
    print(f"  Derivative range: {stats['worst_window_range']:.2f} L/s²")
    print()
    print(f"BEST period: {stats['best_time_minutes']:.1f} min ({stats['best_time_minutes']/60:.1f} hours)")
    print(f"  Derivative range: {stats['best_window_range']:.2f} L/s²")
    print("=" * 70)
    print()

    # Create plot
    fig, config = create_single_night_plot(flow, sample_rate, flow_derivative,
                                           window_data, stats, filename)

    # Save
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{filename}_single_night_suffering.html"
    fig.write_html(str(output_file), config=config)

    print()
    print("=" * 70)
    print(f"Analysis complete!")
    print(f"Saved to: {output_file}")
    print("=" * 70)
    print()
    print("Opening in browser...")

    webbrowser.open(f'file://{output_file.absolute()}')

if __name__ == "__main__":
    main()
