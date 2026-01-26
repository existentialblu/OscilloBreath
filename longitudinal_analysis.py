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
        # Extract YYYYMMDD part
        date_str = filename.split('_')[0]
        date = datetime.strptime(date_str, '%Y%m%d')
        return date
    except:
        return None

def find_edf_files(folder_path):
    """Find all .edf files in folder and subdirectories, sort by date"""
    folder = Path(folder_path)
    # Use ** to search recursively in all subdirectories
    edf_files = list(folder.glob('**/*_BRP.edf'))

    # Sort by date extracted from filename
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

            # Find flow signal
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

def calculate_metrics(flow, sample_rate):
    """Calculate oscillator characterization metrics"""
    # Calculate derivative
    dt = 1.0 / sample_rate
    flow_derivative = np.gradient(flow, dt)

    # Basic statistics
    flow_std = np.std(flow)
    deriv_std = np.std(flow_derivative)
    flow_range = np.ptp(flow)  # peak-to-peak
    deriv_range = np.ptp(flow_derivative)

    # Phase space area estimate (using 95th percentile to avoid outliers)
    flow_95 = np.percentile(np.abs(flow - np.mean(flow)), 95)
    deriv_95 = np.percentile(np.abs(flow_derivative - np.mean(flow_derivative)), 95)
    phase_space_area = flow_95 * deriv_95

    # Attractor tightness: ratio of 50th to 95th percentile distances from center
    flow_centered = flow - np.mean(flow)
    deriv_centered = flow_derivative - np.mean(flow_derivative)
    distances = np.sqrt(flow_centered**2 + deriv_centered**2)
    p50 = np.percentile(distances, 50)
    p95 = np.percentile(distances, 95)
    tightness = p50 / p95 if p95 > 0 else 0

    # Aspect ratio: ratio of principal axis lengths
    # Use covariance matrix to find principal components
    cov_matrix = np.cov(flow, flow_derivative)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    aspect_ratio = np.sqrt(eigenvalues[1] / eigenvalues[0]) if eigenvalues[0] > 0 else 1

    return {
        'flow_std': flow_std,
        'deriv_std': deriv_std,
        'flow_range': flow_range,
        'deriv_range': deriv_range,
        'phase_space_area': phase_space_area,
        'tightness': tightness,
        'aspect_ratio': aspect_ratio
    }

def smooth_data(values, window_size=7):
    """Apply moving average smoothing"""
    if len(values) < window_size:
        window_size = max(3, len(values) // 2)

    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    # Pad to match original length
    pad_size = len(values) - len(smoothed)
    left_pad = pad_size // 2
    right_pad = pad_size - left_pad

    # Extend edges
    smoothed = np.concatenate([
        np.full(left_pad, smoothed[0]),
        smoothed,
        np.full(right_pad, smoothed[-1])
    ])

    return smoothed

def create_longitudinal_plot(dates, metrics_list):
    """Create multi-panel plot showing metrics over time"""

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Flow Variability (Breathing Amplitude)',
            'Derivative Variability (Change Violence)',
            'Phase Space Area (Attractor Size)',
            'Attractor Tightness (1=tight, 0=diffuse)',
            'Flow Range (Peak-to-Peak)',
            'Derivative Range (Max Change Rate)',
            'Aspect Ratio (Elongation)',
            'Summary: Stability Score'
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.12
    )

    # Extract metric arrays
    flow_stds = [m['flow_std'] for m in metrics_list]
    deriv_stds = [m['deriv_std'] for m in metrics_list]
    phase_areas = [m['phase_space_area'] for m in metrics_list]
    tightnesses = [m['tightness'] for m in metrics_list]
    flow_ranges = [m['flow_range'] for m in metrics_list]
    deriv_ranges = [m['deriv_range'] for m in metrics_list]
    aspect_ratios = [m['aspect_ratio'] for m in metrics_list]

    # Calculate stability score (lower is better)
    # Normalize and combine metrics
    stability_scores = []
    for i in range(len(metrics_list)):
        # Lower flow variability = better
        # Lower derivative variability = better
        # Higher tightness = better
        # Lower aspect ratio = better (closer to 1 is circular)
        score = (
            (1 - tightnesses[i]) * 0.3 +  # inverted tightness
            (flow_stds[i] / max(flow_stds)) * 0.3 +
            (deriv_stds[i] / max(deriv_stds)) * 0.2 +
            (abs(aspect_ratios[i] - 1) / max(abs(r - 1) for r in aspect_ratios)) * 0.2
        )
        stability_scores.append(score)

    # Plot each metric
    metrics_config = [
        (flow_stds, 1, 1, 'Flow Std (L/s)', 'blue'),
        (deriv_stds, 1, 2, 'Deriv Std (L/s²)', 'red'),
        (phase_areas, 2, 1, 'Area (L²/s³)', 'green'),
        (tightnesses, 2, 2, 'Tightness', 'purple'),
        (flow_ranges, 3, 1, 'Range (L/s)', 'navy'),
        (deriv_ranges, 3, 2, 'Range (L/s²)', 'darkred'),
        (aspect_ratios, 4, 1, 'Ratio', 'orange'),
        (stability_scores, 4, 2, 'Score (lower=better)', 'black')
    ]

    for values, row, col, ylabel, color in metrics_config:
        # Raw data points (lighter, smaller)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.4),
                name=ylabel + ' (raw)',
                hovertemplate='%{x|%Y-%m-%d}<br>' + ylabel + ': %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

        # Smoothed trend line (bold)
        smoothed = smooth_data(values)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=smoothed,
                mode='lines',
                line=dict(color=color, width=3),
                name=ylabel + ' (trend)',
                hovertemplate='%{x|%Y-%m-%d}<br>Smoothed: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_yaxes(title_text=ylabel, row=row, col=col)
        fig.update_xaxes(title_text="Date", row=row, col=col)

    fig.update_layout(
        title="OscilloBreath Longitudinal Analysis - Oscillator Metrics Over Time",
        height=1400,
        showlegend=False,
        hovermode='closest'
    )

    # Configure for screenshots
    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'oscillobreath_longitudinal',
            'height': 1400,
            'width': 1600,
            'scale': 2
        },
        'displaylogo': False
    }

    return fig, config

def main():
    print("=" * 70)
    print("OscilloBreath - Longitudinal Oscillator Analysis")
    print("=" * 70)
    print()

    # Get folder from user
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

    # Find all EDF files
    edf_files = find_edf_files(folder_path)

    if not edf_files:
        print("No valid EDF files found with date format YYYYMMDD_HHMMSS_BRP.edf")
        return

    print(f"Found {len(edf_files)} EDF files")
    print()

    # Process each file
    dates = []
    metrics_list = []
    processed_count = 0

    for date, filepath in edf_files:
        print(f"Processing {filepath.name}...")

        # Read flow data
        flow, sample_rate, duration_hours = read_flow_data(filepath)

        if flow is None:
            print(f"  Skipped (read error)")
            continue

        # Filter out sessions < 30 minutes
        if duration_hours < 0.5:
            print(f"  Skipped (only {duration_hours*60:.1f} minutes)")
            continue

        # Calculate metrics
        metrics = calculate_metrics(flow, sample_rate)

        dates.append(date)
        metrics_list.append(metrics)
        processed_count += 1

        print(f"  ✓ Processed ({duration_hours:.1f} hours)")
        print(f"    Flow std: {metrics['flow_std']:.3f}, Tightness: {metrics['tightness']:.3f}")

    print()
    print("=" * 70)
    print(f"Successfully processed {processed_count} nights")
    print("=" * 70)
    print()

    if processed_count == 0:
        print("No data to plot!")
        return

    # Create plot
    print("Creating longitudinal visualization...")
    fig, config = create_longitudinal_plot(dates, metrics_list)

    # Save to output folder
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "longitudinal_analysis.html"
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
