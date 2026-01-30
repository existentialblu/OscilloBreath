import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import webbrowser
from tkinter import Tk, filedialog
from data_loader import load_flow_data

def select_folder():
    """Open folder picker for longitudinal data"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    folder_path = filedialog.askdirectory(
        title="Select folder containing multiple nights of data"
    )

    root.destroy()
    return folder_path

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
    """Find all .edf files in folder and subdirectories, group by date"""
    folder = Path(folder_path)
    # Use ** to search recursively in all subdirectories
    edf_files = list(folder.glob('**/*_BRP.edf'))

    # Group by date
    from collections import defaultdict
    date_groups = defaultdict(list)

    for f in edf_files:
        date = parse_date_from_filename(f.stem)
        if date:
            date_groups[date].append(f)

    # Sort dates and return as list of (date, [files]) tuples
    sorted_dates = sorted(date_groups.items(), key=lambda x: x[0])
    return sorted_dates

def time_delay_embedding(signal, tau, embedding_dim):
    """Create time-delay embedded phase space reconstruction"""
    N = len(signal)
    M = N - (embedding_dim - 1) * tau

    if M <= 0:
        raise ValueError(f"Signal too short for embedding: N={N}, need at least {(embedding_dim-1)*tau}")

    embedded = np.zeros((M, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = signal[i * tau : i * tau + M]

    return embedded

def calculate_lle_rosenstein(signal, tau, embedding_dim, fs, max_iter=100):
    """
    Calculate Largest Lyapunov Exponent using simplified Rosenstein method
    """
    try:
        embedded = time_delay_embedding(signal, tau, embedding_dim)
        N = len(embedded)

        if N < 500:
            print(f"    WARNING: Only {N} points, results may be unreliable")
            return None

        # Minimum temporal separation (avoid temporally correlated points)
        min_separation = max(10, int(fs * 1.0))  # At least 1 second apart

        divergences = []

        # Sample fewer points for speed
        num_samples = min(200, N // 10)
        sample_indices = np.linspace(0, N - max_iter - 1, num_samples, dtype=int)

        for idx, i in enumerate(sample_indices):
            try:
                # Find nearest neighbor (excluding temporal neighbors)
                distances = np.sum((embedded - embedded[i])**2, axis=1)

                # Mask out temporally close points
                time_indices = np.arange(N)
                valid_mask = np.abs(time_indices - i) > min_separation
                distances[~valid_mask] = np.inf

                if np.all(np.isinf(distances)):
                    continue

                nearest_idx = np.argmin(distances)
                initial_distance = np.sqrt(distances[nearest_idx])

                # Skip if nearest neighbor is too far (not really a neighbor)
                if initial_distance > np.std(embedded) * 2:
                    continue

                # Track divergence
                divergence = []
                for j in range(max_iter):
                    if i + j >= N or nearest_idx + j >= N:
                        break

                    d = np.sqrt(np.sum((embedded[i + j] - embedded[nearest_idx + j])**2))

                    # Avoid log(0) and very small values
                    if d > initial_distance * 0.01:
                        divergence.append(np.log(d))
                    else:
                        break

                if len(divergence) >= 20:  # Need reasonable length
                    divergences.append(divergence)

            except Exception as e:
                continue

        if len(divergences) < 10:
            return None

        # Average divergence curves
        min_len = min(len(d) for d in divergences)
        if min_len < 10:
            return None

        divergences_array = np.array([d[:min_len] for d in divergences])
        mean_divergence = np.mean(divergences_array, axis=0)

        # Fit linear region
        # Use middle portion (early part often has transients)
        fit_start = max(5, int(min_len * 0.1))
        fit_end = min(min_len, int(min_len * 0.5))

        if fit_end - fit_start < 5:
            return None

        time_steps = np.arange(len(mean_divergence))
        coeffs = np.polyfit(time_steps[fit_start:fit_end], mean_divergence[fit_start:fit_end], 1)

        # LLE in bits per second
        lle = coeffs[0] * fs

        return lle

    except Exception as e:
        return None

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

def process_night(file_paths, date):
    """Process a single night (concatenating multiple sessions) and return LLE"""
    print(f"\nProcessing {date.strftime('%Y-%m-%d')} ({len(file_paths)} session(s))...")

    try:
        # Load and concatenate all sessions from this night
        all_flow = []
        sample_rate = None

        for file_path in file_paths:
            print(f"  Loading session: {file_path.name}")
            flow, fs, data_type = load_flow_data(str(file_path))

            if flow is None:
                print(f"    Failed to load")
                continue

            if sample_rate is None:
                sample_rate = fs
            elif sample_rate != fs:
                print(f"    WARNING: Sample rate mismatch ({fs} vs {sample_rate}), skipping")
                continue

            all_flow.append(flow)

        if not all_flow:
            print(f"  No valid data loaded")
            return None

        # Concatenate all sessions
        flow = np.concatenate(all_flow)
        duration_hours = len(flow) / sample_rate / 3600
        print(f"  Total duration: {duration_hours:.1f} hours ({len(all_flow)} session(s) concatenated)")

        # Downsample heavily for LLE analysis
        target_fs = 2.0  # 2 Hz target
        downsample_factor = int(sample_rate / target_fs)

        print(f"  Downsampling from {sample_rate} Hz to {sample_rate/downsample_factor:.1f} Hz...")
        flow_downsampled = flow[::downsample_factor]
        fs_downsampled = sample_rate / downsample_factor

        # Embedding parameters
        tau = int(fs_downsampled * 1.0)  # 1 second at downsampled rate
        embedding_dim = 3

        print(f"  Calculating LLE...")
        lle = calculate_lle_rosenstein(
            flow_downsampled,
            tau,
            embedding_dim,
            fs_downsampled,
            max_iter=100
        )

        if lle is not None:
            print(f"  LLE: {lle:.4f} bits/second")
        else:
            print(f"  Failed to calculate LLE")

        return lle

    except Exception as e:
        print(f"  Error: {e}")
        return None

def create_longitudinal_plot(dates, lle_values, output_path, log_scale=False):
    """Create longitudinal plot of LLE over time"""

    # Convert dates to strings for plotting
    date_strings = [d.strftime('%Y-%m-%d') for d in dates]

    # Smooth the LLE values
    lle_smoothed = smooth_data(lle_values, window_size=7)

    # Calculate statistics
    mean_lle = np.mean(lle_values)
    median_lle = np.median(lle_values)
    std_lle = np.std(lle_values)

    # Calculate order of magnitude range for log scale decision
    lle_abs = np.abs(lle_values)
    lle_abs = lle_abs[lle_abs > 0]  # Filter out zeros
    if len(lle_abs) > 0:
        oom_range = np.log10(np.max(lle_abs)) - np.log10(np.min(lle_abs))
        print(f"\nLLE spans {oom_range:.1f} orders of magnitude")
        if oom_range > 2 and not log_scale:
            print("  Consider using log scale for better visualization")
    else:
        oom_range = 0

    # Create color array based on LLE values
    # Positive = red (chaotic), Near-zero = yellow (periodic), Negative = green (stable)
    colors = []
    for lle in lle_values:
        if lle > 0.01:
            colors.append('rgba(220, 50, 50, 0.7)')  # Red - chaotic
        elif lle < -0.01:
            colors.append('rgba(50, 200, 50, 0.7)')  # Green - stable
        else:
            colors.append('rgba(220, 180, 50, 0.7)')  # Yellow - periodic

    # Create figure
    fig = go.Figure()

    # Scatter plot with colors
    fig.add_trace(go.Scatter(
        x=date_strings,
        y=lle_values,
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            line=dict(color='white', width=1)
        ),
        name='Nightly LLE',
        hovertemplate='%{x}<br>LLE: %{y:.4f} bits/s<extra></extra>'
    ))

    # Smoothed trend line
    fig.add_trace(go.Scatter(
        x=date_strings,
        y=lle_smoothed,
        mode='lines',
        line=dict(color='rgba(100, 100, 255, 0.8)', width=3),
        name='Smoothed Trend',
        hovertemplate='%{x}<br>Smoothed: %{y:.4f} bits/s<extra></extra>'
    ))

    # Mean line
    fig.add_hline(
        y=mean_lle,
        line=dict(color='gray', dash='dash', width=2),
        annotation_text=f"Mean: {mean_lle:.4f}",
        annotation_position="right"
    )

    # Zero line
    fig.add_hline(
        y=0,
        line=dict(color='white', dash='dot', width=1),
        annotation_text="Zero (periodic)",
        annotation_position="left"
    )

    # +0.01 and -0.01 reference lines
    fig.add_hline(
        y=0.01,
        line=dict(color='rgba(220, 50, 50, 0.3)', dash='dot', width=1),
        annotation_text="Chaotic threshold",
        annotation_position="left"
    )

    fig.add_hline(
        y=-0.01,
        line=dict(color='rgba(50, 200, 50, 0.3)', dash='dot', width=1),
        annotation_text="Stable threshold",
        annotation_position="left"
    )

    # Y-axis configuration
    yaxis_config = dict(
        title="Largest Lyapunov Exponent (bits/second)" + (" [LOG SCALE]" if log_scale else ""),
        zeroline=True,
        zerolinecolor='rgba(255,255,255,0.3)',
        zerolinewidth=2
    )

    if log_scale:
        # Use symlog for values that can be negative
        # This creates a log scale for large values but linear near zero
        yaxis_config['type'] = 'log'
        yaxis_config['title'] = "Largest Lyapunov Exponent (bits/second) [LOG SCALE]"

    # Layout
    fig.update_layout(
        title=f"Longitudinal Lyapunov Exponent Analysis<br>" +
              f"<sub>Mean: {mean_lle:.4f} bits/s | Median: {median_lle:.4f} | Std: {std_lle:.4f} | N={len(dates)} nights | " +
              f"OOM range: {oom_range:.1f}</sub>",
        xaxis=dict(
            title="Date",
            tickangle=-45
        ),
        yaxis=yaxis_config,
        height=700,
        hovermode='closest',
        template='plotly_dark',
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(0,0,0,0.5)'
        )
    )

    # Configure PNG export with descriptive filename
    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'lyapunov_longitudinal_{dates[0].strftime("%Y%m%d")}-{dates[-1].strftime("%Y%m%d")}_mean{mean_lle:.3f}',
            'height': 700,
            'width': 1400,
            'scale': 2
        },
        'displaylogo': False
    }

    # Save
    fig.write_html(str(output_path), config=config)
    print(f"\nPlot saved to: {output_path}")

    return fig

def main():
    print("=" * 70)
    print("OscilloBreath - Longitudinal Lyapunov Exponent Analysis")
    print("Track chaos vs stability over time")
    print("=" * 70)
    print()

    # Select folder
    folder_path = select_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    # Find all EDF files
    print(f"\nScanning folder: {folder_path}")
    file_list = find_edf_files(folder_path)

    if not file_list:
        print("No valid EDF files found!")
        return

    print(f"Found {len(file_list)} files")

    # Process each night (may have multiple sessions)
    dates = []
    lle_values = []

    for date, file_paths in file_list:
        lle = process_night(file_paths, date)

        if lle is not None:
            dates.append(date)
            lle_values.append(lle)

    if len(dates) == 0:
        print("\nNo valid data could be processed!")
        return

    print()
    print("=" * 70)
    print(f"Successfully processed {len(dates)} nights")
    print("=" * 70)

    # Create output
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "lyapunov_longitudinal.html"

    # Determine if log scale is appropriate
    lle_abs = np.abs(lle_values)
    lle_abs = lle_abs[lle_abs > 0]
    if len(lle_abs) > 0:
        oom_range = np.log10(np.max(lle_abs)) - np.log10(np.min(lle_abs))
        use_log = oom_range > 2.0  # Auto-enable log scale if >2 OOM range
    else:
        use_log = False

    # Create plot
    print("\nCreating longitudinal plot...")
    if use_log:
        print("Using logarithmic scale (OOM range > 2)")
    fig = create_longitudinal_plot(dates, lle_values, output_file, log_scale=use_log)

    # Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total nights analyzed: {len(dates)}")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print()
    print(f"Mean LLE: {np.mean(lle_values):.4f} bits/second")
    print(f"Median LLE: {np.median(lle_values):.4f} bits/second")
    print(f"Std Dev: {np.std(lle_values):.4f} bits/second")
    print(f"Min: {np.min(lle_values):.4f} bits/second")
    print(f"Max: {np.max(lle_values):.4f} bits/second")
    print()

    # Categorize nights
    chaotic = sum(1 for lle in lle_values if lle > 0.01)
    stable = sum(1 for lle in lle_values if lle < -0.01)
    periodic = len(lle_values) - chaotic - stable

    print(f"Night categories:")
    print(f"  Chaotic (LLE > 0.01): {chaotic} nights ({100*chaotic/len(lle_values):.1f}%)")
    print(f"  Periodic (|LLE| < 0.01): {periodic} nights ({100*periodic/len(lle_values):.1f}%)")
    print(f"  Stable (LLE < -0.01): {stable} nights ({100*stable/len(lle_values):.1f}%)")
    print("=" * 70)

    # Open in browser
    print("\nOpening in browser...")
    webbrowser.open(f'file://{output_file.absolute()}')

if __name__ == "__main__":
    main()
