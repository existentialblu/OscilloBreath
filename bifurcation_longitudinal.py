"""
OscilloBreath - Longitudinal Bifurcation Analysis

Fast version for processing many nights. Extracts key bifurcation metrics
per night and outputs CSV + summary visualization for trend analysis.

Speed optimizations vs single-night analysis:
- 10-minute windows, no overlap (vs 5-min with 50% overlap)
- 100 LLE sample points (vs 200)
- 30 max iterations (vs 50)
- Skip within-night stable-chaos comparison (compare across nights instead)

Target: ~1 second per night instead of ~10 seconds
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tkinter import Tk, filedialog
from pathlib import Path
import webbrowser
import csv
from datetime import datetime, timedelta
from data_loader import load_flow_data
import pyedflib

# Import LLE computation
from lyapunov_analyzer_fast import time_delay_embedding


# =============================================================================
# Session Grouping - Concatenate multiple files from same night
# =============================================================================

def parse_filename_datetime(filename):
    """
    Parse datetime from ResMed filename format: YYYYMMDD_HHMMSS_BRP.edf
    Returns datetime or None if parsing fails.
    """
    try:
        stem = Path(filename).stem
        parts = stem.split('_')
        date_str = parts[0]
        time_str = parts[1]
        return datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
    except:
        return None


def group_files_into_nights(edf_files, max_gap_hours=8):
    """
    Group EDF files into nights based on timestamps.

    Files within max_gap_hours of each other are considered same night.
    This handles:
    - Multiple mask-off/mask-on events
    - Sessions spanning midnight

    Args:
        edf_files: list of Path objects to EDF files
        max_gap_hours: maximum gap between sessions to consider same night

    Returns:
        List of lists, each inner list contains files from one night,
        sorted by timestamp within each night.
    """
    # Parse timestamps and pair with files
    files_with_times = []
    for f in edf_files:
        dt = parse_filename_datetime(f.name)
        if dt:
            files_with_times.append((dt, f))

    if not files_with_times:
        return []

    # Sort by timestamp
    files_with_times.sort(key=lambda x: x[0])

    # Group into nights
    nights = []
    current_night = [files_with_times[0]]

    for i in range(1, len(files_with_times)):
        prev_dt = files_with_times[i-1][0]
        curr_dt = files_with_times[i][0]

        gap = curr_dt - prev_dt

        if gap <= timedelta(hours=max_gap_hours):
            # Same night
            current_night.append(files_with_times[i])
        else:
            # New night
            nights.append(current_night)
            current_night = [files_with_times[i]]

    # Don't forget the last night
    nights.append(current_night)

    # Convert to just file paths, keeping the first timestamp as night identifier
    result = []
    for night in nights:
        night_files = [f for (dt, f) in night]
        night_date = night[0][0]  # First session's datetime
        result.append({
            'date': night_date,
            'files': night_files
        })

    return result


def load_and_concatenate_night(night_info):
    """
    Load and concatenate all EDF files from a single night.

    Args:
        night_info: dict with 'date' and 'files' keys

    Returns:
        (flow_array, sample_rate, total_duration_hours) or (None, None, None) on failure
    """
    files = night_info['files']

    all_flow = []
    sample_rate = None

    for edf_path in files:
        try:
            # Suppress output during loading
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                flow, sr, _ = load_flow_data(edf_path)
            finally:
                sys.stdout = old_stdout

            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                # Skip mismatched files
                continue

            all_flow.append(flow)

        except Exception as e:
            continue

    if not all_flow or sample_rate is None:
        return None, None, None

    # Concatenate all sessions
    combined_flow = np.concatenate(all_flow)
    total_duration_hours = len(combined_flow) / sample_rate / 3600

    return combined_flow, sample_rate, total_duration_hours


def select_data_folder():
    """Open folder picker to select data directory"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    folder_path = filedialog.askdirectory(
        title="Select folder containing EDF files"
    )

    root.destroy()
    return folder_path


def calculate_lle_fast(signal, tau, embedding_dim, fs, max_iter=30, num_samples=100):
    """
    Stripped-down LLE calculation optimized for speed.
    Same algorithm as lyapunov_analyzer_fast but with reduced sampling.
    """
    try:
        embedded = time_delay_embedding(signal, tau, embedding_dim)
        N = len(embedded)

        if N < 200:
            return None

        min_separation = max(10, int(fs * 1.0))
        divergences = []

        # Fewer sample points for speed
        num_samples = min(num_samples, N // 10)
        if num_samples < 20:
            return None

        sample_indices = np.linspace(0, N - max_iter - 1, num_samples, dtype=int)

        for i in sample_indices:
            try:
                distances = np.sum((embedded - embedded[i])**2, axis=1)
                time_indices = np.arange(N)
                valid_mask = np.abs(time_indices - i) > min_separation
                distances[~valid_mask] = np.inf

                if np.all(np.isinf(distances)):
                    continue

                nearest_idx = np.argmin(distances)
                initial_distance = np.sqrt(distances[nearest_idx])

                if initial_distance > np.std(embedded) * 2:
                    continue

                divergence = []
                for j in range(max_iter):
                    if i + j >= N or nearest_idx + j >= N:
                        break
                    d = np.sqrt(np.sum((embedded[i + j] - embedded[nearest_idx + j])**2))
                    if d > initial_distance * 0.01:
                        divergence.append(np.log(d))
                    else:
                        break

                if len(divergence) >= 10:
                    divergences.append(divergence)

            except:
                continue

        if len(divergences) < 5:
            return None

        min_len = min(len(d) for d in divergences)
        if min_len < 5:
            return None

        divergences_array = np.array([d[:min_len] for d in divergences])
        mean_divergence = np.mean(divergences_array, axis=0)

        fit_start = max(2, int(min_len * 0.1))
        fit_end = min(min_len, int(min_len * 0.5))

        if fit_end - fit_start < 3:
            return None

        time_steps = np.arange(len(mean_divergence))
        coeffs = np.polyfit(time_steps[fit_start:fit_end], mean_divergence[fit_start:fit_end], 1)

        return coeffs[0] * fs

    except:
        return None


def compute_windowed_lle_fast(flow, sample_rate, window_seconds=600, verbose=False):
    """
    Fast windowed LLE computation.
    10-minute windows, no overlap, reduced LLE parameters.
    """
    # Downsample
    target_fs = 2.0
    downsample_factor = int(sample_rate / target_fs)
    flow_downsampled = flow[::downsample_factor]
    fs_downsampled = sample_rate / downsample_factor

    # Embedding parameters
    tau = int(fs_downsampled * 1.0)
    embedding_dim = 3

    # Window parameters - no overlap for speed
    window_samples = int(window_seconds * fs_downsampled)
    hop_samples = window_samples  # No overlap

    num_windows = len(flow_downsampled) // hop_samples

    lle_values = []
    window_centers_sec = []
    window_indices = []

    for i in range(num_windows):
        start_idx_ds = i * hop_samples
        end_idx_ds = start_idx_ds + window_samples

        if end_idx_ds > len(flow_downsampled):
            break

        window_flow = flow_downsampled[start_idx_ds:end_idx_ds]

        start_idx_orig = start_idx_ds * downsample_factor
        end_idx_orig = end_idx_ds * downsample_factor
        center_sec = (start_idx_orig + end_idx_orig) / 2 / sample_rate

        lle = calculate_lle_fast(window_flow, tau, embedding_dim, fs_downsampled)

        lle_values.append(lle if lle is not None else np.nan)
        window_centers_sec.append(center_sec)
        window_indices.append((start_idx_orig, min(end_idx_orig, len(flow))))

    return {
        'lle_values': np.array(lle_values),
        'window_centers_sec': np.array(window_centers_sec),
        'window_indices': window_indices
    }


def compute_reynolds_metrics(flow, sample_rate):
    """
    Compute all Reynolds candidate metrics for a window.
    Returns dict of metric values.
    """
    dt = 1.0 / sample_rate
    flow_deriv = np.gradient(flow, dt)
    flow_deriv2 = np.gradient(flow_deriv, dt)
    flow_deriv3 = np.gradient(flow_deriv2, dt)

    flow_std = np.std(flow)
    deriv_std = np.std(flow_deriv)
    deriv_range = np.ptp(flow_deriv)
    deriv_mean_abs = np.mean(np.abs(flow_deriv))
    accel_mean_abs = np.mean(np.abs(flow_deriv2))
    jerk_mean = np.mean(np.abs(flow_deriv3))

    eps = 1e-10

    # All candidate formulas
    violence = deriv_range / (deriv_std + eps)

    return {
        'derivative_violence': violence / (flow_std + eps),
        'acceleration_ratio': accel_mean_abs / (flow_std + eps),
        'energy_ratio': np.sum(flow_deriv**2) / (np.sum(flow**2) + eps),
        'jerk_ratio': jerk_mean / (flow_std + eps),
        'cv_ratio': (deriv_std / (deriv_mean_abs + eps)) / (flow_std / (np.mean(np.abs(flow)) + eps) + eps),
        'flow_std': flow_std,
        'deriv_std': deriv_std
    }


def analyze_night_data(flow, sample_rate, night_date, night_label, num_sessions=1):
    """
    Analyze pre-loaded flow data for a single night.

    Args:
        flow: concatenated flow array for the night
        sample_rate: Hz
        night_date: datetime of the night
        night_label: string label for the night (e.g., first filename)
        num_sessions: number of sessions concatenated

    Returns dict of metrics or None on failure.
    """
    try:
        duration_hours = len(flow) / sample_rate / 3600

        # Compute windowed LLE
        lle_data = compute_windowed_lle_fast(flow, sample_rate, window_seconds=600)

        valid_lle = lle_data['lle_values'][~np.isnan(lle_data['lle_values'])]
        if len(valid_lle) < 3:
            return None

        # Find minimum LLE (collapse point)
        # Skip first window (sleep onset weirdness)
        lle_values = lle_data['lle_values']
        times = lle_data['window_centers_sec']

        valid_mask = (times >= 600) & ~np.isnan(lle_values)  # Skip first 10 min

        if not np.any(valid_mask):
            valid_mask = ~np.isnan(lle_values)

        if not np.any(valid_mask):
            return None

        valid_indices = np.where(valid_mask)[0]
        min_valid_idx = np.argmin(lle_values[valid_mask])
        min_idx = valid_indices[min_valid_idx]

        min_lle = lle_values[min_idx]
        collapse_time_min = times[min_idx] / 60

        # Get pre-collapse window for Reynolds
        pre_collapse_idx = max(0, min_idx - 1)
        start, end = lle_data['window_indices'][pre_collapse_idx]
        pre_collapse_flow = flow[start:end]

        # Compute Reynolds metrics
        reynolds = compute_reynolds_metrics(pre_collapse_flow, sample_rate)

        # Overall night LLE (use median of all windows)
        overall_lle = np.nanmedian(lle_values)
        lle_std = np.nanstd(lle_values)
        max_lle = np.nanmax(lle_values)

        # LLE range (max - min) indicates how much the system varies
        lle_range = max_lle - min_lle

        return {
            'filename': night_label,
            'date': night_date,
            'duration_hours': duration_hours,
            'num_sessions': num_sessions,
            'min_lle': min_lle,
            'collapse_time_min': collapse_time_min,
            'overall_lle': overall_lle,
            'max_lle': max_lle,
            'lle_range': lle_range,
            'lle_std': lle_std,
            **reynolds
        }

    except Exception as e:
        return None


def process_folder(folder_path, min_duration_minutes=30, progress_callback=None):
    """
    Process all EDF files in folder, grouping by night and concatenating sessions.

    Args:
        folder_path: path to folder containing EDF files
        min_duration_minutes: discard nights with less than this total duration
        progress_callback: optional callback(current, total, label)

    Returns list of result dicts.
    """
    folder = Path(folder_path)

    # Find all EDF files with _BRP in name (ResMed flow files)
    # Search recursively - ResMed stores in nested date folders
    edf_files = list(folder.rglob('*_BRP*.edf'))

    if not edf_files:
        # Try without _BRP filter, still recursive
        edf_files = list(folder.rglob('*.edf'))

    print(f"Found {len(edf_files)} EDF files")

    # Group files into nights
    nights = group_files_into_nights(edf_files, max_gap_hours=8)
    print(f"Grouped into {len(nights)} nights")

    results = []
    skipped_short = 0
    skipped_error = 0

    for i, night_info in enumerate(nights):
        night_date = night_info['date']
        night_files = night_info['files']
        night_label = night_files[0].stem  # Use first file as label

        if progress_callback:
            progress_callback(i, len(nights), night_label)
        else:
            sessions_str = f"({len(night_files)} sessions)" if len(night_files) > 1 else ""
            print(f"  [{i+1}/{len(nights)}] {night_date.strftime('%Y-%m-%d')} {sessions_str}...", end=' ', flush=True)

        # Load and concatenate all sessions for this night
        flow, sample_rate, duration_hours = load_and_concatenate_night(night_info)

        if flow is None:
            if not progress_callback:
                print("LOAD ERROR")
            skipped_error += 1
            continue

        duration_minutes = duration_hours * 60

        if duration_minutes < min_duration_minutes:
            if not progress_callback:
                print(f"SHORT ({duration_minutes:.0f} min)")
            skipped_short += 1
            continue

        # Analyze the concatenated night
        result = analyze_night_data(
            flow, sample_rate,
            night_date, night_label,
            num_sessions=len(night_files)
        )

        if result:
            results.append(result)
            if not progress_callback:
                print(f"OK ({duration_hours:.1f}h, LLE: {result['min_lle']:.3f})")
        else:
            if not progress_callback:
                print("ANALYSIS ERROR")
            skipped_error += 1

    print()
    print(f"Summary: {len(results)} nights processed, {skipped_short} skipped (too short), {skipped_error} errors")

    return results


def save_results_csv(results, output_path):
    """Save results to CSV file."""
    if not results:
        return

    fieldnames = [
        'filename', 'date', 'duration_hours', 'num_sessions',
        'min_lle', 'collapse_time_min', 'overall_lle', 'max_lle', 'lle_range', 'lle_std',
        'derivative_violence', 'acceleration_ratio', 'energy_ratio',
        'jerk_ratio', 'cv_ratio', 'flow_std', 'deriv_std'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            if row['date']:
                row['date'] = row['date'].strftime('%Y-%m-%d')
            writer.writerow(row)


def create_longitudinal_plot(results, output_path):
    """Create longitudinal visualization."""

    # Sort by date
    results_sorted = sorted([r for r in results if r['date']], key=lambda x: x['date'])

    if len(results_sorted) < 2:
        print("Not enough dated results for longitudinal plot")
        return None

    dates = [r['date'] for r in results_sorted]

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Minimum LLE Over Time (lower = more periodic at worst)',
            'Overall LLE (median across night)',
            'Collapse Timing (minutes into night)',
            'LLE Range (max - min, higher = more variable)',
            'Derivative Violence (Reynolds candidate)',
            'Energy Ratio (Reynolds candidate)',
            'CV Ratio (Reynolds candidate)',
            'Jerk Ratio (Reynolds candidate)'
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )

    # Helper for adding traces with smoothing
    def add_metric_trace(row, col, values, name, color):
        fig.add_trace(
            go.Scatter(
                x=dates, y=values,
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.5),
                name=name,
                showlegend=False
            ),
            row=row, col=col
        )

        # Add rolling average if enough points
        if len(values) >= 7:
            rolling = np.convolve(values, np.ones(7)/7, mode='valid')
            rolling_dates = dates[3:-3]
            fig.add_trace(
                go.Scatter(
                    x=rolling_dates, y=rolling,
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f'{name} (7-day avg)',
                    showlegend=False
                ),
                row=row, col=col
            )

    # Row 1: LLE metrics
    add_metric_trace(1, 1, [r['min_lle'] for r in results_sorted], 'Min LLE', 'red')
    add_metric_trace(1, 2, [r['overall_lle'] for r in results_sorted], 'Overall LLE', 'blue')

    # Row 2: Collapse timing and LLE range
    add_metric_trace(2, 1, [r['collapse_time_min'] for r in results_sorted], 'Collapse Time', 'orange')
    add_metric_trace(2, 2, [r['lle_range'] for r in results_sorted], 'LLE Range', 'purple')

    # Row 3-4: Reynolds candidates
    add_metric_trace(3, 1, [r['derivative_violence'] for r in results_sorted], 'Deriv Violence', 'green')
    add_metric_trace(3, 2, [r['energy_ratio'] for r in results_sorted], 'Energy Ratio', 'teal')
    add_metric_trace(4, 1, [r['cv_ratio'] for r in results_sorted], 'CV Ratio', 'magenta')
    add_metric_trace(4, 2, [r['jerk_ratio'] for r in results_sorted], 'Jerk Ratio', 'brown')

    # Update layout
    fig.update_layout(
        title=f"Longitudinal Bifurcation Analysis<br><sub>{len(results_sorted)} nights from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}</sub>",
        height=1400,
        showlegend=False
    )

    # Add zero line to LLE plots
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'bifurcation_longitudinal',
            'height': 1400,
            'width': 1600,
            'scale': 2
        },
        'displaylogo': False
    }

    fig.write_html(str(output_path), config=config)
    return fig


def create_correlation_plot(results, output_path):
    """
    Create correlation matrix between metrics.
    This helps identify which Reynolds candidates track with LLE.
    """
    results_valid = [r for r in results if r['date']]

    if len(results_valid) < 10:
        return None

    metrics = ['min_lle', 'overall_lle', 'lle_range', 'collapse_time_min',
               'derivative_violence', 'acceleration_ratio', 'energy_ratio',
               'jerk_ratio', 'cv_ratio']

    # Build correlation matrix
    data = np.array([[r[m] for m in metrics] for r in results_valid])

    # Handle any NaN/inf
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

    corr = np.corrcoef(data.T)

    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=metrics,
        y=metrics,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title='Metric Correlations<br><sub>Which Reynolds candidates track with LLE?</sub>',
        height=700,
        width=900
    )

    config = {'displaylogo': False}
    fig.write_html(str(output_path), config=config)

    return corr, metrics


def print_summary_stats(results):
    """Print summary statistics."""
    if not results:
        print("No results to summarize")
        return

    print()
    print("=" * 70)
    print("LONGITUDINAL SUMMARY")
    print("=" * 70)
    print(f"Total nights analyzed: {len(results)}")

    dated = [r for r in results if r['date']]
    if dated:
        dates = [r['date'] for r in dated]
        print(f"Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")

    # Session info
    multi_session = [r for r in results if r.get('num_sessions', 1) > 1]
    if multi_session:
        print(f"Nights with multiple sessions: {len(multi_session)}")

    durations = [r['duration_hours'] for r in results]
    print(f"Duration: mean={np.mean(durations):.1f}h, range=[{np.min(durations):.1f}h, {np.max(durations):.1f}h]")

    print()
    print("LLE Statistics:")
    min_lles = [r['min_lle'] for r in results]
    overall_lles = [r['overall_lle'] for r in results]
    print(f"  Min LLE:     mean={np.mean(min_lles):.4f}, std={np.std(min_lles):.4f}, range=[{np.min(min_lles):.4f}, {np.max(min_lles):.4f}]")
    print(f"  Overall LLE: mean={np.mean(overall_lles):.4f}, std={np.std(overall_lles):.4f}")

    collapse_times = [r['collapse_time_min'] for r in results]
    print(f"  Collapse time: mean={np.mean(collapse_times):.1f} min, std={np.std(collapse_times):.1f}")

    print()
    print("Reynolds Candidates (pre-collapse window):")
    for metric in ['derivative_violence', 'acceleration_ratio', 'energy_ratio', 'jerk_ratio', 'cv_ratio']:
        values = [r[metric] for r in results]
        print(f"  {metric:22s}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    # Correlations with min_lle
    print()
    print("Correlations with Min LLE (most periodic moment):")
    for metric in ['derivative_violence', 'acceleration_ratio', 'energy_ratio', 'jerk_ratio', 'cv_ratio']:
        values = [r[metric] for r in results]
        corr = np.corrcoef(min_lles, values)[0, 1]
        print(f"  {metric:22s}: r = {corr:+.3f}")

    print()
    print("=" * 70)


def main():
    print("=" * 70)
    print("OscilloBreath - Longitudinal Bifurcation Analysis")
    print("Fast batch processing for trend analysis")
    print("=" * 70)
    print()
    print("This will process all EDF files in a folder and output:")
    print("  - CSV with all metrics per night")
    print("  - Longitudinal trend visualization")
    print("  - Correlation analysis between metrics")
    print()
    print("Optimized for speed: ~1 second per night")
    print()

    # Select folder
    folder_path = select_data_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    print(f"Processing: {folder_path}")
    print()

    import time
    start_time = time.time()

    # Process all files
    results = process_folder(folder_path)

    elapsed = time.time() - start_time

    if not results:
        print("No valid results. Check that folder contains ResMed EDF files.")
        return

    print()
    print(f"Processed {len(results)} nights in {elapsed:.1f} seconds ({elapsed/len(results):.2f} s/night)")

    # Print summary
    print_summary_stats(results)

    # Save outputs
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # CSV
    csv_path = output_dir / "bifurcation_longitudinal.csv"
    save_results_csv(results, csv_path)
    print(f"\nSaved CSV: {csv_path}")

    # Longitudinal plot
    plot_path = output_dir / "bifurcation_longitudinal.html"
    create_longitudinal_plot(results, plot_path)
    print(f"Saved plot: {plot_path}")

    # Correlation plot
    corr_path = output_dir / "bifurcation_correlations.html"
    corr_result = create_correlation_plot(results, corr_path)
    if corr_result:
        print(f"Saved correlations: {corr_path}")

    print()
    print("Opening visualization...")
    webbrowser.open(f'file://{plot_path.absolute()}')


if __name__ == "__main__":
    main()
