"""
OscilloBreath - Longitudinal Wavelet Analysis

Track wavelet-derived stability metrics across multiple nights.
Candidate metrics for identifying best single-number summary:

1. dominant_freq_std - How much breathing frequency jumps around
2. vlf_power_mean - Mean power in periodic breathing range (0.01-0.04 Hz)
3. vlf_resp_ratio - Ratio of VLF to respiratory band power
4. band_power_spikes - Count of band power threshold crossings
"""

import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
from datetime import datetime, timedelta
import csv
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_flow_data
from wavelet_analyzer import (
    compute_wavelet_transform,
    compute_band_power,
    extract_dominant_frequency
)


def parse_filename_datetime(filename):
    """Parse datetime from ResMed filename format: YYYYMMDD_HHMMSS_BRP.edf"""
    try:
        stem = Path(filename).stem
        parts = stem.split('_')
        date_str = parts[0]
        time_str = parts[1]
        return datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
    except:
        return None


def group_files_into_nights(edf_files, max_gap_hours=8):
    """Group EDF files into nights based on timestamps."""
    files_with_times = []
    for f in edf_files:
        dt = parse_filename_datetime(f.name)
        if dt:
            files_with_times.append((dt, f))

    if not files_with_times:
        return []

    files_with_times.sort(key=lambda x: x[0])

    nights = []
    current_night = [files_with_times[0]]

    for i in range(1, len(files_with_times)):
        prev_dt = files_with_times[i-1][0]
        curr_dt = files_with_times[i][0]
        gap = curr_dt - prev_dt

        if gap <= timedelta(hours=max_gap_hours):
            current_night.append(files_with_times[i])
        else:
            nights.append(current_night)
            current_night = [files_with_times[i]]

    nights.append(current_night)

    result = []
    for night in nights:
        night_files = [f for (dt, f) in night]
        night_date = night[0][0]
        result.append({
            'date': night_date,
            'files': night_files
        })

    return result


def load_and_concatenate_night(night_info):
    """Load and concatenate all EDF files from a single night."""
    files = night_info['files']

    all_flow = []
    sample_rate = None

    for edf_path in files:
        try:
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
                continue

            all_flow.append(flow)

        except Exception as e:
            continue

    if not all_flow or sample_rate is None:
        return None, None, None

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


def compute_wavelet_metrics(flow, sample_rate):
    """
    Compute all candidate wavelet metrics for a single night.

    Returns dict of metrics or None on failure.
    """
    try:
        # Compute wavelet transform (faster settings for longitudinal)
        wavelet_data = compute_wavelet_transform(
            flow, sample_rate,
            freq_min=0.01,
            freq_max=0.5,
            num_freqs=50  # Fewer frequencies for speed
        )

        # Extract band powers
        band_powers = compute_band_power(
            wavelet_data['power'],
            wavelet_data['frequencies'],
            wavelet_data['times']
        )

        # Extract dominant frequency over time
        dominant_times, dominant_freqs = extract_dominant_frequency(
            wavelet_data['power'],
            wavelet_data['frequencies'],
            wavelet_data['times'],
            window_minutes=5
        )

        # Metric 1: Dominant frequency standard deviation
        dominant_freq_std = np.std(dominant_freqs)
        dominant_freq_mean = np.mean(dominant_freqs)

        # Metric 2: Mean VLF band power
        vlf_power = band_powers.get('VLF (PB range)', np.array([0]))
        vlf_power_mean = np.mean(vlf_power)
        vlf_power_max = np.max(vlf_power)

        # Metric 3: VLF/RESP ratio
        resp_power = band_powers.get('RESP (breathing)', np.array([1]))
        resp_power_mean = np.mean(resp_power)

        if resp_power_mean > 0:
            vlf_resp_ratio = vlf_power_mean / resp_power_mean
        else:
            vlf_resp_ratio = 0

        # Metric 4: Band power spike count
        # Count peaks in total band power above threshold
        total_band_power = sum(bp for bp in band_powers.values())

        # Threshold: mean + 2*std
        threshold = np.mean(total_band_power) + 2 * np.std(total_band_power)

        # Count threshold crossings (rising edges)
        above_threshold = total_band_power > threshold
        crossings = np.diff(above_threshold.astype(int))
        band_power_spikes = np.sum(crossings == 1)

        # Also compute LF power for completeness
        lf_power = band_powers.get('LF (slow mod)', np.array([0]))
        lf_power_mean = np.mean(lf_power)

        return {
            'dominant_freq_std': dominant_freq_std,
            'dominant_freq_mean': dominant_freq_mean,
            'vlf_power_mean': vlf_power_mean,
            'vlf_power_max': vlf_power_max,
            'lf_power_mean': lf_power_mean,
            'resp_power_mean': resp_power_mean,
            'vlf_resp_ratio': vlf_resp_ratio,
            'band_power_spikes': band_power_spikes
        }

    except Exception as e:
        print(f"    Error computing wavelet metrics: {e}")
        return None


def analyze_night(night_info, min_duration_minutes=30):
    """Analyze a single night's wavelet metrics."""
    night_date = night_info['date']
    night_files = night_info['files']
    night_label = night_files[0].stem

    # Load and concatenate
    flow, sample_rate, duration_hours = load_and_concatenate_night(night_info)

    if flow is None:
        return None, "LOAD ERROR"

    duration_minutes = duration_hours * 60

    if duration_minutes < min_duration_minutes:
        return None, f"SHORT ({duration_minutes:.0f} min)"

    # Compute wavelet metrics
    metrics = compute_wavelet_metrics(flow, sample_rate)

    if metrics is None:
        return None, "ANALYSIS ERROR"

    result = {
        'filename': night_label,
        'date': night_date,
        'duration_hours': duration_hours,
        'num_sessions': len(night_files),
        **metrics
    }

    return result, "OK"


def process_folder(folder_path, min_duration_minutes=30):
    """Process all EDF files in folder."""
    folder = Path(folder_path)

    edf_files = list(folder.rglob('*_BRP*.edf'))
    if not edf_files:
        edf_files = list(folder.rglob('*.edf'))

    print(f"Found {len(edf_files)} EDF files")

    nights = group_files_into_nights(edf_files, max_gap_hours=8)
    print(f"Grouped into {len(nights)} nights")

    results = []
    skipped_short = 0
    skipped_error = 0

    for i, night_info in enumerate(nights):
        night_date = night_info['date']
        night_files = night_info['files']

        sessions_str = f"({len(night_files)} sessions)" if len(night_files) > 1 else ""
        print(f"  [{i+1}/{len(nights)}] {night_date.strftime('%Y-%m-%d')} {sessions_str}...", end=' ', flush=True)

        result, status = analyze_night(night_info, min_duration_minutes)

        if result:
            results.append(result)
            print(f"{status} (std={result['dominant_freq_std']:.4f})")
        else:
            print(status)
            if "SHORT" in status:
                skipped_short += 1
            else:
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
        'dominant_freq_std', 'dominant_freq_mean',
        'vlf_power_mean', 'vlf_power_max', 'lf_power_mean',
        'resp_power_mean', 'vlf_resp_ratio', 'band_power_spikes'
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
    """Create longitudinal visualization of wavelet metrics."""

    results_sorted = sorted([r for r in results if r['date']], key=lambda x: x['date'])

    if len(results_sorted) < 2:
        print("Not enough dated results for longitudinal plot")
        return None

    dates = [r['date'] for r in results_sorted]

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Dominant Freq Std (lower = more stable)',
            'Dominant Freq Mean',
            'VLF Power Mean (periodic breathing)',
            'VLF/RESP Ratio',
            'Band Power Spikes (count)',
            'LF Power Mean',
            'RESP Power Mean',
            'Duration (hours)'
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )

    def add_metric_trace(row, col, values, name, color):
        fig.add_trace(
            go.Scatter(
                x=dates, y=values,
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.3),
                name=name,
                showlegend=False
            ),
            row=row, col=col
        )

        y_range = None
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

            rolling_min = np.min(rolling)
            rolling_max = np.max(rolling)
            padding = (rolling_max - rolling_min) * 0.3
            if padding > 0:
                y_range = [max(0, rolling_min - padding), rolling_max + padding]

        return row, col, y_range

    y_ranges = {}

    # Row 1: Dominant frequency metrics
    y_ranges[(1, 1)] = add_metric_trace(1, 1, [r['dominant_freq_std'] for r in results_sorted], 'Dom Freq Std', 'red')[2]
    y_ranges[(1, 2)] = add_metric_trace(1, 2, [r['dominant_freq_mean'] for r in results_sorted], 'Dom Freq Mean', 'blue')[2]

    # Row 2: VLF metrics
    y_ranges[(2, 1)] = add_metric_trace(2, 1, [r['vlf_power_mean'] for r in results_sorted], 'VLF Power', 'orange')[2]
    y_ranges[(2, 2)] = add_metric_trace(2, 2, [r['vlf_resp_ratio'] for r in results_sorted], 'VLF/RESP', 'darkorange')[2]

    # Row 3: Spike count and LF
    y_ranges[(3, 1)] = add_metric_trace(3, 1, [r['band_power_spikes'] for r in results_sorted], 'Spikes', 'purple')[2]
    y_ranges[(3, 2)] = add_metric_trace(3, 2, [r['lf_power_mean'] for r in results_sorted], 'LF Power', 'green')[2]

    # Row 4: RESP and duration
    y_ranges[(4, 1)] = add_metric_trace(4, 1, [r['resp_power_mean'] for r in results_sorted], 'RESP Power', 'teal')[2]
    y_ranges[(4, 2)] = add_metric_trace(4, 2, [r['duration_hours'] for r in results_sorted], 'Duration', 'gray')[2]

    # Apply y-axis ranges
    for (row, col), y_range in y_ranges.items():
        if y_range is not None:
            fig.update_yaxes(range=y_range, row=row, col=col)

    fig.update_layout(
        title=f"Longitudinal Wavelet Analysis<br><sub>{len(results_sorted)} nights from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}</sub>",
        height=1400,
        showlegend=False
    )

    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'wavelet_longitudinal',
            'height': 1400,
            'width': 1600,
            'scale': 2
        },
        'displaylogo': False
    }

    fig.write_html(str(output_path), config=config)
    return fig


def print_summary_stats(results):
    """Print summary statistics."""
    if not results:
        print("No results to summarize")
        return

    print()
    print("=" * 70)
    print("LONGITUDINAL WAVELET SUMMARY")
    print("=" * 70)
    print(f"Total nights analyzed: {len(results)}")

    dated = [r for r in results if r['date']]
    if dated:
        dates = [r['date'] for r in dated]
        print(f"Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")

    print()
    print("Candidate Metrics:")
    print("-" * 50)

    for metric in ['dominant_freq_std', 'vlf_power_mean', 'vlf_resp_ratio', 'band_power_spikes']:
        values = [r[metric] for r in results]
        print(f"  {metric:22s}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    print()
    print("=" * 70)


def main():
    print("=" * 70)
    print("OscilloBreath - Longitudinal Wavelet Analysis")
    print("Track wavelet stability metrics over time")
    print("=" * 70)
    print()

    folder_path = select_data_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    print(f"Processing: {folder_path}")
    print()

    import time
    start_time = time.time()

    results = process_folder(folder_path)

    elapsed = time.time() - start_time

    if not results:
        print("No valid results. Check that folder contains ResMed EDF files.")
        return

    print()
    print(f"Processed {len(results)} nights in {elapsed:.1f} seconds ({elapsed/len(results):.2f} s/night)")

    print_summary_stats(results)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "wavelet_longitudinal.csv"
    save_results_csv(results, csv_path)
    print(f"\nSaved CSV: {csv_path}")

    plot_path = output_dir / "wavelet_longitudinal.html"
    create_longitudinal_plot(results, plot_path)
    print(f"Saved plot: {plot_path}")

    print()
    print("Opening visualization...")
    webbrowser.open(f'file://{plot_path.absolute()}')


if __name__ == "__main__":
    main()
