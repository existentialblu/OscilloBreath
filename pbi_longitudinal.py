"""
OscilloBreath - Periodic Breathing Index (PBI) Longitudinal Tracker

Two focused metrics from wavelet analysis:

1. PBI (Periodic Breathing Index) = VLF power (0.01-0.04 Hz)
   - Measures therapy effectiveness / oscillator stability
   - Lower = less periodic breathing = better

2. LF Power = Low frequency power (0.04-0.15 Hz)
   - Measures health status / illness detection
   - Spikes when sick, independent of therapy changes

Based on wavelet analysis of respiratory flow data.
"""

import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
from datetime import datetime, timedelta
import csv
import webbrowser
import plotly.graph_objects as go

from data_loader import load_flow_data
from wavelet_analyzer import compute_wavelet_transform, compute_band_power


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


def compute_pbi_and_lf(flow, sample_rate):
    """
    Compute Periodic Breathing Index and LF Power for a single night.

    PBI = mean power in VLF band (0.01-0.04 Hz) - therapy effectiveness
    LF = mean power in LF band (0.04-0.15 Hz) - health status

    Returns (pbi, lf_power) or (None, None) on failure.
    """
    try:
        # Compute wavelet transform (minimal settings for speed)
        wavelet_data = compute_wavelet_transform(
            flow, sample_rate,
            freq_min=0.008,  # Slightly below VLF to capture edge
            freq_max=0.2,    # Cover both VLF and LF bands
            num_freqs=40     # Fewer frequencies = faster
        )

        # Extract band powers
        band_powers = compute_band_power(
            wavelet_data['power'],
            wavelet_data['frequencies'],
            wavelet_data['times']
        )

        vlf_power = band_powers.get('VLF (PB range)', None)
        lf_power = band_powers.get('LF (slow mod)', None)

        if vlf_power is None or len(vlf_power) == 0:
            return None, None

        # PBI = mean VLF power
        pbi = np.mean(vlf_power)

        # LF = mean LF power
        lf = np.mean(lf_power) if lf_power is not None and len(lf_power) > 0 else None

        return pbi, lf

    except Exception as e:
        return None, None


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

    for i, night_info in enumerate(nights):
        night_date = night_info['date']
        night_files = night_info['files']

        sessions_str = f"({len(night_files)} sessions)" if len(night_files) > 1 else ""
        print(f"  [{i+1}/{len(nights)}] {night_date.strftime('%Y-%m-%d')} {sessions_str}...", end=' ', flush=True)

        # Load and concatenate
        flow, sample_rate, duration_hours = load_and_concatenate_night(night_info)

        if flow is None:
            print("LOAD ERROR")
            continue

        if duration_hours * 60 < min_duration_minutes:
            print(f"SHORT ({duration_hours*60:.0f} min)")
            continue

        # Compute PBI and LF
        pbi, lf = compute_pbi_and_lf(flow, sample_rate)

        if pbi is None:
            print("ANALYSIS ERROR")
            continue

        results.append({
            'date': night_date,
            'pbi': pbi,
            'lf_power': lf,
            'duration_hours': duration_hours
        })

        lf_str = f", LF = {lf:.6f}" if lf is not None else ""
        print(f"PBI = {pbi:.6f}{lf_str}")

    print()
    print(f"Processed {len(results)} nights")

    return results


def save_results_csv(results, output_path):
    """Save results to CSV file."""
    if not results:
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'pbi', 'lf_power', 'duration_hours'])
        writer.writeheader()

        for r in results:
            writer.writerow({
                'date': r['date'].strftime('%Y-%m-%d'),
                'pbi': r['pbi'],
                'lf_power': r.get('lf_power', ''),
                'duration_hours': r['duration_hours']
            })


def create_plot(results, output_path):
    """Create clean PBI + LF longitudinal visualization."""
    from plotly.subplots import make_subplots

    results_sorted = sorted(results, key=lambda x: x['date'])

    if len(results_sorted) < 2:
        print("Not enough results for plot")
        return None

    dates = [r['date'] for r in results_sorted]
    pbi_values = [r['pbi'] for r in results_sorted]
    lf_values = [r.get('lf_power', 0) or 0 for r in results_sorted]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'PBI - Periodic Breathing Index (therapy effectiveness)',
            'LF Power (health status / illness detection)'
        ),
        vertical_spacing=0.12
    )

    # --- PBI Plot ---
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pbi_values,
            mode='markers',
            marker=dict(size=5, color='orange', opacity=0.4),
            name='Nightly PBI',
            showlegend=False
        ),
        row=1, col=1
    )

    if len(pbi_values) >= 7:
        rolling = np.convolve(pbi_values, np.ones(7)/7, mode='valid')
        rolling_dates = dates[3:-3]

        fig.add_trace(
            go.Scatter(
                x=rolling_dates,
                y=rolling,
                mode='lines',
                line=dict(color='darkorange', width=3),
                name='PBI 7-day avg',
                showlegend=True
            ),
            row=1, col=1
        )

        # Scale y-axis to rolling average
        rolling_min = np.min(rolling)
        rolling_max = np.max(rolling)
        padding = (rolling_max - rolling_min) * 0.3
        if padding > 0:
            fig.update_yaxes(range=[max(0, rolling_min - padding), rolling_max + padding], row=1, col=1)

    # --- LF Power Plot ---
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=lf_values,
            mode='markers',
            marker=dict(size=5, color='green', opacity=0.4),
            name='Nightly LF',
            showlegend=False
        ),
        row=2, col=1
    )

    if len(lf_values) >= 7:
        rolling_lf = np.convolve(lf_values, np.ones(7)/7, mode='valid')
        rolling_dates = dates[3:-3]

        fig.add_trace(
            go.Scatter(
                x=rolling_dates,
                y=rolling_lf,
                mode='lines',
                line=dict(color='darkgreen', width=3),
                name='LF 7-day avg',
                showlegend=True
            ),
            row=2, col=1
        )

        # Scale y-axis to rolling average
        rolling_min = np.min(rolling_lf)
        rolling_max = np.max(rolling_lf)
        padding = (rolling_max - rolling_min) * 0.3
        if padding > 0:
            fig.update_yaxes(range=[max(0, rolling_min - padding), rolling_max + padding], row=2, col=1)

    # Layout
    fig.update_layout(
        title=f"Respiratory Stability Metrics<br><sub>{len(results_sorted)} nights from {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}</sub>",
        height=700,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )

    fig.update_yaxes(title_text="PBI (lower = better)", row=1, col=1)
    fig.update_yaxes(title_text="LF Power (spikes = sick?)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'pbi_longitudinal',
            'height': 700,
            'width': 1200,
            'scale': 2
        },
        'displaylogo': False
    }

    fig.write_html(str(output_path), config=config)
    return fig


def main():
    print("=" * 70)
    print("OscilloBreath - PBI & LF Power Tracker")
    print("Two focused respiratory stability metrics")
    print("=" * 70)
    print()
    print("PBI (Periodic Breathing Index) = VLF power (0.01-0.04 Hz)")
    print("  -> Therapy effectiveness. Lower = less periodic breathing = better")
    print()
    print("LF Power = Low frequency power (0.04-0.15 Hz)")
    print("  -> Health status. Spikes may indicate illness")
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
        print("No valid results.")
        return

    print()
    print(f"Completed in {elapsed:.1f} seconds ({elapsed/len(results):.2f} s/night)")

    # Summary stats
    pbi_values = [r['pbi'] for r in results]
    lf_values = [r.get('lf_power', 0) for r in results if r.get('lf_power') is not None]

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("PBI (Periodic Breathing Index) - therapy effectiveness:")
    print(f"  Mean:   {np.mean(pbi_values):.6f}")
    print(f"  Median: {np.median(pbi_values):.6f}")
    print(f"  Std:    {np.std(pbi_values):.6f}")
    print()
    print("LF Power - health status (spikes may indicate illness):")
    if lf_values:
        print(f"  Mean:   {np.mean(lf_values):.6f}")
        print(f"  Median: {np.median(lf_values):.6f}")
        print(f"  Std:    {np.std(lf_values):.6f}")
    else:
        print("  No LF data available")
    print("=" * 70)

    # Save outputs
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "pbi_longitudinal.csv"
    save_results_csv(results, csv_path)
    print(f"\nSaved CSV: {csv_path}")

    plot_path = output_dir / "pbi_longitudinal.html"
    create_plot(results, plot_path)
    print(f"Saved plot: {plot_path}")

    print()
    print("Opening visualization...")
    webbrowser.open(f'file://{plot_path.absolute()}')


if __name__ == "__main__":
    main()
