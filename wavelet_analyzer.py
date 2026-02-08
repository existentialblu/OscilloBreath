"""
OscilloBreath - Wavelet Analysis

Time-frequency decomposition of respiratory flow data using continuous
wavelet transform. Shows how different frequency components evolve
throughout the night.

Useful for:
- Seeing periodic breathing cycles at different timescales
- Tracking respiratory rate changes over time
- Identifying when oscillatory patterns emerge/disappear
- Visualizing the "wobble" in time-frequency space
"""

import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tkinter import Tk, filedialog
from pathlib import Path
import webbrowser
from data_loader import load_flow_data


def select_file():
    """Open file picker dialog"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select EDF file for wavelet analysis",
        filetypes=[("EDF files", "*.edf"), ("All files", "*.*")]
    )

    root.destroy()
    return file_path


def morlet_wavelet(M, w=6.0):
    """
    Generate a Morlet wavelet.

    Args:
        M: length of the wavelet
        w: wavelet parameter (frequency)

    Returns:
        Complex Morlet wavelet array
    """
    t = np.linspace(-4, 4, M)
    wavelet = np.exp(1j * w * t) * np.exp(-t**2 / 2)
    return wavelet


def compute_cwt(data, scales, wavelet_func, w=6.0):
    """
    Compute continuous wavelet transform using FFT-based convolution.

    Args:
        data: input signal
        scales: array of scales to use
        wavelet_func: wavelet function
        w: wavelet parameter

    Returns:
        2D array of CWT coefficients (scales x time)
    """
    n = len(data)
    output = np.zeros((len(scales), n), dtype=complex)

    # FFT of data
    data_fft = np.fft.fft(data)

    for i, scale in enumerate(scales):
        # Wavelet length based on scale
        wavelet_len = min(int(10 * scale), n)
        if wavelet_len < 4:
            wavelet_len = 4

        # Generate wavelet
        wavelet = wavelet_func(wavelet_len, w=w)

        # Normalize
        wavelet = wavelet / np.sqrt(scale)

        # Pad wavelet to data length
        wavelet_padded = np.zeros(n, dtype=complex)
        start = (n - wavelet_len) // 2
        wavelet_padded[start:start + wavelet_len] = wavelet

        # FFT convolution
        wavelet_fft = np.fft.fft(wavelet_padded)
        output[i, :] = np.fft.ifft(data_fft * np.conj(wavelet_fft))

    return output


def compute_wavelet_transform(flow, sample_rate, freq_min=0.01, freq_max=1.0, num_freqs=100):
    """
    Compute continuous wavelet transform using Morlet wavelet.

    Args:
        flow: flow signal array
        sample_rate: Hz
        freq_min: minimum frequency of interest (Hz) - default 0.01 = 100 sec periods
        freq_max: maximum frequency of interest (Hz) - default 1.0 = 1 sec periods
        num_freqs: number of frequency bins

    Returns:
        dict with 'power', 'frequencies', 'times'
    """
    # Downsample for computational efficiency
    target_fs = 5.0  # 5 Hz is plenty for respiratory analysis
    if sample_rate > target_fs:
        downsample_factor = int(sample_rate / target_fs)
        flow_ds = flow[::downsample_factor]
        fs = sample_rate / downsample_factor
    else:
        flow_ds = flow
        fs = sample_rate

    # Remove mean
    flow_ds = flow_ds - np.mean(flow_ds)

    # Define frequencies of interest (log-spaced for better resolution at low freqs)
    frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), num_freqs)

    # Convert frequencies to scales for Morlet wavelet
    # For Morlet wavelet with w=6 (default), scale = w / (2 * pi * freq) * fs
    w = 6.0  # Morlet wavelet parameter
    scales = (w * fs) / (2 * np.pi * frequencies)

    # Compute CWT
    print(f"Computing wavelet transform ({num_freqs} frequencies, {len(flow_ds)} samples)...")

    # Use our own CWT implementation (scipy.signal.cwt was removed in newer versions)
    coefficients = compute_cwt(flow_ds, scales, morlet_wavelet, w=w)

    # Compute power (magnitude squared)
    power = np.abs(coefficients) ** 2

    # Time axis
    times = np.arange(len(flow_ds)) / fs

    return {
        'power': power,
        'frequencies': frequencies,
        'times': times,
        'flow_downsampled': flow_ds,
        'sample_rate_ds': fs
    }


def extract_dominant_frequency(power, frequencies, times, window_minutes=5):
    """
    Extract the dominant frequency over time using sliding windows.

    Returns array of dominant frequencies at each time point.
    """
    fs = 1 / (times[1] - times[0]) if len(times) > 1 else 1
    window_samples = int(window_minutes * 60 * fs)

    dominant_freqs = []
    dominant_times = []

    for i in range(0, len(times) - window_samples, window_samples // 2):
        window_power = power[:, i:i + window_samples]
        mean_power = np.mean(window_power, axis=1)
        dominant_idx = np.argmax(mean_power)
        dominant_freqs.append(frequencies[dominant_idx])
        dominant_times.append(times[i + window_samples // 2])

    return np.array(dominant_times), np.array(dominant_freqs)


def compute_band_power(power, frequencies, times):
    """
    Compute power in physiologically relevant frequency bands.

    Bands:
    - Very low frequency (VLF): 0.01-0.04 Hz (25-100 sec periods) - periodic breathing
    - Low frequency (LF): 0.04-0.15 Hz (7-25 sec periods) - slow respiratory modulation
    - Respiratory (RESP): 0.15-0.5 Hz (2-7 sec periods) - normal breathing range
    - High frequency (HF): 0.5-1.0 Hz (1-2 sec periods) - fast oscillations
    """
    bands = {
        'VLF (PB range)': (0.01, 0.04),
        'LF (slow mod)': (0.04, 0.15),
        'RESP (breathing)': (0.15, 0.5),
        'HF (fast)': (0.5, 1.0)
    }

    band_powers = {}

    for band_name, (f_low, f_high) in bands.items():
        mask = (frequencies >= f_low) & (frequencies <= f_high)
        if np.any(mask):
            band_power = np.mean(power[mask, :], axis=0)
            band_powers[band_name] = band_power

    return band_powers


def create_wavelet_plot(wavelet_data, band_powers, dominant_times, dominant_freqs,
                        flow, sample_rate, filename, output_path):
    """Create interactive wavelet analysis visualization."""

    power = wavelet_data['power']
    frequencies = wavelet_data['frequencies']
    times = wavelet_data['times']

    # Convert times to minutes for display
    times_min = times / 60
    dominant_times_min = dominant_times / 60

    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Flow Signal (downsampled)',
            'Wavelet Scalogram (time-frequency power)',
            'Frequency Band Power Over Time',
            'Dominant Frequency Over Time'
        ),
        vertical_spacing=0.08,
        row_heights=[0.15, 0.4, 0.25, 0.2]
    )

    # Row 1: Flow signal
    flow_ds = wavelet_data['flow_downsampled']
    flow_times_min = np.arange(len(flow_ds)) / wavelet_data['sample_rate_ds'] / 60

    fig.add_trace(
        go.Scatter(
            x=flow_times_min,
            y=flow_ds,
            mode='lines',
            line=dict(color='blue', width=0.5),
            name='Flow'
        ),
        row=1, col=1
    )

    # Row 2: Scalogram (heatmap)
    # Subsample for plotting if too large
    max_time_points = 2000
    if power.shape[1] > max_time_points:
        step = power.shape[1] // max_time_points
        power_plot = power[:, ::step]
        times_plot = times_min[::step]
    else:
        power_plot = power
        times_plot = times_min

    # Log scale for power visualization
    power_log = np.log10(power_plot + 1e-10)

    fig.add_trace(
        go.Heatmap(
            x=times_plot,
            y=frequencies,
            z=power_log,
            colorscale='Viridis',
            name='Power',
            hovertemplate='Time: %{x:.1f} min<br>Freq: %{y:.3f} Hz<br>Period: %{customdata:.1f} sec<br>Log Power: %{z:.2f}<extra></extra>',
            customdata=1/frequencies[:, np.newaxis] * np.ones((1, len(times_plot)))
        ),
        row=2, col=1
    )

    # Add period annotations on right side
    fig.update_yaxes(
        type='log',
        title_text='Frequency (Hz)',
        row=2, col=1
    )

    # Row 3: Band power over time
    colors = {'VLF (PB range)': 'red', 'LF (slow mod)': 'orange',
              'RESP (breathing)': 'green', 'HF (fast)': 'blue'}

    for band_name, band_power in band_powers.items():
        # Smooth the band power
        window = min(100, len(band_power) // 10)
        if window > 1:
            band_power_smooth = np.convolve(band_power, np.ones(window)/window, mode='same')
        else:
            band_power_smooth = band_power

        # Subsample if needed
        if len(band_power_smooth) > max_time_points:
            step = len(band_power_smooth) // max_time_points
            bp_plot = band_power_smooth[::step]
            bp_times = times_min[::step]
        else:
            bp_plot = band_power_smooth
            bp_times = times_min[:len(bp_plot)]

        fig.add_trace(
            go.Scatter(
                x=bp_times,
                y=bp_plot,
                mode='lines',
                name=band_name,
                line=dict(color=colors.get(band_name, 'gray'), width=1.5)
            ),
            row=3, col=1
        )

    # Row 4: Dominant frequency
    fig.add_trace(
        go.Scatter(
            x=dominant_times_min,
            y=dominant_freqs,
            mode='lines+markers',
            marker=dict(size=4, color='purple'),
            line=dict(color='purple', width=1),
            name='Dominant Freq'
        ),
        row=4, col=1
    )

    # Add secondary y-axis annotation for period
    fig.add_trace(
        go.Scatter(
            x=dominant_times_min,
            y=1/dominant_freqs,
            mode='lines',
            line=dict(color='purple', width=1, dash='dot'),
            name='Dominant Period (sec)',
            yaxis='y8',
            visible='legendonly'
        ),
        row=4, col=1
    )

    # Update layout
    duration_hours = times[-1] / 3600

    fig.update_layout(
        title=f"Wavelet Analysis: {filename}<br><sub>Duration: {duration_hours:.2f} hours | Sample rate: {sample_rate} Hz</sub>",
        height=1200,
        showlegend=True,
        legend=dict(x=1.02, y=0.5)
    )

    # X-axis labels
    fig.update_xaxes(title_text='Time (minutes)', row=4, col=1)
    fig.update_xaxes(title_text='', row=1, col=1)
    fig.update_xaxes(title_text='', row=2, col=1)
    fig.update_xaxes(title_text='', row=3, col=1)

    # Y-axis labels
    fig.update_yaxes(title_text='Flow', row=1, col=1)
    fig.update_yaxes(title_text='Band Power', row=3, col=1)
    fig.update_yaxes(title_text='Frequency (Hz)', row=4, col=1)

    # Add horizontal lines for key frequencies
    key_periods = [60, 30, 15, 5]  # seconds
    for period in key_periods:
        freq = 1 / period
        if frequencies.min() <= freq <= frequencies.max():
            fig.add_hline(
                y=freq,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                annotation_text=f"{period}s",
                row=2, col=1
            )

    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'wavelet_{Path(filename).stem}',
            'height': 1200,
            'width': 1600,
            'scale': 2
        },
        'displaylogo': False
    }

    fig.write_html(str(output_path), config=config)
    return fig


def print_summary(wavelet_data, band_powers, dominant_freqs, filename):
    """Print summary statistics."""

    print()
    print("=" * 70)
    print(f"WAVELET ANALYSIS: {Path(filename).name}")
    print("=" * 70)

    duration_min = wavelet_data['times'][-1] / 60
    print(f"Duration: {duration_min:.1f} minutes ({duration_min/60:.2f} hours)")

    print()
    print("Frequency Band Summary:")
    print("-" * 50)

    for band_name, band_power in band_powers.items():
        mean_power = np.mean(band_power)
        max_power = np.max(band_power)
        print(f"  {band_name:20s}: mean={mean_power:.2e}, max={max_power:.2e}")

    print()
    print("Dominant Frequency Stats:")
    print("-" * 50)
    print(f"  Mean: {np.mean(dominant_freqs):.4f} Hz ({1/np.mean(dominant_freqs):.1f} sec period)")
    print(f"  Std:  {np.std(dominant_freqs):.4f} Hz")
    print(f"  Min:  {np.min(dominant_freqs):.4f} Hz ({1/np.min(dominant_freqs):.1f} sec period)")
    print(f"  Max:  {np.max(dominant_freqs):.4f} Hz ({1/np.max(dominant_freqs):.1f} sec period)")

    # Check for periodic breathing signature
    pb_freq_range = (0.01, 0.04)  # 25-100 second periods
    pb_dominant = np.sum((dominant_freqs >= pb_freq_range[0]) & (dominant_freqs <= pb_freq_range[1]))
    pb_percent = 100 * pb_dominant / len(dominant_freqs)

    print()
    print(f"Time with dominant freq in PB range (25-100s): {pb_percent:.1f}%")

    print()
    print("=" * 70)


def main():
    print("=" * 70)
    print("OscilloBreath - Wavelet Analysis")
    print("Time-frequency decomposition of respiratory flow")
    print("=" * 70)
    print()

    # Select file
    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Loading: {Path(file_path).name}")

    # Load data
    flow, sample_rate, data_type = load_flow_data(file_path)

    print(f"Data type: {data_type}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(flow)/sample_rate/3600:.2f} hours")
    print()

    # Compute wavelet transform
    wavelet_data = compute_wavelet_transform(
        flow, sample_rate,
        freq_min=0.01,   # 100 sec period (periodic breathing range)
        freq_max=0.5,    # 2 sec period (fast breathing)
        num_freqs=80
    )

    # Extract features
    print("Extracting frequency bands...")
    band_powers = compute_band_power(
        wavelet_data['power'],
        wavelet_data['frequencies'],
        wavelet_data['times']
    )

    print("Finding dominant frequencies...")
    dominant_times, dominant_freqs = extract_dominant_frequency(
        wavelet_data['power'],
        wavelet_data['frequencies'],
        wavelet_data['times'],
        window_minutes=5
    )

    # Print summary
    print_summary(wavelet_data, band_powers, dominant_freqs, file_path)

    # Create output
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"wavelet_{Path(file_path).stem}.html"

    print(f"\nCreating visualization...")
    create_wavelet_plot(
        wavelet_data, band_powers,
        dominant_times, dominant_freqs,
        flow, sample_rate,
        Path(file_path).name,
        output_path
    )

    print(f"Saved: {output_path}")
    print()
    print("Opening in browser...")
    webbrowser.open(f'file://{output_path.absolute()}')


if __name__ == "__main__":
    main()
