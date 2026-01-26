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
            raise ValueError("Could not find flow signal.")

        flow = f.readSignal(flow_idx)
        sample_rate = f.getSampleFrequency(flow_idx)

        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(flow)/sample_rate/3600:.2f} hours")

    return flow, sample_rate

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
        print(f"  Reconstructing phase space...")
        embedded = time_delay_embedding(signal, tau, embedding_dim)
        N = len(embedded)
        print(f"  Embedded points: {N}")

        if N < 500:
            print(f"  WARNING: Only {N} points available, results may be unreliable")

        # Minimum temporal separation (avoid temporally correlated points)
        min_separation = max(10, int(fs * 1.0))  # At least 1 second apart

        divergences = []

        # Sample fewer points for speed
        num_samples = min(200, N // 10)
        sample_indices = np.linspace(0, N - max_iter - 1, num_samples, dtype=int)

        print(f"  Calculating divergence for {len(sample_indices)} sample points...")

        for idx, i in enumerate(sample_indices):
            if idx % 20 == 0:
                print(f"    Progress: {idx}/{len(sample_indices)}")

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
                print(f"    Error at point {i}: {e}")
                continue

        print(f"  Found {len(divergences)} valid trajectory pairs")

        if len(divergences) < 10:
            print(f"  ERROR: Not enough valid trajectories (only {len(divergences)})")
            return None

        # Average divergence curves
        min_len = min(len(d) for d in divergences)
        if min_len < 10:
            print(f"  ERROR: Divergence curves too short ({min_len} points)")
            return None

        divergences_array = np.array([d[:min_len] for d in divergences])
        mean_divergence = np.mean(divergences_array, axis=0)

        # Fit linear region
        # Use middle portion (early part often has transients)
        fit_start = max(5, int(min_len * 0.1))
        fit_end = min(min_len, int(min_len * 0.5))

        if fit_end - fit_start < 5:
            print(f"  ERROR: Fit region too small")
            return None

        time_steps = np.arange(len(mean_divergence))
        coeffs = np.polyfit(time_steps[fit_start:fit_end], mean_divergence[fit_start:fit_end], 1)

        # LLE in bits per second
        lle = coeffs[0] * fs

        print(f"  LLE calculated: {lle:.4f} bits/second")

        return lle

    except Exception as e:
        print(f"  ERROR in LLE calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 70)
    print("OscilloBreath - Fast Lyapunov Exponent Analysis")
    print("Measuring chaos in the respiratory oscillator")
    print("=" * 70)
    print()

    # Select file
    edf_path = select_edf_file()
    if not edf_path:
        print("No file selected. Exiting.")
        return

    filename = Path(edf_path).stem

    try:
        # Read data
        flow, sample_rate = read_flow_data(edf_path)

        # HEAVILY downsample for LLE analysis
        # 25 Hz is overkill - we only need ~1-2 Hz for breathing patterns
        target_fs = 2.0  # 2 Hz target
        downsample_factor = int(sample_rate / target_fs)

        print()
        print(f"Downsampling from {sample_rate} Hz to {sample_rate/downsample_factor:.1f} Hz for analysis...")
        flow_downsampled = flow[::downsample_factor]
        fs_downsampled = sample_rate / downsample_factor

        print(f"Downsampled length: {len(flow_downsampled)} points ({len(flow_downsampled)/fs_downsampled/60:.1f} minutes)")

        # Embedding parameters
        # tau ~ 1 breathing cycle / 4
        # At ~15 breaths/min = 4 seconds/breath, so tau ~ 1 second
        tau = int(fs_downsampled * 1.0)  # 1 second at downsampled rate
        embedding_dim = 3

        print()
        print(f"Embedding parameters:")
        print(f"  Time delay (tau): {tau} samples ({tau/fs_downsampled:.1f} seconds)")
        print(f"  Embedding dimension: {embedding_dim}")
        print()

        # Calculate overall LLE
        print("Calculating Largest Lyapunov Exponent...")
        print("(This will take 2-5 minutes)")
        print()

        lle = calculate_lle_rosenstein(
            flow_downsampled,
            tau,
            embedding_dim,
            fs_downsampled,
            max_iter=100
        )

        print()
        print("=" * 70)
        print("LYAPUNOV EXPONENT ANALYSIS")
        print("=" * 70)

        if lle is None:
            print("ERROR: Could not calculate LLE")
            print("The signal may be too short, too noisy, or lack proper structure.")
            print("=" * 70)
            return

        print(f"Largest Lyapunov Exponent: {lle:.4f} bits/second")
        print()

        if lle > 0.01:
            print("INTERPRETATION: CHAOTIC DYNAMICS")
            print("  Positive LLE indicates your respiratory oscillator is chaotic.")
            print("  Small perturbations grow exponentially.")
            print("  System is unpredictable and sensitive to initial conditions.")
            print("  This suggests poor/no ASV control or APAP struggling.")
        elif lle < -0.01:
            print("INTERPRETATION: STABLE DYNAMICS")
            print("  Negative LLE indicates convergent, stable behavior.")
            print("  Trajectories converge over time.")
            print("  System is well-damped and predictable.")
            print("  This suggests well-tuned ASV providing good control.")
        else:
            print("INTERPRETATION: PERIODIC/NEUTRAL DYNAMICS")
            print("  Near-zero LLE indicates limit cycle behavior.")
            print("  System is periodic but not chaotic.")
            print("  Neither strongly chaotic nor strongly damped.")

        print("=" * 70)
        print()

        # Simple visualization
        print("Creating visualization...")

        # Reconstruct phase space for viz
        embedded = time_delay_embedding(flow_downsampled, tau, embedding_dim)

        # Calculate derivative for standard phase space
        dt = 1.0 / fs_downsampled
        flow_derivative = np.gradient(flow_downsampled, dt)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Standard Phase Space',
                '3D Reconstructed Attractor',
                'Flow Over Time',
                'Stats'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter3d"}],
                   [{"type": "scatter"}, {"type": "table"}]],
            row_heights=[0.5, 0.5]
        )

        # Standard phase space
        time_colors = np.arange(len(flow_downsampled)) / fs_downsampled / 60

        fig.add_trace(
            go.Scattergl(
                x=flow_downsampled,
                y=flow_derivative,
                mode='markers',
                marker=dict(size=2, color=time_colors, colorscale='Viridis', opacity=0.5),
                showlegend=False
            ),
            row=1, col=1
        )

        # 3D embedded space
        fig.add_trace(
            go.Scatter3d(
                x=embedded[:, 0],
                y=embedded[:, 1],
                z=embedded[:, 2],
                mode='markers',
                marker=dict(size=1, color=np.arange(len(embedded)), colorscale='Viridis', opacity=0.4),
                showlegend=False
            ),
            row=1, col=2
        )

        # Flow over time
        time_minutes = np.arange(len(flow_downsampled)) / fs_downsampled / 60
        fig.add_trace(
            go.Scattergl(
                x=time_minutes,
                y=flow_downsampled,
                mode='lines',
                line=dict(color='blue', width=0.5),
                showlegend=False
            ),
            row=2, col=1
        )

        # Stats table
        interpretation = "CHAOTIC" if lle > 0.01 else ("STABLE" if lle < -0.01 else "PERIODIC")
        color = "red" if lle > 0.01 else ("green" if lle < -0.01 else "yellow")

        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[
                    ['LLE (bits/s)', 'Interpretation', 'Embedding Dim', 'Time Delay', 'Duration'],
                    [f'{lle:.4f}', interpretation, embedding_dim, f'{tau/fs_downsampled:.1f}s', f'{len(flow)/sample_rate/60:.1f} min']
                ])
            ),
            row=2, col=2
        )

        # Update axes
        fig.update_xaxes(title_text="Flow (L/s)", row=1, col=1)
        fig.update_yaxes(title_text="dFlow/dt (L/s²)", row=1, col=1)

        fig.update_scenes(
            xaxis_title="Flow(t)",
            yaxis_title=f"Flow(t-{tau/fs_downsampled:.1f}s)",
            zaxis_title=f"Flow(t-{2*tau/fs_downsampled:.1f}s)",
            row=1, col=2
        )

        fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
        fig.update_yaxes(title_text="Flow (L/s)", row=2, col=1)

        fig.update_layout(
            title=f"Lyapunov Analysis: {filename}<br>" +
                  f"<sub>LLE = {lle:.4f} bits/s ({interpretation})</sub>",
            height=1000
        )

        # Save
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"{filename}_lyapunov.html"

        # Config for PNG export with better naming
        config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{filename}_lyapunov_LLE{lle:.3f}_{interpretation}',
                'height': 1000,
                'width': 1600,
                'scale': 2
            },
            'displaylogo': False
        }

        fig.write_html(str(output_file), config=config)

        print(f"Saved to: {output_file}")
        print()
        print("Opening in browser...")
        webbrowser.open(f'file://{output_file.absolute()}')

    except Exception as e:
        print()
        print("=" * 70)
        print("FATAL ERROR")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)

if __name__ == "__main__":
    main()
