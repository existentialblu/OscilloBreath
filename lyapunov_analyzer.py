import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyedflib
from tkinter import Tk, filedialog
from pathlib import Path
import webbrowser
from scipy.spatial.distance import cdist

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

def time_delay_embedding(signal, tau, embedding_dim):
    """
    Create time-delay embedded phase space reconstruction
    Using Takens' theorem to reconstruct the attractor
    """
    N = len(signal)
    M = N - (embedding_dim - 1) * tau

    embedded = np.zeros((M, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = signal[i * tau : i * tau + M]

    return embedded

def estimate_lyapunov_rosenstein(signal, tau, embedding_dim, fs, max_iter=None):
    """
    Calculate Largest Lyapunov Exponent using Rosenstein et al. method

    This tracks how nearby trajectories in phase space diverge over time.
    Positive = chaos, Zero = periodic, Negative = convergent

    Parameters:
    - signal: 1D time series
    - tau: time delay for embedding
    - embedding_dim: embedding dimension
    - fs: sampling frequency
    - max_iter: maximum iterations to track divergence
    """

    # Reconstruct phase space
    embedded = time_delay_embedding(signal, tau, embedding_dim)
    N = len(embedded)

    if max_iter is None:
        max_iter = min(200, N // 10)

    # For each point, find nearest neighbor (excluding nearby points)
    min_temporal_separation = int(fs * 2)  # At least 2 seconds apart

    divergences = []

    print(f"  Tracking trajectory divergence (this may take a moment)...")

    # Sample points to speed up computation
    sample_indices = np.arange(0, N - max_iter, max(1, N // 1000))

    for i in sample_indices:
        # Find nearest neighbor (temporally separated)
        distances = np.sqrt(np.sum((embedded - embedded[i])**2, axis=1))

        # Exclude points too close in time
        valid_mask = np.abs(np.arange(N) - i) > min_temporal_separation
        distances[~valid_mask] = np.inf

        if np.all(np.isinf(distances)):
            continue

        nearest_idx = np.argmin(distances)

        if nearest_idx + max_iter >= N or i + max_iter >= N:
            continue

        # Track divergence over time
        divergence = []
        for j in range(max_iter):
            if i + j >= N or nearest_idx + j >= N:
                break
            d = np.sqrt(np.sum((embedded[i + j] - embedded[nearest_idx + j])**2))
            if d > 0:
                divergence.append(np.log(d))

        if len(divergence) > 0:
            divergences.append(divergence)

    if len(divergences) == 0:
        return None, None

    # Average divergence curves
    min_len = min(len(d) for d in divergences)
    divergences_array = np.array([d[:min_len] for d in divergences])
    mean_divergence = np.mean(divergences_array, axis=0)

    # LLE is the slope of the linear region
    # Fit line to early part (where it's most linear)
    fit_region = slice(int(min_len * 0.1), int(min_len * 0.4))
    time_steps = np.arange(len(mean_divergence))

    # Linear fit
    coeffs = np.polyfit(time_steps[fit_region], mean_divergence[fit_region], 1)
    lle = coeffs[0] * fs  # Convert to bits per second

    return lle, mean_divergence

def calculate_windowed_lle(flow, sample_rate, window_minutes=10):
    """
    Calculate LLE in sliding windows to see how chaos evolves over the night
    """
    print("Calculating Largest Lyapunov Exponent in windows...")
    print("(This is computationally intensive and will take a few minutes)")

    # Embedding parameters
    # tau = average period / 4 (rough heuristic)
    # For breathing ~15 breaths/min = 4 seconds per breath
    tau = int(sample_rate * 1.0)  # 1 second delay
    embedding_dim = 3  # 3D embedding (flow, delayed flow, double-delayed flow)

    window_samples = int(window_minutes * 60 * sample_rate)
    hop_samples = window_samples // 2  # 50% overlap

    num_windows = (len(flow) - window_samples) // hop_samples + 1

    window_data = {
        'lle': [],
        'window_center_time': []
    }

    for i in range(num_windows):
        start_idx = i * hop_samples
        end_idx = start_idx + window_samples

        if end_idx > len(flow):
            break

        print(f"  Window {i+1}/{num_windows} ({start_idx/sample_rate/60:.1f} min)...")

        window_flow = flow[start_idx:end_idx]

        # Calculate LLE for this window
        lle, divergence = estimate_lyapunov_rosenstein(
            window_flow, tau, embedding_dim, sample_rate
        )

        if lle is not None:
            window_data['lle'].append(lle)
            center_time = (start_idx + window_samples // 2) / sample_rate / 60
            window_data['window_center_time'].append(center_time)
        else:
            print(f"    Warning: Could not calculate LLE for this window")

    # Convert to arrays
    for key in window_data:
        window_data[key] = np.array(window_data[key])

    return window_data, tau, embedding_dim

def create_lyapunov_plot(flow, sample_rate, window_data, tau, embedding_dim, filename):
    """Create visualization of Lyapunov analysis"""

    print("Creating visualization...")

    # Downsample for phase space
    downsample_factor = max(1, len(flow) // 30000)
    if downsample_factor > 1:
        flow_viz = flow[::downsample_factor]
        time_viz = np.arange(len(flow_viz)) * downsample_factor
    else:
        flow_viz = flow
        time_viz = np.arange(len(flow))

    time_minutes_viz = time_viz / sample_rate / 60

    # Calculate derivative for phase space
    dt = 1.0 / sample_rate
    flow_derivative = np.gradient(flow, dt)
    flow_derivative_viz = flow_derivative[::downsample_factor] if downsample_factor > 1 else flow_derivative

    # Reconstruct embedded phase space for visualization
    embedded = time_delay_embedding(flow_viz, tau // downsample_factor if downsample_factor > 1 else tau, embedding_dim)

    # LLE statistics
    if len(window_data['lle']) > 0:
        mean_lle = np.mean(window_data['lle'])
        median_lle = np.median(window_data['lle'])
        max_lle = np.max(window_data['lle'])
        min_lle = np.min(window_data['lle'])
    else:
        mean_lle = median_lle = max_lle = min_lle = 0

    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Standard Phase Space (Flow vs dFlow/dt)',
            f'Reconstructed Phase Space (Embedding dim={embedding_dim}, tau={tau/sample_rate:.2f}s)',
            'Largest Lyapunov Exponent Over Time',
            'Flow Rate Over Time'
        ),
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.3, 0.2],
        specs=[[{"type": "scatter"}],
               [{"type": "scatter3d"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}]]
    )

    # 1. Standard phase space
    fig.add_trace(
        go.Scattergl(
            x=flow_viz,
            y=flow_derivative_viz,
            mode='markers',
            marker=dict(
                size=2,
                color=time_minutes_viz,
                colorscale='Viridis',
                showscale=False,
                opacity=0.4
            ),
            name='Standard Phase Space',
            showlegend=False,
            hovertemplate='Flow: %{x:.2f}<br>dFlow/dt: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. 3D Embedded phase space
    if embedding_dim == 3:
        fig.add_trace(
            go.Scatter3d(
                x=embedded[:, 0],
                y=embedded[:, 1],
                z=embedded[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=np.arange(len(embedded)),
                    colorscale='Viridis',
                    showscale=False,
                    opacity=0.3
                ),
                name='Embedded Attractor',
                showlegend=False,
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

    # 3. LLE over time
    if len(window_data['lle']) > 0:
        window_times = window_data['window_center_time']
        lles = window_data['lle']

        # Color-code by chaos level
        fig.add_trace(
            go.Scatter(
                x=window_times,
                y=lles,
                mode='lines+markers',
                line=dict(color='purple', width=2),
                marker=dict(
                    size=8,
                    color=lles,
                    colorscale='RdYlGn_r',  # Red = chaotic, Green = stable
                    showscale=True,
                    colorbar=dict(
                        title="LLE<br>(chaos)",
                        x=1.15,
                        len=0.25,
                        y=0.35
                    )
                ),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.2)',
                name='LLE',
                showlegend=False,
                hovertemplate='Time: %{x:.1f} min<br>LLE: %{y:.4f}<extra></extra>'
            ),
            row=3, col=1
        )

        # Add mean line
        fig.add_hline(
            y=mean_lle,
            line=dict(color='gray', dash='dash', width=1),
            annotation_text=f"Mean: {mean_lle:.4f}",
            annotation_position="right",
            row=3, col=1
        )

        # Add zero line (separates chaos from stability)
        fig.add_hline(
            y=0,
            line=dict(color='black', dash='dot', width=1),
            annotation_text="Zero (periodic)",
            annotation_position="left",
            row=3, col=1
        )

    # 4. Flow rate
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
        row=4, col=1
    )

    # Update axes
    fig.update_xaxes(title_text="Flow (L/s)", row=1, col=1)
    fig.update_yaxes(title_text="dFlow/dt (L/s²)", row=1, col=1)

    fig.update_scenes(
        xaxis_title="Flow(t)",
        yaxis_title=f"Flow(t-{tau/sample_rate:.1f}s)",
        zaxis_title=f"Flow(t-{2*tau/sample_rate:.1f}s)",
        row=2, col=1
    )

    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    fig.update_yaxes(title_text="LLE (bits/second)", row=3, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=4, col=1)
    fig.update_yaxes(title_text="Flow (L/s)", row=4, col=1)

    # Stats annotation
    stats_text = (
        f"<b>Lyapunov Exponent Statistics:</b><br>"
        f"Mean LLE: {mean_lle:.4f} bits/s<br>"
        f"Median LLE: {median_lle:.4f} bits/s<br>"
        f"Max LLE: {max_lle:.4f} bits/s<br>"
        f"Min LLE: {min_lle:.4f} bits/s<br>"
        f"<br>"
        f"<b>Interpretation:</b><br>"
        f"Positive = Chaotic (unstable)<br>"
        f"Zero = Periodic (neutral)<br>"
        f"Negative = Convergent (stable)<br>"
        f"<br>"
        f"<b>Embedding Parameters:</b><br>"
        f"Dimension: {embedding_dim}<br>"
        f"Time delay: {tau/sample_rate:.2f} seconds"
    )

    # Layout
    duration_hours = len(flow) / sample_rate / 3600
    fig.update_layout(
        title=f"Lyapunov Exponent Analysis: {filename}<br>" +
              f"<sub>Duration: {duration_hours:.1f} hours | Mean LLE: {mean_lle:.4f} bits/s</sub>",
        height=1600,
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
            'filename': f'{filename}_lyapunov',
            'height': 1600,
            'width': 1600,
            'scale': 2
        },
        'displaylogo': False
    }

    return fig, config

def main():
    print("=" * 70)
    print("OscilloBreath - Lyapunov Exponent Analysis")
    print("Measuring chaos in the respiratory oscillator")
    print("=" * 70)
    print()
    print("NOTE: This analysis is computationally intensive.")
    print("It may take several minutes depending on file size.")
    print()

    # Select file
    edf_path = select_edf_file()
    if not edf_path:
        print("No file selected. Exiting.")
        return

    filename = Path(edf_path).stem

    # Read data
    flow, sample_rate = read_flow_data(edf_path)

    # Calculate windowed LLE
    print()
    window_data, tau, embedding_dim = calculate_windowed_lle(flow, sample_rate, window_minutes=10)

    if len(window_data['lle']) == 0:
        print()
        print("ERROR: Could not calculate LLE for any windows.")
        print("The data may be too short or too noisy.")
        return

    print()
    print("=" * 70)
    print("LYAPUNOV EXPONENT ANALYSIS")
    print("=" * 70)
    print(f"Mean LLE: {np.mean(window_data['lle']):.4f} bits/second")
    print(f"Median LLE: {np.median(window_data['lle']):.4f} bits/second")
    print()
    if np.mean(window_data['lle']) > 0.01:
        print("INTERPRETATION: Positive LLE indicates CHAOTIC dynamics")
        print("Your respiratory oscillator is sensitive to perturbations.")
    elif np.mean(window_data['lle']) < -0.01:
        print("INTERPRETATION: Negative LLE indicates STABLE dynamics")
        print("Your respiratory oscillator is convergent and well-damped.")
    else:
        print("INTERPRETATION: Near-zero LLE indicates PERIODIC dynamics")
        print("Your respiratory oscillator is in a limit cycle.")
    print("=" * 70)
    print()

    # Create plot
    fig, config = create_lyapunov_plot(flow, sample_rate, window_data, tau, embedding_dim, filename)

    # Save
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{filename}_lyapunov.html"
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
