import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import load_flow_data
from tkinter import Tk, filedialog
from pathlib import Path
import webbrowser
from scipy.signal import savgol_filter

def select_data_file():
    """Open file picker to select data file or folder"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select data file (ResMed EDF or Philips folder)",
        filetypes=[("Data files", "*.edf *.005 *.006"), ("All files", "*.*")]
    )

    # If they selected a Philips file, use its parent folder
    if file_path and Path(file_path).suffix in ['.005', '.006', '.000', '.001', '.002']:
        file_path = str(Path(file_path).parent)

    root.destroy()
    return file_path

def detect_phase_transitions(flow, sample_rate, window_seconds=60):
    """
    Detect phase transitions - moments when oscillator behavior shifts

    Uses sliding window to track:
    - Flow amplitude (breathing depth)
    - Flow variability (stability)
    - Derivative characteristics (violence)

    A phase transition = significant change in oscillator state
    """

    # Calculate derivative
    dt = 1.0 / sample_rate
    flow_derivative = np.gradient(flow, dt)

    # Window size in samples
    window_samples = int(window_seconds * sample_rate)
    hop_samples = window_samples // 4  # 75% overlap for smooth detection

    # Calculate windowed features
    num_windows = (len(flow) - window_samples) // hop_samples + 1

    features = {
        'flow_std': [],
        'flow_mean': [],
        'deriv_std': [],
        'deriv_mean': [],
        'deriv_range': [],
        'window_centers': []
    }

    print("Calculating windowed features...")
    for i in range(num_windows):
        start_idx = i * hop_samples
        end_idx = start_idx + window_samples

        if end_idx > len(flow):
            break

        window_flow = flow[start_idx:end_idx]
        window_deriv = flow_derivative[start_idx:end_idx]

        features['flow_std'].append(np.std(window_flow))
        features['flow_mean'].append(np.mean(np.abs(window_flow)))
        features['deriv_std'].append(np.std(window_deriv))
        features['deriv_mean'].append(np.mean(np.abs(window_deriv)))
        features['deriv_range'].append(np.ptp(window_deriv))
        features['window_centers'].append((start_idx + end_idx) / 2)

    # Convert to arrays
    for key in features:
        features[key] = np.array(features[key])

    # Normalize features for comparison
    normalized_features = {}
    for key in ['flow_std', 'deriv_std', 'deriv_range']:
        data = features[key]
        if np.std(data) > 0:
            normalized_features[key] = (data - np.mean(data)) / np.std(data)
        else:
            normalized_features[key] = np.zeros_like(data)

    # Create composite state vector
    state_vector = (
        normalized_features['flow_std'] * 0.3 +
        normalized_features['deriv_std'] * 0.4 +
        normalized_features['deriv_range'] * 0.3
    )

    # Smooth state vector to reduce noise
    if len(state_vector) > 5:
        state_vector_smooth = savgol_filter(state_vector,
                                            window_length=min(11, len(state_vector)//2*2+1),
                                            polyorder=2)
    else:
        state_vector_smooth = state_vector

    # Detect transitions: where state changes significantly
    print("Detecting phase transitions...")
    state_diff = np.abs(np.diff(state_vector_smooth))

    # Threshold: transitions are changes > 0.75 standard deviations
    threshold = 0.75 * np.std(state_diff)
    transition_indices = np.where(state_diff > threshold)[0]

    # Convert window indices to sample indices
    transition_times = []
    transition_samples = []

    for idx in transition_indices:
        sample_idx = int(features['window_centers'][idx])
        time_seconds = sample_idx / sample_rate

        # Avoid clustering: only keep transitions >30s apart
        if not transition_times or (time_seconds - transition_times[-1]) > 30:
            transition_times.append(time_seconds)
            transition_samples.append(sample_idx)

    return {
        'transition_times': np.array(transition_times),
        'transition_samples': np.array(transition_samples),
        'state_vector': state_vector_smooth,
        'window_centers': features['window_centers'],
        'features': features,
        'flow_derivative': flow_derivative
    }

def create_phase_transition_plot(flow, sample_rate, transition_data, filename):
    """Create visualization showing phase transitions"""

    print("Preparing visualization data...")

    flow_derivative = transition_data['flow_derivative']
    transition_times = transition_data['transition_times']
    transition_samples = transition_data['transition_samples']
    state_vector = transition_data['state_vector']
    window_centers = transition_data['window_centers']

    # Downsample for visualization (keep every Nth point to reduce plot size)
    # Target: ~50,000 points max for smooth rendering
    downsample_factor = max(1, len(flow) // 50000)

    if downsample_factor > 1:
        print(f"Downsampling by factor of {downsample_factor} for visualization...")
        flow_viz = flow[::downsample_factor]
        flow_derivative_viz = flow_derivative[::downsample_factor]
        time_viz = np.arange(len(flow_viz)) * downsample_factor
    else:
        flow_viz = flow
        flow_derivative_viz = flow_derivative
        time_viz = np.arange(len(flow))

    # Time arrays
    time_seconds_viz = time_viz / sample_rate
    time_minutes_viz = time_seconds_viz / 60
    window_time_minutes = window_centers / sample_rate / 60

    # Duration
    duration_hours = len(flow) / sample_rate / 3600
    transitions_per_hour = len(transition_times) / duration_hours

    print(f"Creating plots with {len(flow_viz)} points...")

    # Create subplots
    print("Building subplot structure...")
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f'Phase Transitions: {len(transition_times)} events ({transitions_per_hour:.1f}/hour)',
            'Flow Rate (Breathing Pattern)',
            'Flow Derivative (Rate of Change)',
            'Oscillator State Vector (Higher = More Chaotic)'
        ),
        vertical_spacing=0.06,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    # 1. Phase space with transitions marked
    print("Adding phase space plot...")
    # Color-code by time
    time_colors = time_minutes_viz

    fig.add_trace(
        go.Scattergl(
            x=flow_viz,
            y=flow_derivative_viz,
            mode='markers',
            marker=dict(
                size=2,
                color=time_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Time<br>(min)",
                    x=1.15,
                    len=0.3,
                    y=0.85
                ),
                opacity=0.4
            ),
            name='Phase Space',
            hovertemplate='Flow: %{x:.2f}<br>dFlow/dt: %{y:.2f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )

    # Mark transitions on phase space
    if len(transition_samples) > 0:
        transition_flow = flow[transition_samples]
        transition_deriv = flow_derivative[transition_samples]

        fig.add_trace(
            go.Scatter(
                x=transition_flow,
                y=transition_deriv,
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='x',
                    line=dict(width=2)
                ),
                name='Phase Transitions',
                hovertemplate='Transition at %{text}<extra></extra>',
                text=[f"{t/60:.1f} min" for t in transition_times],
                showlegend=True
            ),
            row=1, col=1
        )

    # 2. Flow rate with transition markers
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
        row=2, col=1
    )

    # Add transition markers as scatter points instead of vlines (much faster)
    if len(transition_times) > 0:
        transition_minutes = transition_times / 60
        # Find flow values at transition points
        transition_flow_indices = np.searchsorted(time_viz, transition_samples)
        transition_flow_indices = np.clip(transition_flow_indices, 0, len(flow_viz)-1)
        transition_flow_vals = flow_viz[transition_flow_indices]

        fig.add_trace(
            go.Scatter(
                x=transition_minutes,
                y=transition_flow_vals,
                mode='markers',
                marker=dict(size=8, color='red', symbol='line-ns', line=dict(width=2)),
                name='Transitions',
                showlegend=False,
                hovertemplate='Transition<extra></extra>'
            ),
            row=2, col=1
        )

    # 3. Derivative with transition markers
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
        row=3, col=1
    )

    # Add transition markers
    if len(transition_times) > 0:
        transition_deriv_vals = flow_derivative_viz[transition_flow_indices]

        fig.add_trace(
            go.Scatter(
                x=transition_minutes,
                y=transition_deriv_vals,
                mode='markers',
                marker=dict(size=8, color='red', symbol='line-ns', line=dict(width=2)),
                name='Transitions',
                showlegend=False,
                hovertemplate='Transition<extra></extra>'
            ),
            row=3, col=1
        )

    # 4. State vector showing oscillator behavior
    fig.add_trace(
        go.Scatter(
            x=window_time_minutes,
            y=state_vector,
            mode='lines',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)',
            name='State',
            showlegend=False,
            hovertemplate='Time: %{x:.1f} min<br>State: %{y:.2f}<extra></extra>'
        ),
        row=4, col=1
    )

    # Mark transitions
    transition_state_indices = []
    for t_sec in transition_times:
        # Find closest window
        closest_idx = np.argmin(np.abs(window_centers / sample_rate - t_sec))
        transition_state_indices.append(closest_idx)

    if len(transition_state_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=window_time_minutes[transition_state_indices],
                y=state_vector[transition_state_indices],
                mode='markers',
                marker=dict(size=8, color='red', symbol='x'),
                name='Transitions',
                showlegend=False,
                hovertemplate='Transition<extra></extra>'
            ),
            row=4, col=1
        )

    # Update axes
    print("Formatting axes and layout...")
    fig.update_xaxes(title_text="Flow Rate (L/s)", row=1, col=1)
    fig.update_yaxes(title_text="dFlow/dt (L/s²)", row=1, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    fig.update_yaxes(title_text="Flow (L/s)", row=2, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    fig.update_yaxes(title_text="dFlow/dt (L/s²)", row=3, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=4, col=1)
    fig.update_yaxes(title_text="State Index", row=4, col=1)

    # Layout
    fig.update_layout(
        title=f"Phase Transition Analysis: {filename}<br>" +
              f"<sub>Transitions/Hour: {transitions_per_hour:.2f} | " +
              f"Total Transitions: {len(transition_times)} | " +
              f"Duration: {duration_hours:.1f} hours</sub>",
        height=1400,
        hovermode='closest'
    )

    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{filename}_phase_transitions',
            'height': 1400,
            'width': 1600,
            'scale': 2
        },
        'displaylogo': False
    }

    return fig, config, transitions_per_hour

def main():
    print("=" * 70)
    print("OscilloBreath - Phase Transition Analyzer")
    print("Detecting oscillator state changes")
    print("=" * 70)
    print()

    # Select file
    data_path = select_data_file()
    if not data_path:
        print("No file selected. Exiting.")
        return

    filename = Path(data_path).stem

    # Read data (automatically detects ResMed or Philips format)
    print()
    flow, sample_rate, data_type = load_flow_data(data_path)

    # Detect transitions
    print()
    print("Analyzing oscillator behavior...")
    transition_data = detect_phase_transitions(flow, sample_rate)

    duration_hours = len(flow) / sample_rate / 3600
    num_transitions = len(transition_data['transition_times'])
    transitions_per_hour = num_transitions / duration_hours

    print()
    print("=" * 70)
    print(f"PHASE TRANSITION ANALYSIS")
    print("=" * 70)
    print(f"Total transitions detected: {num_transitions}")
    print(f"Duration: {duration_hours:.2f} hours")
    print(f"Transitions per hour: {transitions_per_hour:.2f}")
    print()
    print("Transition times (minutes):")
    for t in transition_data['transition_times']:
        print(f"  {t/60:.1f}")
    print("=" * 70)
    print()

    # Create plot
    print("Creating visualization...")
    fig, config, tph = create_phase_transition_plot(flow, sample_rate, transition_data, filename)

    # Save to output folder
    print("Saving HTML file...")
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{filename}_phase_transitions.html"
    fig.write_html(str(output_file), config=config)

    print()
    print("=" * 70)
    print(f"Analysis complete!")
    print(f"Saved to: {output_file}")
    print(f"")
    print(f"*** TRANSITIONS PER HOUR: {tph:.2f} ***")
    print("=" * 70)
    print()
    print("Opening in browser...")

    # Open in browser
    webbrowser.open(f'file://{output_file.absolute()}')

if __name__ == "__main__":
    main()
