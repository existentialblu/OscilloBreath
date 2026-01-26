import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import load_flow_data
from tkinter import Tk, filedialog
from pathlib import Path
import webbrowser

def select_data_file():
    """Open file picker to select data file or folder"""
    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring to front

    file_path = filedialog.askopenfilename(
        title="Select data file (ResMed EDF or Philips folder)",
        filetypes=[("Data files", "*.edf *.005 *.006"), ("All files", "*.*")]
    )

    # If they selected a Philips file, use its parent folder
    if file_path and Path(file_path).suffix in ['.005', '.006', '.000', '.001', '.002']:
        file_path = str(Path(file_path).parent)

    root.destroy()
    return file_path

def calculate_derivative(flow, sample_rate):
    """Calculate rate of change of flow"""
    # Use numpy gradient for smooth derivative
    dt = 1.0 / sample_rate
    flow_derivative = np.gradient(flow, dt)
    return flow_derivative

def create_phase_space_plot(flow, flow_derivative, sample_rate, filename):
    """Create interactive phase space plot with time color-coding"""

    # Create time array in minutes for color coding
    time_minutes = np.arange(len(flow)) / sample_rate / 60

    # Create subplots: phase space, raw flow, and derivative
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Phase Space Portrait (Flow vs Rate of Change)',
            'Flow Rate Over Time',
            'Flow Rate Derivative Over Time'
        ),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )

    # Phase space plot (main attraction!)
    fig.add_trace(
        go.Scattergl(
            x=flow,
            y=flow_derivative,
            mode='markers',
            marker=dict(
                size=2,
                color=time_minutes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Time<br>(min)",
                    x=1.15,
                    len=0.4,
                    y=0.75
                ),
                opacity=0.6
            ),
            name='Phase Space',
            hovertemplate='Flow: %{x:.2f}<br>dFlow/dt: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Raw flow over time
    fig.add_trace(
        go.Scattergl(
            x=time_minutes,
            y=flow,
            mode='lines',
            line=dict(color='blue', width=0.5),
            name='Flow Rate',
            hovertemplate='Time: %{x:.1f} min<br>Flow: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Derivative over time
    fig.add_trace(
        go.Scattergl(
            x=time_minutes,
            y=flow_derivative,
            mode='lines',
            line=dict(color='red', width=0.5),
            name='dFlow/dt',
            hovertemplate='Time: %{x:.1f} min<br>dFlow/dt: %{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )

    # Update axes labels
    fig.update_xaxes(title_text="Flow Rate (L/s)", row=1, col=1)
    fig.update_yaxes(title_text="dFlow/dt (L/s²)", row=1, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    fig.update_yaxes(title_text="Flow (L/s)", row=2, col=1)

    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    fig.update_yaxes(title_text="dFlow/dt (L/s²)", row=3, col=1)

    # Update layout with screenshot button enabled
    fig.update_layout(
        title=f"OscilloBreath Analysis: {filename}",
        height=1200,
        showlegend=False,
        hovermode='closest'
    )

    # Configure the modebar with screenshot options
    config = {
        'toImageButtonOptions': {
            'format': 'png',  # one of png, svg, jpeg, webp
            'filename': f'{filename}_phase_space',
            'height': 1200,
            'width': 1400,
            'scale': 2  # Multiply resolution for crisp images
        },
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
    }

    return fig, config

def main():
    print("=" * 60)
    print("OscilloBreath - Coupled Oscillator Analysis")
    print("=" * 60)
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

    # Calculate derivative
    print("Calculating flow derivative...")
    flow_derivative = calculate_derivative(flow, sample_rate)

    # Create plot
    print("Creating interactive phase space plot...")
    fig, config = create_phase_space_plot(flow, flow_derivative, sample_rate, filename)

    # Save to output folder
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{filename}_phase_space.html"
    fig.write_html(str(output_file), config=config)

    print()
    print("=" * 60)
    print(f"Analysis complete!")
    print(f"Saved to: {output_file}")
    print("=" * 60)
    print()
    print("Opening in browser...")

    # Open in browser
    webbrowser.open(f'file://{output_file.absolute()}')

if __name__ == "__main__":
    main()
