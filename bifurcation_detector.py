"""
OscilloBreath - Bifurcation Detector

The core insight: healthy breathing lives "just into strange" — slightly positive
Lyapunov exponents indicating bounded chaos with adaptive flexibility. Pathological
breathing gets trapped in rigid limit cycles (LLE near zero) or escapes into
instability. We want to find the *bifurcation point* — the phase transition.

Instead of looking at the collapse itself (boring, you're already trapped), this
module looks at the approach to collapse:

1. Scan the whole night for the window with LOWEST LLE (maximum periodicity)
2. Extract the window immediately PRECEDING that collapse
3. Compute a "Respiratory Reynolds" ratio from the pre-collapse window
4. Compare pre-collapse windows to stable-chaos windows to find the threshold

The Respiratory Reynolds number is, like the original, a dimensionless ratio:
    RR = (destabilizing forces) / (stabilizing forces)

In fluid dynamics: RR = inertial/viscous. Here we're trying to find what predicts
"about to fall into limit cycle" vs "stable adaptive chaos."

Candidate formulas (TBD which works best):
- RR1: derivative_violence / flow_variability (simplest)
- RR2: lle_instability / flow_variability (uses rate of LLE change)
- RR3: attractor_contraction / flow_variability (phase space geometry)

Once we add IPAP data:
- RR_full: (IPAP_variability * lle_sensitivity) / flow_variability
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tkinter import Tk, filedialog
from pathlib import Path
import webbrowser
from scipy.signal import savgol_filter
from data_loader import load_flow_data, load_flow_and_pressure

# Import LLE computation from the fast analyzer
from lyapunov_analyzer_fast import time_delay_embedding, calculate_lle_rosenstein


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


def compute_windowed_lle(flow, sample_rate, window_seconds=300, overlap=0.5, verbose=True):
    """
    Compute LLE for each window across the entire night.

    This is the expensive part — we're computing full LLE for every window.
    Using 5-minute windows (300s) to match standard sleep medicine epochs,
    with 50% overlap for temporal resolution.

    Args:
        flow: 1D numpy array of flow data at native sample rate
        sample_rate: Hz
        window_seconds: window size (default 300 = 5 minutes)
        overlap: fraction overlap between windows (default 0.5)
        verbose: print progress

    Returns:
        dict with:
            'lle_values': array of LLE for each window
            'window_centers_sec': center time of each window in seconds
            'window_indices': (start, end) sample indices for each window
    """
    # Downsample for LLE (same as lyapunov_analyzer_fast)
    target_fs = 2.0
    downsample_factor = int(sample_rate / target_fs)
    flow_downsampled = flow[::downsample_factor]
    fs_downsampled = sample_rate / downsample_factor

    # Embedding parameters
    tau = int(fs_downsampled * 1.0)  # 1 second delay
    embedding_dim = 3

    # Window parameters (in downsampled space)
    window_samples = int(window_seconds * fs_downsampled)
    hop_samples = int(window_samples * (1 - overlap))

    num_windows = (len(flow_downsampled) - window_samples) // hop_samples + 1

    if verbose:
        print(f"Computing windowed LLE:")
        print(f"  Windows: {num_windows}")
        print(f"  Window size: {window_seconds}s ({window_samples} samples at {fs_downsampled:.1f} Hz)")
        print(f"  Overlap: {overlap*100:.0f}%")
        print()

    lle_values = []
    window_centers_sec = []
    window_indices = []  # In original sample space

    for i in range(num_windows):
        start_idx_ds = i * hop_samples
        end_idx_ds = start_idx_ds + window_samples

        if end_idx_ds > len(flow_downsampled):
            break

        window_flow = flow_downsampled[start_idx_ds:end_idx_ds]

        # Convert back to original sample space for tracking
        start_idx_orig = start_idx_ds * downsample_factor
        end_idx_orig = end_idx_ds * downsample_factor
        center_sec = (start_idx_orig + end_idx_orig) / 2 / sample_rate

        if verbose and i % 5 == 0:
            print(f"  Window {i+1}/{num_windows} (center: {center_sec/60:.1f} min)...")

        # Compute LLE for this window (suppress individual window output)
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            lle = calculate_lle_rosenstein(
                window_flow, tau, embedding_dim, fs_downsampled, max_iter=50
            )
        finally:
            sys.stdout = old_stdout

        if lle is not None:
            lle_values.append(lle)
            window_centers_sec.append(center_sec)
            window_indices.append((start_idx_orig, min(end_idx_orig, len(flow))))
        else:
            # Mark failed windows with NaN
            lle_values.append(np.nan)
            window_centers_sec.append(center_sec)
            window_indices.append((start_idx_orig, min(end_idx_orig, len(flow))))

    return {
        'lle_values': np.array(lle_values),
        'window_centers_sec': np.array(window_centers_sec),
        'window_indices': window_indices
    }


def find_minimum_lle_window(lle_data, min_minutes_from_start=10):
    """
    Find the window with the LOWEST (most periodic) LLE.

    This is the "collapse" point — where the oscillator is most stuck
    in a limit cycle. We skip the first N minutes because sleep onset
    often has weird transients.

    Args:
        lle_data: output from compute_windowed_lle()
        min_minutes_from_start: skip windows before this time (default 10)

    Returns:
        dict with:
            'min_lle': the lowest LLE value
            'min_idx': index into lle_data arrays
            'min_time_sec': time in seconds of the minimum
            'window_bounds': (start_sample, end_sample) in original space
    """
    lle_values = lle_data['lle_values']
    times = lle_data['window_centers_sec']

    # Mask out early windows and NaN values
    valid_mask = (times >= min_minutes_from_start * 60) & ~np.isnan(lle_values)

    if not np.any(valid_mask):
        raise ValueError("No valid LLE windows found after filtering")

    # Find minimum among valid windows
    valid_indices = np.where(valid_mask)[0]
    valid_lle = lle_values[valid_mask]

    min_valid_idx = np.argmin(valid_lle)
    min_idx = valid_indices[min_valid_idx]

    return {
        'min_lle': lle_values[min_idx],
        'min_idx': min_idx,
        'min_time_sec': times[min_idx],
        'window_bounds': lle_data['window_indices'][min_idx]
    }


def extract_preceding_window(flow, sample_rate, collapse_time_sec, lookback_sec=300, pressure=None):
    """
    Extract the window BEFORE the collapse point.

    This is the window we analyze to predict collapse. If collapse is at T,
    we extract [T - lookback_sec, T].

    Args:
        flow: full flow array
        sample_rate: Hz
        collapse_time_sec: when collapse occurs (center of min-LLE window)
        lookback_sec: how far back to look (default 300 = 5 minutes)
        pressure: full pressure array (optional, same length as flow)

    Returns:
        dict with:
            'flow': flow data for preceding window
            'pressure': pressure data for preceding window (or None)
            'start_sec': start time
            'end_sec': end time
            'start_sample': start index
            'end_sample': end index
    """
    end_sample = int(collapse_time_sec * sample_rate)
    start_sample = max(0, end_sample - int(lookback_sec * sample_rate))

    result = {
        'flow': flow[start_sample:end_sample],
        'pressure': None,
        'start_sec': start_sample / sample_rate,
        'end_sec': end_sample / sample_rate,
        'start_sample': start_sample,
        'end_sample': end_sample
    }

    if pressure is not None:
        result['pressure'] = pressure[start_sample:end_sample]

    return result


def compute_respiratory_reynolds(window_flow, sample_rate, window_pressure=None, method='derivative_violence'):
    """
    Compute candidate Respiratory Reynolds numbers.

    The Reynolds number is dimensionless: (destabilizing) / (stabilizing).
    We're looking for the ratio that best predicts regime transition.

    Flow-only methods:
    - derivative_violence, acceleration_ratio, energy_ratio, jerk_ratio, cv_ratio

    Pressure-based methods (when window_pressure provided):
    - ipap_variability: std(pressure) / mean(pressure)
    - ipap_rate: mean(|dP/dt|) — how fast pressure is changing
    - ipap_flow_ratio: pressure_variability / flow_variability
    - pressure_energy: pressure derivative energy / flow derivative energy
    - reynolds_full: (ipap_variability * energy_ratio) / flow_std — the full formula

    Args:
        window_flow: flow data for the window
        sample_rate: Hz
        window_pressure: pressure data for the window (optional, same length as flow)
        method: which formula to use

    Returns:
        dict with computed Reynolds values (one per method or all if 'all')
    """
    dt = 1.0 / sample_rate

    # Compute derivatives
    flow_deriv = np.gradient(window_flow, dt)
    flow_deriv2 = np.gradient(flow_deriv, dt)  # acceleration

    # Core statistics
    flow_std = np.std(window_flow)
    flow_range = np.ptp(window_flow)
    deriv_std = np.std(flow_deriv)
    deriv_range = np.ptp(flow_deriv)
    deriv_mean_abs = np.mean(np.abs(flow_deriv))
    accel_mean_abs = np.mean(np.abs(flow_deriv2))

    # Avoid division by zero
    eps = 1e-10

    results = {}

    # Method 1: Derivative violence
    # High = violent breathing relative to depth variability
    violence = deriv_range / (deriv_std + eps)
    rr_violence = violence / (flow_std + eps)
    results['derivative_violence'] = rr_violence

    # Method 2: Acceleration ratio
    # High = lots of acceleration relative to flow variability
    rr_accel = accel_mean_abs / (flow_std + eps)
    results['acceleration_ratio'] = rr_accel

    # Method 3: Normalized derivative energy
    # High = derivative energy dominates flow energy
    deriv_energy = np.sum(flow_deriv**2)
    flow_energy = np.sum(window_flow**2) + eps
    rr_energy = deriv_energy / flow_energy
    results['energy_ratio'] = rr_energy

    # Method 4: Jerk ratio (rate of acceleration change)
    # High = breathing is "jerky" relative to smooth
    flow_deriv3 = np.gradient(flow_deriv2, dt)
    jerk_mean = np.mean(np.abs(flow_deriv3))
    rr_jerk = jerk_mean / (flow_std + eps)
    results['jerk_ratio'] = rr_jerk

    # Method 5: Coefficient of variation of derivative
    # High = derivative is highly variable relative to its mean
    cv_deriv = deriv_std / (deriv_mean_abs + eps)
    rr_cv = cv_deriv / (flow_std / (np.mean(np.abs(window_flow)) + eps) + eps)
    results['cv_ratio'] = rr_cv

    # =========================================================================
    # PRESSURE-BASED METHODS (when IPAP data available)
    # =========================================================================
    if window_pressure is not None and len(window_pressure) > 0:
        # Pressure statistics
        pressure_mean = np.mean(window_pressure)
        pressure_std = np.std(window_pressure)
        pressure_range = np.ptp(window_pressure)

        # Pressure derivatives
        pressure_deriv = np.gradient(window_pressure, dt)
        pressure_deriv_abs_mean = np.mean(np.abs(pressure_deriv))

        # Method 6: IPAP variability (coefficient of variation)
        # High = pressure is varying a lot relative to its mean
        # This captures the machine "hunting" or compensating
        ipap_cv = pressure_std / (pressure_mean + eps)
        results['ipap_variability'] = ipap_cv

        # Method 7: IPAP rate of change
        # High = pressure is changing rapidly (machine actively adjusting)
        results['ipap_rate'] = pressure_deriv_abs_mean

        # Method 8: IPAP/Flow variability ratio
        # High = pressure more variable than flow (machine fighting the airway)
        flow_cv = flow_std / (np.mean(np.abs(window_flow)) + eps)
        results['ipap_flow_ratio'] = ipap_cv / (flow_cv + eps)

        # Method 9: Pressure energy ratio
        # Compares pressure derivative energy to flow derivative energy
        # High = machine is "working hard" relative to flow changes
        pressure_deriv_energy = np.sum(pressure_deriv**2)
        flow_deriv_energy = np.sum(flow_deriv**2) + eps
        results['pressure_energy'] = pressure_deriv_energy / flow_deriv_energy

        # Method 10: FULL REYNOLDS NUMBER
        # The original hypothesis: (IPAP_variability × energy_ratio) / flow_variability
        # Combines machine response (ipap_cv) with breathing inefficiency (energy_ratio)
        # normalized by how much the flow itself is varying
        results['reynolds_full'] = (ipap_cv * rr_energy) / (flow_std + eps)

        # Method 11: Pressure-flow coupling
        # Cross-correlation at zero lag - how synchronized are pressure and flow?
        # High positive = pressure tracks flow (passive)
        # Low/negative = pressure fighting flow (active compensation)
        if len(window_pressure) == len(window_flow):
            # Normalize both signals
            flow_norm = (window_flow - np.mean(window_flow)) / (flow_std + eps)
            pres_norm = (window_pressure - pressure_mean) / (pressure_std + eps)
            coupling = np.mean(flow_norm * pres_norm)
            results['pressure_flow_coupling'] = coupling

        # Method 12: Pressure trend (is IPAP increasing over window?)
        # Fit linear trend to pressure
        time_vec = np.arange(len(window_pressure))
        if len(time_vec) > 10:
            slope = np.polyfit(time_vec, window_pressure, 1)[0]
            # Normalize by sample rate to get cmH2O per second
            results['ipap_trend'] = slope * sample_rate

    # Return requested method or all
    if method == 'all':
        return results
    else:
        return {method: results.get(method, results['derivative_violence'])}


def find_stable_chaos_windows(lle_data, percentile=75, min_lle=0.01):
    """
    Find windows representing "stable chaos" — healthy adaptive breathing.

    These are windows with POSITIVE LLE (chaotic) but not too extreme.
    We use these as the comparison group for pre-collapse windows.

    Args:
        lle_data: output from compute_windowed_lle()
        percentile: take windows above this percentile of LLE (default 75th)
        min_lle: minimum LLE to count as "chaotic" (default 0.01)

    Returns:
        list of indices into lle_data representing stable-chaos windows
    """
    lle_values = lle_data['lle_values']
    valid_mask = ~np.isnan(lle_values)

    # Find chaotic windows (positive LLE)
    chaotic_mask = valid_mask & (lle_values >= min_lle)

    if not np.any(chaotic_mask):
        print("WARNING: No windows with positive LLE found")
        # Fall back to top percentile of whatever we have
        threshold = np.nanpercentile(lle_values, percentile)
        return list(np.where(valid_mask & (lle_values >= threshold))[0])

    # Among chaotic windows, take top percentile (most robustly chaotic)
    chaotic_values = lle_values[chaotic_mask]
    threshold = np.percentile(chaotic_values, percentile)

    stable_chaos_indices = np.where(chaotic_mask & (lle_values >= threshold))[0]

    return list(stable_chaos_indices)


def characterize_windows(flow, sample_rate, lle_data, pre_collapse_idx, stable_chaos_indices, pressure=None):
    """
    Compute Reynolds numbers for pre-collapse vs stable-chaos windows.

    This is the comparison that will let us find the bifurcation threshold.

    Args:
        flow: full flow array
        sample_rate: Hz
        lle_data: output from compute_windowed_lle()
        pre_collapse_idx: index of the pre-collapse window
        stable_chaos_indices: list of indices for stable chaos windows
        pressure: full pressure array (optional, same length as flow)

    Returns:
        dict with:
            'pre_collapse': Reynolds values for pre-collapse window
            'stable_chaos': list of Reynolds values for stable windows
            'threshold_candidates': suggested threshold values
    """
    window_indices = lle_data['window_indices']

    # Compute Reynolds for pre-collapse window
    start, end = window_indices[pre_collapse_idx]
    pre_collapse_flow = flow[start:end]
    pre_collapse_pressure = pressure[start:end] if pressure is not None else None
    pre_collapse_rr = compute_respiratory_reynolds(pre_collapse_flow, sample_rate,
                                                    window_pressure=pre_collapse_pressure, method='all')

    # Compute Reynolds for each stable-chaos window
    stable_rr_list = []
    for idx in stable_chaos_indices:
        start, end = window_indices[idx]
        window_flow = flow[start:end]
        window_pressure = pressure[start:end] if pressure is not None else None
        rr = compute_respiratory_reynolds(window_flow, sample_rate,
                                          window_pressure=window_pressure, method='all')
        stable_rr_list.append(rr)

    # Compute threshold candidates (midpoint between pre-collapse and stable means)
    threshold_candidates = {}

    for method in pre_collapse_rr.keys():
        pre_val = pre_collapse_rr[method]
        stable_vals = [rr[method] for rr in stable_rr_list]
        stable_mean = np.mean(stable_vals)
        stable_std = np.std(stable_vals)

        # Threshold = midpoint between pre-collapse and stable mean
        threshold = (pre_val + stable_mean) / 2

        # Also compute separation (how well does this method separate the classes?)
        separation = abs(pre_val - stable_mean) / (stable_std + 1e-10)

        threshold_candidates[method] = {
            'threshold': threshold,
            'pre_collapse_value': pre_val,
            'stable_mean': stable_mean,
            'stable_std': stable_std,
            'separation': separation  # Higher = better discriminator
        }

    return {
        'pre_collapse': pre_collapse_rr,
        'stable_chaos': stable_rr_list,
        'threshold_candidates': threshold_candidates
    }


def create_bifurcation_plot(flow, sample_rate, lle_data, min_lle_info,
                            pre_collapse_window, characterization, filename):
    """
    Create visualization of bifurcation detection results.
    """
    lle_values = lle_data['lle_values']
    times_min = lle_data['window_centers_sec'] / 60

    # Prepare flow visualization (downsampled)
    downsample_factor = max(1, len(flow) // 50000)
    flow_viz = flow[::downsample_factor]
    time_viz_min = np.arange(len(flow_viz)) * downsample_factor / sample_rate / 60

    # Derivative for phase space
    dt = 1.0 / sample_rate * downsample_factor
    flow_deriv_viz = np.gradient(flow_viz, dt)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'LLE Over Time (Collapse = Minimum)',
            'Respiratory Reynolds Comparison',
            'Flow with Collapse Region Marked',
            'Phase Space',
            'Pre-Collapse Window Detail',
            'Reynolds Method Separation Scores'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )

    # 1. LLE over time
    fig.add_trace(
        go.Scatter(
            x=times_min,
            y=lle_values,
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(color='blue', width=1),
            name='LLE',
            hovertemplate='Time: %{x:.1f} min<br>LLE: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Mark minimum (collapse point)
    fig.add_trace(
        go.Scatter(
            x=[min_lle_info['min_time_sec'] / 60],
            y=[min_lle_info['min_lle']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x', line=dict(width=3)),
            name='Collapse (min LLE)',
            hovertemplate='COLLAPSE<br>Time: %{x:.1f} min<br>LLE: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Mark pre-collapse window
    pre_start_min = pre_collapse_window['start_sec'] / 60
    pre_end_min = pre_collapse_window['end_sec'] / 60

    fig.add_vrect(
        x0=pre_start_min, x1=pre_end_min,
        fillcolor="orange", opacity=0.3,
        layer="below", line_width=0,
        row=1, col=1
    )

    # Add zero line for LLE interpretation
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    # 2. Reynolds comparison bar chart
    methods = list(characterization['pre_collapse'].keys())
    pre_vals = [characterization['pre_collapse'][m] for m in methods]
    stable_means = [characterization['threshold_candidates'][m]['stable_mean'] for m in methods]
    stable_stds = [characterization['threshold_candidates'][m]['stable_std'] for m in methods]

    fig.add_trace(
        go.Bar(
            name='Pre-Collapse',
            x=methods,
            y=pre_vals,
            marker_color='red',
            opacity=0.7
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(
            name='Stable Chaos (mean)',
            x=methods,
            y=stable_means,
            error_y=dict(type='data', array=stable_stds),
            marker_color='green',
            opacity=0.7
        ),
        row=1, col=2
    )

    # 3. Flow with regions marked
    fig.add_trace(
        go.Scattergl(
            x=time_viz_min,
            y=flow_viz,
            mode='lines',
            line=dict(color='blue', width=0.5),
            name='Flow',
            showlegend=False
        ),
        row=2, col=1
    )

    # Mark collapse region
    collapse_start = min_lle_info['window_bounds'][0] / sample_rate / 60
    collapse_end = min_lle_info['window_bounds'][1] / sample_rate / 60

    fig.add_vrect(
        x0=collapse_start, x1=collapse_end,
        fillcolor="red", opacity=0.2,
        layer="below", line_width=0,
        row=2, col=1
    )

    fig.add_vrect(
        x0=pre_start_min, x1=pre_end_min,
        fillcolor="orange", opacity=0.3,
        layer="below", line_width=0,
        row=2, col=1
    )

    # 4. Phase space
    fig.add_trace(
        go.Scattergl(
            x=flow_viz,
            y=flow_deriv_viz,
            mode='markers',
            marker=dict(size=1, color=time_viz_min, colorscale='Viridis', opacity=0.3),
            name='Phase Space',
            showlegend=False
        ),
        row=2, col=2
    )

    # 5. Pre-collapse window detail
    pre_flow = pre_collapse_window['flow']
    pre_time = np.arange(len(pre_flow)) / sample_rate

    fig.add_trace(
        go.Scatter(
            x=pre_time,
            y=pre_flow,
            mode='lines',
            line=dict(color='orange', width=1),
            name='Pre-Collapse Flow',
            showlegend=False
        ),
        row=3, col=1
    )

    # 6. Separation scores (which method best discriminates?)
    separations = [characterization['threshold_candidates'][m]['separation'] for m in methods]

    fig.add_trace(
        go.Bar(
            x=methods,
            y=separations,
            marker_color='purple',
            name='Separation Score',
            showlegend=False
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_xaxes(title_text="Time (minutes)", row=1, col=1)
    fig.update_yaxes(title_text="LLE (bits/s)", row=1, col=1)

    fig.update_xaxes(title_text="Method", row=1, col=2)
    fig.update_yaxes(title_text="Reynolds Value", row=1, col=2)

    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    fig.update_yaxes(title_text="Flow (L/s)", row=2, col=1)

    fig.update_xaxes(title_text="Flow (L/s)", row=2, col=2)
    fig.update_yaxes(title_text="dFlow/dt", row=2, col=2)

    fig.update_xaxes(title_text="Time in window (s)", row=3, col=1)
    fig.update_yaxes(title_text="Flow (L/s)", row=3, col=1)

    fig.update_xaxes(title_text="Method", row=3, col=2)
    fig.update_yaxes(title_text="Separation (higher=better)", row=3, col=2)

    fig.update_layout(
        title=f"Bifurcation Detection: {filename}<br>" +
              f"<sub>Collapse at {min_lle_info['min_time_sec']/60:.1f} min | " +
              f"Min LLE: {min_lle_info['min_lle']:.4f}</sub>",
        height=1200,
        barmode='group',
        showlegend=True
    )

    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'{filename}_bifurcation',
            'height': 1200,
            'width': 1600,
            'scale': 2
        },
        'displaylogo': False
    }

    return fig, config


def print_analysis_summary(min_lle_info, characterization):
    """Print human-readable analysis summary."""

    print()
    print("=" * 70)
    print("BIFURCATION ANALYSIS RESULTS")
    print("=" * 70)
    print()
    print(f"COLLAPSE DETECTED:")
    print(f"  Time: {min_lle_info['min_time_sec']/60:.1f} minutes into recording")
    print(f"  Minimum LLE: {min_lle_info['min_lle']:.4f} bits/s")

    if min_lle_info['min_lle'] < -0.01:
        print(f"  Interpretation: STRONGLY PERIODIC (over-damped limit cycle)")
    elif min_lle_info['min_lle'] < 0.01:
        print(f"  Interpretation: NEUTRAL/PERIODIC (limit cycle boundary)")
    else:
        print(f"  Interpretation: Still chaotic, but least chaotic window")

    print()
    print("RESPIRATORY REYNOLDS COMPARISON:")
    print("-" * 70)
    print(f"{'Method':<22} {'Pre-Collapse':>12} {'Stable Mean':>12} {'Separation':>12}")
    print("-" * 70)

    # Sort by separation score
    methods_sorted = sorted(
        characterization['threshold_candidates'].keys(),
        key=lambda m: characterization['threshold_candidates'][m]['separation'],
        reverse=True
    )

    for method in methods_sorted:
        tc = characterization['threshold_candidates'][method]
        print(f"{method:<22} {tc['pre_collapse_value']:>12.4f} {tc['stable_mean']:>12.4f} {tc['separation']:>12.2f}")

    print("-" * 70)
    print()

    # Recommend best method
    best_method = methods_sorted[0]
    best_sep = characterization['threshold_candidates'][best_method]['separation']

    print(f"BEST DISCRIMINATOR: {best_method}")
    print(f"  Separation score: {best_sep:.2f} standard deviations")
    print(f"  Threshold candidate: {characterization['threshold_candidates'][best_method]['threshold']:.4f}")
    print()

    if best_sep < 1.0:
        print("  WARNING: Separation is weak (<1 std). This method may not reliably")
        print("  predict bifurcation. Consider: more data, different window sizes,")
        print("  or adding IPAP data for full Reynolds formula.")
    elif best_sep < 2.0:
        print("  MODERATE separation. Method shows promise but needs validation")
        print("  across multiple nights.")
    else:
        print("  STRONG separation! This method clearly distinguishes pre-collapse")
        print("  from stable chaos. Worth pursuing further.")

    print()
    print("=" * 70)


def main():
    print("=" * 70)
    print("OscilloBreath - Bifurcation Detector")
    print("Finding the edge of chaos: where adaptive breathing becomes trapped")
    print("=" * 70)
    print()
    print("This analysis will:")
    print("  1. Compute LLE for each 5-minute window across the night")
    print("  2. Find the 'collapse' point (minimum LLE = maximum periodicity)")
    print("  3. Analyze the pre-collapse window for predictive signatures")
    print("  4. Compare against stable-chaos windows to find thresholds")
    print()
    print("WARNING: This is computationally expensive. Each window requires")
    print("full Lyapunov analysis. Expect 10-30 minutes for a full night.")
    print()

    # Select file
    data_path = select_data_file()
    if not data_path:
        print("No file selected. Exiting.")
        return

    filename = Path(data_path).stem

    try:
        # Load data (with pressure if available)
        print("Loading data...")
        data = load_flow_and_pressure(data_path)
        flow = data['flow']
        pressure = data['pressure']
        sample_rate = data['sample_rate']
        duration_hours = len(flow) / sample_rate / 3600
        print(f"  Duration: {duration_hours:.2f} hours")
        if pressure is not None:
            print(f"  IPAP data: AVAILABLE ({pressure.min():.1f} - {pressure.max():.1f} cmH2O)")
        else:
            print(f"  IPAP data: NOT AVAILABLE (flow-only analysis)")
        print()

        # Step 1: Compute windowed LLE
        print("=" * 70)
        print("STEP 1: Computing windowed Lyapunov exponents")
        print("=" * 70)
        lle_data = compute_windowed_lle(flow, sample_rate, window_seconds=300, overlap=0.5)

        valid_count = np.sum(~np.isnan(lle_data['lle_values']))
        print(f"\nComputed {valid_count} valid LLE values")
        print(f"LLE range: {np.nanmin(lle_data['lle_values']):.4f} to {np.nanmax(lle_data['lle_values']):.4f}")

        # Step 2: Find minimum LLE window
        print()
        print("=" * 70)
        print("STEP 2: Finding collapse point (minimum LLE)")
        print("=" * 70)
        min_lle_info = find_minimum_lle_window(lle_data, min_minutes_from_start=10)
        print(f"  Collapse at: {min_lle_info['min_time_sec']/60:.1f} minutes")
        print(f"  Minimum LLE: {min_lle_info['min_lle']:.4f}")

        # Step 3: Extract preceding window
        print()
        print("=" * 70)
        print("STEP 3: Extracting pre-collapse window")
        print("=" * 70)
        pre_collapse_window = extract_preceding_window(
            flow, sample_rate,
            min_lle_info['min_time_sec'],
            lookback_sec=300,
            pressure=pressure
        )
        print(f"  Pre-collapse window: {pre_collapse_window['start_sec']/60:.1f} - {pre_collapse_window['end_sec']/60:.1f} min")
        if pre_collapse_window['pressure'] is not None:
            print(f"  Pre-collapse pressure: {pre_collapse_window['pressure'].min():.1f} - {pre_collapse_window['pressure'].max():.1f} cmH2O")

        # Find the index in lle_data that corresponds to pre-collapse
        # (the window just before the minimum)
        pre_collapse_idx = max(0, min_lle_info['min_idx'] - 1)

        # Step 4: Find stable chaos windows for comparison
        print()
        print("=" * 70)
        print("STEP 4: Identifying stable-chaos windows for comparison")
        print("=" * 70)
        stable_chaos_indices = find_stable_chaos_windows(lle_data)
        print(f"  Found {len(stable_chaos_indices)} stable-chaos windows")

        if len(stable_chaos_indices) == 0:
            print("  WARNING: No stable chaos windows found!")
            print("  Using all windows for comparison instead.")
            stable_chaos_indices = list(range(len(lle_data['lle_values'])))

        # Step 5: Characterize and compare
        print()
        print("=" * 70)
        print("STEP 5: Computing Respiratory Reynolds numbers")
        if pressure is not None:
            print("(Including IPAP-based metrics)")
        print("=" * 70)
        characterization = characterize_windows(
            flow, sample_rate, lle_data,
            pre_collapse_idx, stable_chaos_indices,
            pressure=pressure
        )

        # Print summary
        print_analysis_summary(min_lle_info, characterization)

        # Create visualization
        print("Creating visualization...")
        fig, config = create_bifurcation_plot(
            flow, sample_rate, lle_data, min_lle_info,
            pre_collapse_window, characterization, filename
        )

        # Save
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{filename}_bifurcation.html"
        fig.write_html(str(output_file), config=config)

        print(f"Saved to: {output_file}")
        print()
        print("Opening in browser...")
        webbrowser.open(f'file://{output_file.absolute()}')

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
