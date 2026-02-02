"""
Universal data loader for OscilloBreath
Supports both ResMed (EDF) and Philips (encrypted/unencrypted) formats
"""

from pathlib import Path
import pyedflib
from philips_loader import load_philips_flow_data
import numpy as np


def detect_data_type(path):
    """
    Detect whether path contains ResMed or Philips data

    Returns: 'resmed', 'philips', or None
    """
    path = Path(path)

    if path.is_file():
        # Single file - check extension
        if path.suffix.lower() == '.edf':
            return 'resmed'
        elif path.suffix.lower() in ['.000', '.001', '.002', '.005', '.006']:
            return 'philips'

    elif path.is_dir():
        # Directory - check what files are inside
        files = list(path.iterdir())

        # Check for ResMed EDF files
        edf_files = [f for f in files if f.suffix.lower() == '.edf' and '_BRP' in f.stem]
        if edf_files:
            return 'resmed'

        # Check for Philips waveform files
        philips_files = [f for f in files if f.suffix.lower() in ['.005', '.006']]
        if philips_files:
            return 'philips'

    return None


def load_flow_data(path):
    """
    Universal loader - automatically detects format and loads flow data

    Args:
        path: Path to EDF file (ResMed) or folder containing Philips data

    Returns:
        (flow_array, sample_rate, data_type)
    """
    path = Path(path)
    data_type = detect_data_type(path)

    if data_type == 'resmed':
        print(f"Detected ResMed EDF format")
        flow, sample_rate = load_resmed_edf(path)
        return flow, sample_rate, 'resmed'

    elif data_type == 'philips':
        print(f"Detected Philips format")

        # If single file, use its parent directory
        if path.is_file():
            folder = path.parent
        else:
            folder = path

        flow, sample_rate = load_philips_flow_data(folder)
        return flow, sample_rate, 'philips'

    else:
        raise ValueError(f"Could not detect data format for: {path}")


def load_resmed_edf(edf_path):
    """
    Load flow data from ResMed EDF file
    (Original OscilloBreath implementation)
    """
    print(f"Reading ResMed EDF: {Path(edf_path).name}")

    with pyedflib.EdfReader(str(edf_path)) as f:
        signal_labels = f.getSignalLabels()

        # Find flow signal
        flow_keywords = ['flow', 'Flow', 'FLOW']
        flow_idx = None

        for i, label in enumerate(signal_labels):
            if any(keyword in label for keyword in flow_keywords):
                flow_idx = i
                print(f"  Found flow signal: {label}")
                break

        if flow_idx is None:
            raise ValueError("Could not find flow signal in EDF file")

        flow = f.readSignal(flow_idx)
        sample_rate = f.getSampleFrequency(flow_idx)

        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {len(flow)/sample_rate/3600:.2f} hours")

    return flow, sample_rate


def load_resmed_edf_with_pressure(edf_path, verbose=True):
    """
    Load flow AND pressure data from ResMed EDF file.

    ResMed BRP files contain:
    - Flow.40ms (index 0): Respiratory flow at 25 Hz
    - Press.40ms (index 1): Mask pressure at 25 Hz (shows IPAP/EPAP cycling)

    Args:
        edf_path: path to EDF file
        verbose: print info about signals found

    Returns:
        dict with keys:
            'flow': numpy array of flow data (L/s)
            'pressure': numpy array of pressure data (cmH2O), or None if not found
            'sample_rate': sample rate in Hz
            'signals_found': list of signal labels found
    """
    edf_path = Path(edf_path)

    if verbose:
        print(f"Reading ResMed EDF: {edf_path.name}")

    with pyedflib.EdfReader(str(edf_path)) as f:
        signal_labels = f.getSignalLabels()

        if verbose:
            print(f"  Signals in file: {signal_labels}")

        # Find flow signal
        flow_keywords = ['flow', 'Flow', 'FLOW']
        flow_idx = None

        for i, label in enumerate(signal_labels):
            if any(keyword in label for keyword in flow_keywords):
                flow_idx = i
                if verbose:
                    print(f"  Found flow signal: {label} (index {i})")
                break

        if flow_idx is None:
            raise ValueError("Could not find flow signal in EDF file")

        # Find pressure signal
        # ResMed uses "Press.40ms" for high-res pressure in BRP files
        # May also see "Insp Pres", "Exp Pres", "Mask Pres" in other file types
        pressure_keywords = ['press', 'Press', 'PRESS', 'IPAP', 'ipap']
        pressure_idx = None

        for i, label in enumerate(signal_labels):
            if any(keyword in label for keyword in pressure_keywords):
                # Skip checksum signals
                if 'crc' in label.lower():
                    continue
                pressure_idx = i
                if verbose:
                    print(f"  Found pressure signal: {label} (index {i})")
                break

        # Read flow
        flow = f.readSignal(flow_idx)
        sample_rate = f.getSampleFrequency(flow_idx)

        # Read pressure if found
        pressure = None
        if pressure_idx is not None:
            pressure = f.readSignal(pressure_idx)
            pressure_sr = f.getSampleFrequency(pressure_idx)

            # Verify sample rates match
            if pressure_sr != sample_rate:
                if verbose:
                    print(f"  WARNING: Pressure sample rate ({pressure_sr} Hz) differs from flow ({sample_rate} Hz)")
                # Resample pressure to match flow if needed
                if len(pressure) != len(flow):
                    # Simple linear interpolation
                    pressure_time = np.arange(len(pressure)) / pressure_sr
                    flow_time = np.arange(len(flow)) / sample_rate
                    pressure = np.interp(flow_time, pressure_time, pressure)
                    if verbose:
                        print(f"  Resampled pressure to match flow")

            if verbose:
                print(f"  Pressure range: {pressure.min():.1f} to {pressure.max():.1f} cmH2O")
        else:
            if verbose:
                print(f"  No pressure signal found")

        if verbose:
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Duration: {len(flow)/sample_rate/3600:.2f} hours")

    return {
        'flow': flow,
        'pressure': pressure,
        'sample_rate': sample_rate,
        'signals_found': signal_labels
    }


def load_flow_and_pressure(path, verbose=True):
    """
    Universal loader that returns both flow and pressure data.

    Args:
        path: Path to EDF file (ResMed) or folder containing Philips data
        verbose: print loading info

    Returns:
        dict with keys:
            'flow': numpy array of flow data
            'pressure': numpy array of pressure data (or None if not available)
            'sample_rate': sample rate in Hz
            'data_type': 'resmed' or 'philips'
    """
    path = Path(path)
    data_type = detect_data_type(path)

    if data_type == 'resmed':
        if verbose:
            print(f"Detected ResMed EDF format")
        result = load_resmed_edf_with_pressure(path, verbose=verbose)
        result['data_type'] = 'resmed'
        return result

    elif data_type == 'philips':
        if verbose:
            print(f"Detected Philips format")

        # If single file, use its parent directory
        if path.is_file():
            folder = path.parent
        else:
            folder = path

        # Philips loader already returns pressure but we're not using it
        # For now, just return flow with no pressure
        flow, sample_rate = load_philips_flow_data(folder)
        return {
            'flow': flow,
            'pressure': None,  # TODO: extract from Philips loader
            'sample_rate': sample_rate,
            'data_type': 'philips'
        }

    else:
        raise ValueError(f"Could not detect data format for: {path}")


# Test function
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <path_to_data>")
        print("  For ResMed: path to .edf file")
        print("  For Philips: path to folder containing .005/.006 files")
        sys.exit(1)

    path = sys.argv[1]

    try:
        flow, sr, dtype = load_flow_data(path)

        print(f"\nSuccess!")
        print(f"  Data type: {dtype}")
        print(f"  Samples: {len(flow)}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(flow)/sr/3600:.2f} hours")
        print(f"  Flow range: {flow.min():.2f} to {flow.max():.2f}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
