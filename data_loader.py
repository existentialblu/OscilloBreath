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
