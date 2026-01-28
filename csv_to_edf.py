import numpy as np
import pyedflib
from pathlib import Path
from datetime import datetime
from tkinter import Tk, filedialog
import sys

def select_csv_file():
    """Open file picker to select CSV file"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select synthetic data CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    root.destroy()
    return file_path

def csv_to_edf(csv_path, output_path=None, sample_rate=25):
    """
    Convert synthetic CSV data to ResMed-compatible EDF format

    Parameters:
    -----------
    csv_path : str
        Path to CSV file with columns: Time (s), Flow (L/s)
    output_path : str, optional
        Output EDF path. If None, generates based on CSV name
    sample_rate : int
        Sample rate in Hz (default 25 to match ResMed)
    """

    print("=" * 60)
    print("OscilloBreath - CSV to EDF Converter")
    print("=" * 60)
    print()

    # Read CSV
    print(f"Reading CSV: {csv_path}")
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

    time_data = data[:, 0]
    flow_data = data[:, 1]

    print(f"  Loaded {len(flow_data)} samples")
    print(f"  Duration: {time_data[-1] / 60:.2f} minutes")
    print()

    # Resample to exactly 25 Hz if needed
    original_rate = len(time_data) / time_data[-1] if len(time_data) > 1 else sample_rate
    print(f"  Original sample rate: {original_rate:.2f} Hz")

    if abs(original_rate - sample_rate) > 0.1:
        print(f"  Resampling to {sample_rate} Hz...")

        # Create uniform time grid at target sample rate
        duration = time_data[-1]
        num_samples = int(duration * sample_rate)
        new_time = np.linspace(0, duration, num_samples)

        # Interpolate flow data
        flow_data = np.interp(new_time, time_data, flow_data)
        time_data = new_time

        print(f"  Resampled to {len(flow_data)} samples")

    # Generate output path if not provided
    if output_path is None:
        csv_stem = Path(csv_path).stem
        output_dir = Path(csv_path).parent

        # Extract date from filename or use current date
        # Format: YYYYMMDD_HHMMSS_BRP.edf (ResMed format)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{timestamp}_{csv_stem}_BRP.edf"

    print()
    print(f"Creating EDF: {output_path}")

    # Create EDF file
    try:
        # Open EDF file for writing
        edf = pyedflib.EdfWriter(str(output_path), 1, file_type=pyedflib.FILETYPE_EDFPLUS)

        # Set patient info (synthetic data marker)
        edf.setPatientName("SYNTHETIC")
        edf.setPatientCode("SYNTH-001")

        # Set recording info
        edf.setTechnician("OscilloBreath Synthetic Generator")
        edf.setRecordingAdditional("Synthetic respiratory data")

        # Set start time
        start_time = datetime.now()
        edf.setStartdatetime(start_time)

        # Define the Flow signal channel
        channel_info = {
            'label': 'Flow',
            'dimension': 'L/s',
            'sample_rate': sample_rate,
            'physical_max': float(np.max(flow_data)),
            'physical_min': float(np.min(flow_data)),
            'digital_max': 32767,
            'digital_min': -32768,
            'transducer': 'Synthetic Oscillator Model',
            'prefilter': 'None'
        }

        edf.setSignalHeader(0, channel_info)

        # Write data
        print("  Writing flow data...")
        edf.writeSamples([flow_data])

        # Close file
        edf.close()

        print()
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"EDF file created: {output_path}")
        print()
        print(f"File info:")
        print(f"  Signal: Flow")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Samples: {len(flow_data)}")
        print(f"  Duration: {len(flow_data) / sample_rate / 60:.2f} minutes")
        print(f"  Flow range: {np.min(flow_data):.3f} to {np.max(flow_data):.3f} L/s")
        print()
        print("This file can now be analyzed with all OscilloBreath tools!")
        print("=" * 60)

        return str(output_path)

    except Exception as e:
        print()
        print("=" * 60)
        print("ERROR!")
        print("=" * 60)
        print(f"Failed to create EDF file: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return None

def main():
    print()

    # Check command line argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not Path(csv_path).exists():
            print(f"Error: File not found: {csv_path}")
            return
    else:
        # Interactive file selection
        csv_path = select_csv_file()
        if not csv_path:
            print("No file selected. Exiting.")
            return

    # Convert
    output_path = csv_to_edf(csv_path, sample_rate=25)

    if output_path:
        print()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
