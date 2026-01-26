"""
Philips Respironics DreamStation data loader for OscilloBreath

Based on OSCAR's prs1_loader.cpp implementation
Supports DreamStation 2 (encrypted) and older DreamStation/System One models
"""

import struct
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import numpy as np

# DreamStation 2 encryption constants (from OSCAR prs1_loader.cpp)
OSCAR_KEY = b"Patient access to their own data"  # 32 bytes
COMMON_KEY = bytes([
    0x75, 0xB3, 0xA2, 0x12, 0x4A, 0x65, 0xAF, 0x97,
    0x54, 0xD8, 0xC1, 0xF3, 0xE5, 0x2E, 0xB6, 0xF0,
    0x23, 0x20, 0x57, 0x69, 0x7E, 0x38, 0x0E, 0xC9,
    0x4A, 0xDC, 0x46, 0x45, 0xB6, 0x92, 0x5A, 0x98
])

DS2_HEADER_SIZE = 202


class PhilipsDataFile:
    """Base class for Philips data files"""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.data = None
        self.is_encrypted = False

    def read(self):
        """Read and potentially decrypt the file"""
        with open(self.filepath, 'rb') as f:
            raw_data = f.read()

        # Check if it's DreamStation 2 encrypted format
        if len(raw_data) >= 6 and raw_data[0:3] == b'\x0d\x01\x01':
            print(f"  Detected DreamStation 2 encrypted file: {self.filepath.name}")
            self.is_encrypted = True
            self.data = self._decrypt_ds2(raw_data)
        else:
            print(f"  Reading unencrypted Philips file: {self.filepath.name}")
            self.data = raw_data

        return self.data

    def _decrypt_ds2(self, raw_data):
        """
        Decrypt DreamStation 2 file
        Based on OSCAR's PRDS2File::decrypt() implementation
        """
        if len(raw_data) < DS2_HEADER_SIZE:
            raise ValueError("File too small to be valid DS2 format")

        # Parse header
        magic = raw_data[0:3]
        if magic != b'\x0d\x01\x01':
            raise ValueError("Invalid DS2 magic bytes")

        # Extract crypto parameters from header
        guid = raw_data[6:42]
        iv = raw_data[42:54]  # 12 bytes for GCM
        salt = raw_data[54:70]  # 16 bytes for PBKDF2
        flags = raw_data[70:72]

        # Encrypted keys
        import_key_encrypted = raw_data[72:104]  # 32 bytes
        import_key_tag = raw_data[104:120]  # 16 bytes
        export_key_encrypted = raw_data[120:152]  # 32 bytes
        export_key_tag = raw_data[152:168]  # 16 bytes

        # Payload tag and data
        payload_tag = raw_data[168:184]  # 16 bytes
        encrypted_payload = raw_data[184:]

        print(f"    GUID: {guid[:8].hex()}...")
        print(f"    Salt: {salt.hex()}")
        print(f"    IV: {iv.hex()}")

        # Step 1: Decrypt COMMON_KEY with OSCAR_KEY using AES-256 ECB
        cipher = Cipher(algorithms.AES(OSCAR_KEY), modes.ECB(), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_common = decryptor.update(COMMON_KEY) + decryptor.finalize()

        # Step 2: Derive salted key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=10000,
            backend=default_backend()
        )
        salted_key = kdf.derive(decrypted_common)

        # Step 3: Decrypt export key using AES-256-GCM
        aesgcm = AESGCM(salted_key)
        try:
            export_key = aesgcm.decrypt(iv, export_key_encrypted + export_key_tag, None)
        except Exception as e:
            raise ValueError(f"Failed to decrypt export key: {e}")

        print(f"    Export key decrypted successfully")

        # Step 4: Decrypt payload using AES-256-GCM
        aesgcm_payload = AESGCM(export_key)
        try:
            # Check if payload tag is all zeros (truncated file workaround)
            if payload_tag == b'\x00' * 16:
                print(f"    WARNING: Payload tag is zero, attempting recovery...")
                # Try without tag validation (risky but OSCAR does it)
                decrypted_payload = aesgcm_payload.decrypt(iv, encrypted_payload, None)
            else:
                decrypted_payload = aesgcm_payload.decrypt(iv, encrypted_payload + payload_tag, None)
        except Exception as e:
            raise ValueError(f"Failed to decrypt payload: {e}")

        print(f"    Payload decrypted: {len(decrypted_payload)} bytes")

        return decrypted_payload


class PhilipsWaveformFile(PhilipsDataFile):
    """Parser for Philips waveform files (.005, .006)"""

    def __init__(self, filepath):
        super().__init__(filepath)
        self.flow_rate = None
        self.pressure = None
        self.sample_rate = None

    def parse_waveform(self):
        """
        Extract flow rate and pressure waveforms
        Based on OSCAR's waveform parsing in prs1_loader.cpp
        """
        if self.data is None:
            self.read()

        data = self.data
        pos = 0

        flow_samples = []
        pressure_samples = []

        print(f"  Parsing waveform data ({len(data)} bytes)...")

        # Parse blocks
        while pos < len(data) - 4:
            # Read block header
            if pos + 3 > len(data):
                break

            code = data[pos]
            size = struct.unpack('<H', data[pos+1:pos+3])[0]
            pos += 3

            # Check if we have enough data
            if pos + size > len(data):
                print(f"    Warning: Block size {size} exceeds remaining data at pos {pos}")
                break

            block_data = data[pos:pos+size]
            pos += size

            # Look for waveform data blocks
            # Code format varies, but waveform data typically has interleaved format
            if size > 4:
                # Try to parse as interleaved waveform
                # First 2 bytes often indicate interleave count
                interleave = struct.unpack('<H', block_data[0:2])[0]

                if 1 <= interleave <= 16 and len(block_data) > interleave * 2:
                    # This looks like waveform data
                    # Skip header bytes
                    waveform_data = block_data[2:]

                    # De-interleave channels
                    # Channel 0: Flow rate
                    # Channel 1: Pressure
                    num_samples = len(waveform_data) // (interleave * 2)

                    for i in range(num_samples):
                        base_idx = i * interleave * 2
                        # Extract flow (first 'interleave' bytes)
                        flow_chunk = waveform_data[base_idx:base_idx+interleave]
                        # Extract pressure (next 'interleave' bytes)
                        pressure_chunk = waveform_data[base_idx+interleave:base_idx+interleave*2]

                        flow_samples.extend(flow_chunk)
                        pressure_samples.extend(pressure_chunk)

        if len(flow_samples) == 0:
            # Fallback: treat entire file as raw samples
            print(f"    No structured waveform found, treating as raw data...")
            # Assume simple interleaved format: alternating flow/pressure bytes
            for i in range(0, len(data) - 1, 2):
                flow_samples.append(data[i])
                pressure_samples.append(data[i+1])

        # Convert to numpy arrays
        # Flow: gain 1.0, raw byte value
        # Pressure: gain 0.1, so divide by 10 to get cmH2O
        self.flow_rate = np.array(flow_samples, dtype=np.float32)
        self.pressure = np.array(pressure_samples, dtype=np.float32) * 0.1

        # Estimate sample rate (Philips typically uses 1 Hz for waveforms)
        # This is a rough estimate - actual rate should come from session data
        self.sample_rate = 1.0  # Default assumption

        print(f"    Extracted {len(self.flow_rate)} flow samples")
        print(f"    Flow range: {self.flow_rate.min():.2f} to {self.flow_rate.max():.2f}")

        return self.flow_rate, self.pressure, self.sample_rate


def find_philips_data(folder_path):
    """
    Find Philips data files in a folder
    Returns paths to waveform files (.005, .006)
    """
    folder = Path(folder_path)

    # Look for .005 or .006 files (waveform data)
    waveform_files = list(folder.glob('*.005')) + list(folder.glob('*.006'))

    # Could also look for .001 (summary), .002 (events), etc.
    # But for OscilloBreath we only need flow rate from waveforms

    return waveform_files


def load_philips_flow_data(folder_path):
    """
    Load flow rate data from a Philips data folder
    Returns: (flow_array, sample_rate)
    """
    print(f"Loading Philips data from: {folder_path}")

    waveform_files = find_philips_data(folder_path)

    if not waveform_files:
        raise ValueError("No Philips waveform files (.005, .006) found in folder")

    print(f"Found {len(waveform_files)} waveform file(s)")

    # Load and concatenate all waveform files
    all_flow_data = []
    sample_rate = None

    for wf_file in sorted(waveform_files):
        try:
            wf = PhilipsWaveformFile(wf_file)
            flow, pressure, sr = wf.parse_waveform()

            if sample_rate is None:
                sample_rate = sr

            all_flow_data.append(flow)

        except Exception as e:
            print(f"  Error loading {wf_file.name}: {e}")
            continue

    if len(all_flow_data) == 0:
        raise ValueError("Could not load any waveform data")

    # Concatenate all sessions
    combined_flow = np.concatenate(all_flow_data)

    print(f"\nTotal flow data: {len(combined_flow)} samples at {sample_rate} Hz")
    print(f"Duration: {len(combined_flow)/sample_rate/3600:.2f} hours")

    return combined_flow, sample_rate


# Example usage for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python philips_loader.py <path_to_philips_data_folder>")
        sys.exit(1)

    folder = sys.argv[1]

    try:
        flow, sr = load_philips_flow_data(folder)
        print(f"\nSuccess! Loaded {len(flow)} flow samples")
        print(f"Sample rate: {sr} Hz")
        print(f"Flow range: {flow.min():.2f} to {flow.max():.2f}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
