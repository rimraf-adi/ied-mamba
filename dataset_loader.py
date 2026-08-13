"""
Dataset Loader for TUH EEG Event Corpus (TU-v2.0.1) & IED Analysis.

This module provides data structures and parsing functions for:
- European Data Format (.edf) raw EEG signal files.
- Recording-level annotations (.rec) with second-precision timestamps.
- Channel-level annotations (.lab) with microsecond-precision timestamps.
- HTK differential energy feature files (.htk).
- Standard ACNS TCP Montage construction (22 differential channels).
- PyTorch Dataset & DataLoader integration for training deep learning models.
"""

import os
import re
import struct
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset

# Global Label Mapping for TUH EEG Event Corpus
LABEL_MAP = {
    'bckg': 0,  # Background
    'spsw': 1,  # Spike and Slow Wave
    'gped': 2,  # Generalized Periodic Epileptiform Discharge
    'pled': 3,  # Periodic Lateralized Epileptiform Discharge
    'eyem': 4,  # Eye Movement
    'artf': 5,  # Artifact
}

# Numeric code mapping (used in .rec files)
REC_CODE_MAP = {
    1: 'spsw',
    2: 'gped',
    3: 'pled',
    4: 'eyem',
    5: 'artf',
    6: 'bckg'
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Standard ACNS TCP Montage Channel Pairs (22 Channels)
TCP_MONTAGE_DEFINITIONS = [
    (0, "FP1-F7", "EEG FP1-REF", "EEG F7-REF"),
    (1, "F7-T3", "EEG F7-REF", "EEG T3-REF"),
    (2, "T3-T5", "EEG T3-REF", "EEG T5-REF"),
    (3, "T5-O1", "EEG T5-REF", "EEG O1-REF"),
    (4, "FP2-F8", "EEG FP2-REF", "EEG F8-REF"),
    (5, "F8-T4", "EEG F8-REF", "EEG T4-REF"),
    (6, "T4-T6", "EEG T4-REF", "EEG T6-REF"),
    (7, "T6-O2", "EEG T6-REF", "EEG O2-REF"),
    (8, "A1-T3", "EEG A1-REF", "EEG T3-REF"),
    (9, "T3-C3", "EEG T3-REF", "EEG C3-REF"),
    (10, "C3-CZ", "EEG C3-REF", "EEG CZ-REF"),
    (11, "CZ-C4", "EEG CZ-REF", "EEG C4-REF"),
    (12, "C4-T4", "EEG C4-REF", "EEG T4-REF"),
    (13, "T4-A2", "EEG T4-REF", "EEG A2-REF"),
    (14, "FP1-F3", "EEG FP1-REF", "EEG F3-REF"),
    (15, "F3-C3", "EEG F3-REF", "EEG C3-REF"),
    (16, "C3-P3", "EEG C3-REF", "EEG P3-REF"),
    (17, "P3-O1", "EEG P3-REF", "EEG O1-REF"),
    (18, "FP2-F4", "EEG FP2-REF", "EEG F4-REF"),
    (19, "F4-C4", "EEG F4-REF", "EEG C4-REF"),
    (20, "C4-P4", "EEG C4-REF", "EEG P4-REF"),
    (21, "P4-O2", "EEG P4-REF", "EEG O2-REF"),
]


def parse_rec_file(rec_path: str) -> List[Dict]:
    """
    Parses a .rec annotation file.
    Format per line: channel_idx, start_sec, stop_sec, label_code
    Example: 19,28.6,29.6,6
    """
    events = []
    if not os.path.exists(rec_path):
        return events

    with open(rec_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    ch_idx = int(parts[0])
                    start_sec = float(parts[1])
                    stop_sec = float(parts[2])
                    code = int(parts[3])
                    label_str = REC_CODE_MAP.get(code, 'bckg')
                    label_id = LABEL_MAP.get(label_str, 0)
                    events.append({
                        'channel_idx': ch_idx,
                        'start_sec': start_sec,
                        'stop_sec': stop_sec,
                        'duration_sec': max(0.0, stop_sec - start_sec),
                        'label_code': code,
                        'label_str': label_str,
                        'label_id': label_id,
                        'line_num': line_num
                    })
                except ValueError:
                    continue
    return events


def parse_lab_file(lab_path: str) -> List[Dict]:
    """
    Parses a .lab annotation file for a specific channel.
    Format per line: start_10us stop_10us label_str
    Example: 15760000 15860000 artf (timestamps in 10-microseconds = 1e-5 sec)
    """
    events = []
    if not os.path.exists(lab_path):
        return events

    # Infer channel from filename if format is *_chXXX.lab
    ch_match = re.search(r'_ch(\d+)\.lab$', lab_path)
    ch_idx = int(ch_match.group(1)) if ch_match else -1

    with open(lab_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    start_10us = int(parts[0])
                    stop_10us = int(parts[1])
                    label_str = parts[2].lower()
                    start_sec = start_10us * 1e-5
                    stop_sec = stop_10us * 1e-5
                    label_id = LABEL_MAP.get(label_str, 0)
                    events.append({
                        'channel_idx': ch_idx,
                        'start_sec': start_sec,
                        'stop_sec': stop_sec,
                        'duration_sec': max(0.0, stop_sec - start_sec),
                        'label_str': label_str,
                        'label_id': label_id
                    })
                except ValueError:
                    continue
    return events


def parse_htk_file(htk_path: str) -> Tuple[np.ndarray, int, int]:
    """
    Parses HTK binary feature files containing extracted EEG differential energy features.
    HTK Header format (12 bytes):
      nSamples (int32, big-endian)
      samplePeriod (int32, big-endian, in 100ns units)
      sampleSize (int16, big-endian, bytes per sample)
      parmKind (int16, big-endian)
    Followed by nSamples * (sampleSize // 4) float32 big-endian values.
    Returns:
      (features array shape [nSamples, nFeatures], samplePeriod_ns, parmKind)
    """
    if not os.path.exists(htk_path):
        raise FileNotFoundError(f"HTK file not found: {htk_path}")

    with open(htk_path, 'rb') as f:
        header = f.read(12)
        if len(header) < 12:
            raise ValueError(f"Invalid HTK header in {htk_path}")
        n_samples, sample_period, sample_size, parm_kind = struct.unpack('>iihh', header)
        n_features = sample_size // 4
        raw_data = f.read()
        features = np.frombuffer(raw_data, dtype='>f4').reshape(n_samples, n_features).astype(np.float32)
    return features, sample_period, parm_kind


def read_edf_file(edf_path: str) -> Tuple[np.ndarray, float, List[str]]:
    """
    Reads an EDF file using mne, pyedflib, or custom raw binary parser fallback.
    Returns:
      signals: ndarray of shape (n_channels, n_samples)
      fs: sampling frequency (Hz)
      channel_names: list of channel labels
    """
    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    # Try MNE Python first
    try:
        import mne
        mne.set_log_level('WARNING')
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        signals = raw.get_data()  # shape (n_channels, n_samples) in Volts
        fs = raw.info['sfreq']
        ch_names = raw.ch_names
        return signals, float(fs), ch_names
    except Exception:
        pass

    # Try pyedflib next
    try:
        import pyedflib
        f = pyedflib.EdfReader(edf_path)
        n_channels = f.signals_in_file
        ch_names = f.getSignalLabels()
        fs = f.getSampleFrequency(0)
        n_samples = f.getNSamples()[0]
        signals = np.zeros((n_channels, n_samples), dtype=np.float32)
        for i in range(n_channels):
            signals[i, :] = f.readSignal(i)
        f._close()
        return signals, float(fs), ch_names
    except Exception:
        pass

    # Custom basic EDF header parser fallback
    with open(edf_path, 'rb') as f:
        header = f.read(256)
        n_bytes_header = int(header[184:192].decode('ascii', errors='ignore').strip())
        n_records = int(header[236:244].decode('ascii', errors='ignore').strip())
        duration_record = float(header[244:252].decode('ascii', errors='ignore').strip())
        n_channels = int(header[252:256].decode('ascii', errors='ignore').strip())

        ch_header_len = n_channels * 256
        ch_header = f.read(ch_header_len)
        ch_names = [ch_header[i*16:(i+1)*16].decode('ascii', errors='ignore').strip() for i in range(n_channels)]

        # Read physical min/max and digital min/max for scaling
        offset = n_channels * 16 + n_channels * 80  # skip transducer label
        offset_pmin = offset
        p_mins = [float(ch_header[offset_pmin + i*8: offset_pmin + (i+1)*8].decode('ascii', errors='ignore').strip() or 0) for i in range(n_channels)]
        offset_pmax = offset_pmin + n_channels * 8
        p_maxs = [float(ch_header[offset_pmax + i*8: offset_pmax + (i+1)*8].decode('ascii', errors='ignore').strip() or 1) for i in range(n_channels)]
        offset_dmin = offset_pmax + n_channels * 8
        d_mins = [float(ch_header[offset_dmin + i*8: offset_dmin + (i+1)*8].decode('ascii', errors='ignore').strip() or -32768) for i in range(n_channels)]
        offset_dmax = offset_dmin + n_channels * 8
        d_maxs = [float(ch_header[offset_dmax + i*8: offset_dmax + (i+1)*8].decode('ascii', errors='ignore').strip() or 32767) for i in range(n_channels)]
        offset_nsamples = offset_dmax + n_channels * 8 + n_channels * 80  # skip prefiltering
        n_samples_per_record = [int(ch_header[offset_nsamples + i*8: offset_nsamples + (i+1)*8].decode('ascii', errors='ignore').strip() or 256) for i in range(n_channels)]

        fs = n_samples_per_record[0] / duration_record if duration_record > 0 else 250.0
        total_samples = n_records * n_samples_per_record[0]

        f.seek(n_bytes_header)
        raw_bytes = f.read()
        raw_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

        # Reshape data records
        rec_size = sum(n_samples_per_record)
        signals = np.zeros((n_channels, total_samples), dtype=np.float32)

        for r in range(min(n_records, len(raw_int16) // rec_size)):
            idx_start = r * rec_size
            curr_pos = idx_start
            for ch in range(n_channels):
                ns = n_samples_per_record[ch]
                data_chunk = raw_int16[curr_pos : curr_pos + ns]
                curr_pos += ns
                # Physical value calibration
                gain = (p_maxs[ch] - p_mins[ch]) / max(1.0, (d_maxs[ch] - d_mins[ch]))
                phys_data = (data_chunk - d_mins[ch]) * gain + p_mins[ch]
                dst_start = r * ns
                signals[ch, dst_start:dst_start+len(phys_data)] = phys_data

    return signals, fs, ch_names


def build_tcp_montage(signals: np.ndarray, ch_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Constructs the standard 22-channel ACNS TCP Montage from raw scalp reference channels.
    Returns:
      montage_signals: array shape (22, n_samples)
      montage_names: list of montage channel names (e.g. 'FP1-F7')
    """
    # Clean channel names
    clean_names = [re.sub(r'^(EEG\s*|\s*REF|-REF)', '', ch.strip()).upper() for ch in ch_names]
    ch_map = {name: idx for idx, name in enumerate(clean_names)}

    n_samples = signals.shape[1]
    montage_signals = np.zeros((22, n_samples), dtype=np.float32)
    montage_names = []

    for idx, name, pos_name, neg_name in TCP_MONTAGE_DEFINITIONS:
        pos_clean = re.sub(r'^(EEG\s*|\s*REF|-REF)', '', pos_name.strip()).upper()
        neg_clean = re.sub(r'^(EEG\s*|\s*REF|-REF)', '', neg_name.strip()).upper()

        pos_idx = ch_map.get(pos_clean, None)
        neg_idx = ch_map.get(neg_clean, None)

        if pos_idx is not None and neg_idx is not None:
            montage_signals[idx, :] = signals[pos_idx, :] - signals[neg_idx, :]
        elif pos_idx is not None:
            montage_signals[idx, :] = signals[pos_idx, :]
        elif neg_idx is not None:
            montage_signals[idx, :] = -signals[neg_idx, :]

        montage_names.append(name)

    return montage_signals, montage_names


class EEGEventDataset(Dataset):
    """
    PyTorch Dataset loader for TUH EEG Event Corpus.
    Extracts sliding windows of TCP montage signals and associated event labels.
    """
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 window_sec: float = 2.0,
                 stride_sec: float = 1.0,
                 target_fs: float = 250.0,
                 include_htk: bool = False,
                 max_files: Optional[int] = None):
        """
        Args:
            root_dir: Path to TU-v2.0.1 directory.
            split: 'train' or 'eval'.
            window_sec: Window length in seconds.
            stride_sec: Stride length in seconds between sliding windows.
            target_fs: Target sampling frequency in Hz.
            include_htk: Whether to load HTK features alongside raw signals.
            max_files: Optional limit on number of session files to load.
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.target_fs = target_fs
        self.include_htk = include_htk

        self.samples = []  # Metadata for each window sample
        self._index_dataset(max_files)

    def _index_dataset(self, max_files: Optional[int]):
        target_dir = os.path.join(self.root_dir, 'edf', self.split)
        if not os.path.exists(target_dir):
            print(f"Warning: Directory not found: {target_dir}")
            return

        edf_files = []
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith('.edf'):
                    edf_files.append(os.path.join(root, file))

        if max_files:
            edf_files = edf_files[:max_files]

        for edf_path in edf_files:
            base_path = os.path.splitext(edf_path)[0]
            rec_path = base_path + '.rec'
            events = parse_rec_file(rec_path)

            try:
                signals, fs, ch_names = read_edf_file(edf_path)
                n_samples_file = signals.shape[1]
                duration_sec = n_samples_file / fs
            except Exception as e:
                continue

            window_len = int(self.window_sec * fs)
            stride_len = int(self.stride_sec * fs)

            if n_samples_file < window_len:
                continue

            # Sliding window indices
            for start_idx in range(0, n_samples_file - window_len + 1, stride_len):
                stop_idx = start_idx + window_len
                win_start_sec = start_idx / fs
                win_stop_sec = stop_idx / fs

                # Determine dominant label in window
                win_label_id = 0  # Default bckg
                max_overlap = 0.0

                for ev in events:
                    # Compute overlap between window and event interval
                    overlap_start = max(win_start_sec, ev['start_sec'])
                    overlap_stop = min(win_stop_sec, ev['stop_sec'])
                    overlap_dur = max(0.0, overlap_stop - overlap_start)

                    if overlap_dur > max_overlap and ev['label_id'] != 0:
                        max_overlap = overlap_dur
                        win_label_id = ev['label_id']

                self.samples.append({
                    'edf_path': edf_path,
                    'rec_path': rec_path,
                    'start_idx': start_idx,
                    'stop_idx': stop_idx,
                    'label_id': win_label_id,
                    'label_str': REVERSE_LABEL_MAP.get(win_label_id, 'bckg'),
                    'fs': fs
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        info = self.samples[idx]
        signals, fs, ch_names = read_edf_file(info['edf_path'])
        montage_signals, _ = build_tcp_montage(signals, ch_names)

        # Slice window
        window = montage_signals[:, info['start_idx']:info['stop_idx']]

        # Resample if needed
        if fs != self.target_fs:
            from scipy.signal import resample
            n_target_samples = int(self.window_sec * self.target_fs)
            window = resample(window, n_target_samples, axis=1)

        x_tensor = torch.from_numpy(window).float()
        y_tensor = torch.tensor(info['label_id'], dtype=torch.long)

        meta = {
            'edf_path': info['edf_path'],
            'start_idx': info['start_idx'],
            'stop_idx': info['stop_idx'],
            'label_str': info['label_str']
        }

        return x_tensor, y_tensor, meta
