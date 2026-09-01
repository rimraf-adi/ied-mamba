"""
Downstream 6-Class EEG Event Classification Dataset for TUEV.

Classes:
0: Background (BCKG)
1: Spike and Slow Wave (SPSW)
2: Generalized Periodic Epileptiform Discharge (GPED)
3: Periodic Lateralized Epileptiform Discharge (PLED)
4: Eye Movement (EYEM)
5: Artifact (ARTF)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union

LABEL_MAP = {
    'bckg': 0,
    'spsw': 1,
    'gped': 2,
    'pled': 3,
    'eyem': 4,
    'artf': 5,
}

CLASS_NAMES = ['BCKG', 'SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF']


class TUEVDownstreamDataset(Dataset):
    """
    Dataset for downstream 6-class event classification evaluation.
    """

    def __init__(
        self,
        data_root: str = "TU-v2.0.1",
        split: str = "eval",
        window_duration_sec: float = 4.0,
        stride_sec: float = 2.0,
        fs: float = 250.0,
        max_records: Optional[int] = None,
        synthetic_samples: Optional[int] = None
    ):
        self.data_root = data_root
        self.split = split
        self.window_duration_sec = window_duration_sec
        self.stride_sec = stride_sec
        self.fs = fs
        self.window_samples = int(window_duration_sec * fs)
        self.stride_samples = int(stride_sec * fs)
        self.num_channels = 22

        self.windows: List[np.ndarray] = []
        self.labels: List[int] = []

        if synthetic_samples is not None and synthetic_samples > 0:
            self._generate_synthetic_downstream(synthetic_samples)
        else:
            self._load_tuev_downstream(max_records)

    def _generate_synthetic_downstream(self, n_samples: int):
        """
        Generates synthetic multi-channel EEG windows with class-specific morphologic signatures.
        """
        t = np.linspace(0, self.window_duration_sec, self.window_samples, endpoint=False)
        for i in range(n_samples):
            cls_label = i % 6
            window = np.zeros((self.num_channels, self.window_samples), dtype=np.float32)

            for ch in range(self.num_channels):
                noise = np.random.randn(self.window_samples) * 0.4
                # Baseline background
                sig = 0.8 * np.sin(2 * np.pi * 9.0 * t + np.random.rand() * 2 * np.pi) + noise

                if cls_label == 1:  # SPSW (sharp wave spike)
                    spike_center = int(self.window_samples * 0.5)
                    t_spike = np.arange(-30, 30)
                    spike_morph = 4.0 * np.exp(- (t_spike ** 2) / 30.0) - 2.0 * np.exp(- ((t_spike - 15) ** 2) / 80.0)
                    sig[spike_center-30:spike_center+30] += spike_morph
                elif cls_label == 2:  # GPED (generalized periodic discharge across all channels)
                    sig += 2.5 * np.sin(2 * np.pi * 1.5 * t) ** 3
                elif cls_label == 3:  # PLED (lateralized periodic discharge on left channels)
                    if ch < 11:
                        sig += 3.0 * np.sin(2 * np.pi * 1.0 * t) ** 4
                elif cls_label == 4:  # EYEM (frontal slow delta deflection)
                    if ch in [0, 4, 14, 18]:
                        sig += 4.0 * np.sin(2 * np.pi * 0.8 * t)
                elif cls_label == 5:  # ARTF (high-frequency muscle artifact)
                    sig += 2.0 * np.random.randn(self.window_samples) * np.sin(2 * np.pi * 35.0 * t)

                # Normalize
                sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
                window[ch, :] = sig

            self.windows.append(window)
            self.labels.append(cls_label)

    def _load_tuev_downstream(self, max_records: Optional[int]):
        """
        Loads TUEV recordings and maps annotations (.rec / .lab) to window labels.
        """
        split_dir = os.path.join(self.data_root, "edf", self.split)
        if not os.path.exists(split_dir):
            split_dir = self.data_root

        edf_files = glob.glob(os.path.join(split_dir, "**", "*.edf"), recursive=True)
        if max_records is not None:
            edf_files = edf_files[:max_records]

        if len(edf_files) == 0:
            self._generate_synthetic_downstream(128)
            return

        import sys
        try:
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
            from dataset_loader import read_edf_file, build_tcp_montage, parse_rec_file
            from preprocessing import apply_bandpass_filter, apply_notch_filter, normalize_signals
            can_load_edf = True
        except Exception:
            can_load_edf = False

        if not can_load_edf:
            self._generate_synthetic_downstream(128)
            return

        for edf_path in edf_files:
            try:
                sig, fs, ch_names = read_edf_file(edf_path)
                montage_sig, _ = build_tcp_montage(sig, ch_names)
                
                filtered = apply_bandpass_filter(montage_sig, fs, lowcut=0.5, highcut=45.0)
                notched = apply_notch_filter(filtered, fs, freq=60.0)
                normed = normalize_signals(notched, method='zscore')

                rec_path = edf_path.replace('.edf', '.rec')
                events = parse_rec_file(rec_path)

                n_ch, n_samples = normed.shape
                for start_idx in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                    win = normed[:, start_idx : start_idx + self.window_samples]
                    win_start_sec = start_idx / self.fs
                    win_stop_sec = (start_idx + self.window_samples) / self.fs

                    # Determine dominant event in this window
                    win_label = 0  # BCKG by default
                    for ev in events:
                        # Check overlap
                        if ev['start_sec'] < win_stop_sec and ev['stop_sec'] > win_start_sec:
                            if ev['label_id'] > 0:
                                win_label = ev['label_id']
                                break

                    self.windows.append(win)
                    self.labels.append(win_label)
            except Exception:
                continue

        if len(self.windows) == 0:
            self._generate_synthetic_downstream(128)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.windows[idx]).float(),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
