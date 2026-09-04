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
        self.window_samples = int(window_duration_sec * self.fs)
        self.stride_samples = int(self.window_samples) # Non-overlapping for eval
        
        self._windows_mmap = None
        self._labels_mmap = None
        self.n_samples = 0

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

    def __getstate__(self):
        """Exclude memmap handles from pickle so workers can lazily reopen them."""
        state = self.__dict__.copy()
        state['_windows_mmap'] = None
        state['_labels_mmap'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_loaded(self):
        if self._windows_mmap is None:
            self._windows_mmap = np.load(self.mmap_win_path, mmap_mode='r')
            self._labels_mmap = np.load(self.mmap_label_path, mmap_mode='r')

    def _load_tuev_downstream(self, max_records: Optional[int]):
        """
        Loads TUEV recordings, maps annotations, and caches to disk via memmap.
        """
        cache_dir = os.path.join(self.data_root, "cache_downstream", self.split)
        os.makedirs(cache_dir, exist_ok=True)
        self.mmap_win_path = os.path.join(cache_dir, "windows.npy")
        self.mmap_label_path = os.path.join(cache_dir, "labels.npy")

        if os.path.exists(self.mmap_win_path) and os.path.exists(self.mmap_label_path):
            tmp = np.load(self.mmap_win_path, mmap_mode='r')
            self.n_samples = tmp.shape[0]
            del tmp
            print(f"Loaded downstream memmap cache from {cache_dir} with {self.n_samples} samples.", flush=True)
            return

        split_dir = os.path.join(self.data_root, "edf", self.split)
        if not os.path.exists(split_dir):
            split_dir = self.data_root

        edf_files = glob.glob(os.path.join(split_dir, "**", "*.edf"), recursive=True)
        if max_records is not None:
            edf_files = edf_files[:max_records]

        if len(edf_files) == 0:
            raise RuntimeError(f"No EDF files found in {split_dir}.")

        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")))
        from dataset_loader import read_edf_file, build_tcp_montage, parse_rec_file
        from preprocessing import apply_bandpass_filter, apply_notch_filter, normalize_signals

        local_windows = []
        local_labels = []

        print(f"Found {len(edf_files)} downstream EDF files. Building memmap cache at {cache_dir}...", flush=True)
        for idx, edf_path in enumerate(edf_files):
            if (idx + 1) % 20 == 0 or idx == 0 or idx == len(edf_files) - 1:
                print(f"[{idx+1}/{len(edf_files)}] Downstream Loading: {os.path.basename(edf_path)}", flush=True)
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

                    win_label = 0
                    for ev in events:
                        if ev['start_sec'] < win_stop_sec and ev['stop_sec'] > win_start_sec:
                            if ev['label_id'] > 0:
                                win_label = ev['label_id']
                                break

                    local_windows.append(win)
                    local_labels.append(win_label)
            except Exception:
                continue

        if len(local_windows) == 0:
            raise RuntimeError(f"Failed to load any valid EEG windows.")

        print("Saving lists to downstream memmap cache...", flush=True)
        win_arr = np.stack(local_windows).astype(np.float32)
        lbl_arr = np.array(local_labels, dtype=np.int64)
        self.n_samples = win_arr.shape[0]
        np.save(self.mmap_win_path, win_arr)
        np.save(self.mmap_label_path, lbl_arr)
        del local_windows, local_labels, win_arr, lbl_arr
        print(f"Successfully cached {self.n_samples} downstream samples.", flush=True)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_loaded()
        return (
            torch.from_numpy(self._windows_mmap[idx].copy()).float(),
            torch.tensor(self._labels_mmap[idx].item(), dtype=torch.long)
        )
