"""
Self-Supervised Pretraining Dataset Loader for TUEV.

Provides:
- Unlabeled/masked sliding EEG windows (e.g., 4 seconds at 250 Hz = 1000 samples)
- Notch (60 Hz) and Bandpass (0.5 - 45 Hz) filtering
- Per-channel Z-score normalization
- Online / cached PSD band-power extraction
- Variant A, B, C ranking target generation
- Augmentation pipeline integration
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Union

from ..spectral.psd_extractor import PSDBandPowerExtractor
from ..spectral.rank_target_builder import RankTargetBuilder
from .augmentations import EEGRankAugmenter


class TUEVPretrainDataset(Dataset):
    """
    Dataset for Self-Supervised Ordinal Ranking Pretraining on TUEV.
    Returns:
    - `x_window`: Multi-channel EEG window `(n_channels, in_samples)`
    - `targets_a`: Variant A targets (cross-channel ranking for 5 bands)
    - `targets_c`: Variant C targets (cross-band ranking for 22 channels)
    - `seq_windows`: Multi-window temporal sequence `(seq_len, n_channels, in_samples)` for Variant B
    - `targets_b`: Variant B targets (cross-time ranking for temporal sequence)
    - `log_psd`: Shape `(n_channels, num_bands)` for baseline regression
    """

    def __init__(
        self,
        data_root: str = "TU-v2.0.1",
        split: str = "train",
        window_duration_sec: float = 4.0,
        stride_sec: float = 2.0,
        fs: float = 250.0,
        variant_b_seq_len: int = 4,
        tie_margin: float = 1e-3,
        use_augmentation: bool = True,
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
        self.variant_b_seq_len = variant_b_seq_len
        self.num_channels = 22

        self.psd_extractor = PSDBandPowerExtractor(fs=fs)
        self.target_builder = RankTargetBuilder(tie_margin=tie_margin)
        self.augmenter = EEGRankAugmenter() if use_augmentation else None

        self._windows_mmap = None
        self._bps_mmap = None
        self.n_samples = 0

        if synthetic_samples is not None and synthetic_samples > 0:
            self._generate_synthetic_data(synthetic_samples)
        else:
            self._load_tuev_data(max_records)

    def _generate_synthetic_data(self, n_samples: int):
        """
        Generates realistic synthetic multi-channel EEG signals with known rhythmic components.
        """
        t = np.linspace(0, self.window_duration_sec, self.window_samples, endpoint=False)
        for i in range(n_samples):
            window = np.zeros((self.num_channels, self.window_samples), dtype=np.float32)
            for ch in range(self.num_channels):
                # Add background 1/f pink noise approximation
                noise = np.random.randn(self.window_samples) * 0.5
                
                # Modulate rhythm by channel region (e.g. occipital channels ch 3, 7 have strong alpha 10Hz)
                alpha_amp = 2.5 if ch in [3, 7, 17, 21] else 0.5
                delta_amp = 1.8 if ch in [0, 4, 14, 18] else 0.4
                beta_amp = 1.0 if ch in [10, 11, 15, 19] else 0.3

                sig = (
                    alpha_amp * np.sin(2 * np.pi * 10.0 * t + np.random.rand() * 2 * np.pi) +
                    delta_amp * np.sin(2 * np.pi * 2.0 * t + np.random.rand() * 2 * np.pi) +
                    beta_amp * np.sin(2 * np.pi * 20.0 * t + np.random.rand() * 2 * np.pi) +
                    noise
                )
                # Z-score normalize
                sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
                window[ch, :] = sig

            bp = self.psd_extractor.extract(window)
            self.windows.append(window)
            self.band_powers.append(bp)

    def __getstate__(self):
        """Exclude memmap handles from pickle so workers can lazily reopen them."""
        state = self.__dict__.copy()
        state['_windows_mmap'] = None
        state['_bps_mmap'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_loaded(self):
        if self._windows_mmap is None:
            self._windows_mmap = np.load(self.mmap_win_path, mmap_mode='r')
            self._bps_mmap = np.load(self.mmap_bp_path, mmap_mode='r')

    def _load_tuev_data(self, max_records: Optional[int]):
        """
        Scans EDF files in TUEV corpus, preprocesses, extracts sliding windows, and saves them to a disk memmap cache.
        """
        cache_dir = os.path.join(self.data_root, "cache", self.split)
        os.makedirs(cache_dir, exist_ok=True)
        self.mmap_win_path = os.path.join(cache_dir, "windows.npy")
        self.mmap_bp_path = os.path.join(cache_dir, "bps.npy")

        if os.path.exists(self.mmap_win_path) and os.path.exists(self.mmap_bp_path):
            tmp = np.load(self.mmap_win_path, mmap_mode='r')
            self.n_samples = tmp.shape[0]
            del tmp
            print(f"Loaded memmap cache from {cache_dir} with {self.n_samples} samples.", flush=True)
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
        from dataset_loader import read_edf_file, build_tcp_montage
        from preprocessing import apply_bandpass_filter, apply_notch_filter, normalize_signals

        local_windows = []
        local_bps = []

        print(f"Found {len(edf_files)} EDF files. Building memmap cache at {cache_dir}...", flush=True)
        for idx, edf_path in enumerate(edf_files):
            if (idx + 1) % 20 == 0 or idx == 0 or idx == len(edf_files) - 1:
                print(f"[{idx+1}/{len(edf_files)}] Loading: {os.path.basename(edf_path)}", flush=True)
            try:
                sig, fs, ch_names = read_edf_file(edf_path)
                montage_sig, _ = build_tcp_montage(sig, ch_names)
                
                filtered = apply_bandpass_filter(montage_sig, fs, lowcut=0.5, highcut=45.0)
                notched = apply_notch_filter(filtered, fs, freq=60.0)
                normed = normalize_signals(notched, method='zscore')
                
                # EEG-specific preprocessing: clip extreme artifacts at ±6σ
                normed = np.clip(normed, -6.0, 6.0).astype(np.float32)

                if abs(fs - self.fs) > 1.0:
                    from scipy.signal import resample
                    target_len = int(normed.shape[1] * self.fs / fs)
                    normed = resample(normed, target_len, axis=-1).astype(np.float32)

                n_ch, n_samples = normed.shape
                for start_idx in range(0, n_samples - self.window_samples + 1, self.stride_samples):
                    win = normed[:, start_idx : start_idx + self.window_samples]
                    
                    # Per-window re-normalization (removes slow drift within window)
                    win_mean = win.mean(axis=-1, keepdims=True)
                    win_std = win.std(axis=-1, keepdims=True)
                    win_std[win_std < 1e-8] = 1.0
                    win = ((win - win_mean) / win_std).astype(np.float32)
                    
                    bp = self.psd_extractor.extract(win)
                    local_windows.append(win)
                    local_bps.append(bp)
            except Exception:
                continue

        if len(local_windows) == 0:
            raise RuntimeError(f"Failed to load any valid EEG windows.")

        print("Saving lists to memmap cache...", flush=True)
        win_arr = np.stack(local_windows).astype(np.float32)
        bp_arr = np.stack(local_bps).astype(np.float32)
        self.n_samples = win_arr.shape[0]
        np.save(self.mmap_win_path, win_arr)
        np.save(self.mmap_bp_path, bp_arr)
        del local_windows, local_bps, win_arr, bp_arr
        print(f"Successfully cached {self.n_samples} samples.", flush=True)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        self._ensure_loaded()
        window = self._windows_mmap[idx].copy()
        
        # Apply augmentation if configured
        if self.augmenter is not None:
            window = self.augmenter(window)
            bp = self.psd_extractor.extract(window)
        else:
            bp = self._bps_mmap[idx].copy()

        # Targets for Variant A (Spatial cross-channel)
        targ_a_np = self.target_builder.build_variant_a_targets(bp)
        # Targets for Variant C (Spectral profile cross-band)
        targ_c_np = self.target_builder.build_variant_c_targets(bp)

        # Variant B sequence: fetch consecutive windows around idx
        seq_len = self.variant_b_seq_len
        seq_wins = []
        seq_bps = []
        for offset in range(seq_len):
            s_idx = (idx + offset) % self.n_samples
            s_win = self._windows_mmap[s_idx].copy()
            seq_wins.append(s_win)
            seq_bps.append(self._bps_mmap[s_idx].copy())

        seq_wins_arr = np.stack(seq_wins, axis=0)  # (seq_len, C, in_samples)
        seq_bps_arr = np.stack(seq_bps, axis=0)    # (seq_len, C, num_bands)
        targ_b_np = self.target_builder.build_variant_b_targets(seq_bps_arr)

        # Convert to torch tensors
        def to_torch_dict(d: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
            return {
                k: (torch.from_numpy(v[0]) if v.shape[0] == 1 else torch.from_numpy(v))
                for k, v in d.items()
            }

        return {
            'window': torch.from_numpy(window).float(),  # (C, in_samples)
            'targets_a': to_torch_dict(targ_a_np),
            'targets_c': to_torch_dict(targ_c_np),
            'seq_windows': torch.from_numpy(seq_wins_arr).float(),  # (seq_len, C, in_samples)
            'targets_b': to_torch_dict(targ_b_np),
            'log_psd': torch.from_numpy(np.log(np.maximum(bp, 1e-12))).float(),  # (C, num_bands)
        }
