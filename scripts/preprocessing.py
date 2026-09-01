"""
Preprocessing Pipeline for TUH EEG Event Corpus (TU-v2.0.1).

Provides high-performance signal processing functions:
- Bandpass filtering (Butterworth 0.5-50.0 Hz).
- Powerline notch filtering (60.0 Hz).
- Baseline correction and Z-score / Robust normalization.
- Sliding window segmentation and label rasterization.
- Spectral and time-domain feature extraction (Band Powers, RMS, Variance, Hjorth parameters).
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, List, Optional, Union
from dataset_loader import build_tcp_montage, TCP_MONTAGE_DEFINITIONS, LABEL_MAP


def apply_bandpass_filter(signals: np.ndarray,
                          fs: float,
                          lowcut: float = 0.5,
                          highcut: float = 50.0,
                          order: int = 4) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter across all channels.
    Signals shape: (n_channels, n_samples)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Ensure filter frequencies are within valid (0, 1) range
    low = max(0.001, low)
    high = min(0.999, high)

    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    filtered = signal.sosfiltfilt(sos, signals, axis=-1)
    return filtered.astype(np.float32)


def apply_notch_filter(signals: np.ndarray,
                        fs: float,
                        freq: float = 60.0,
                        quality_factor: float = 30.0) -> np.ndarray:
    """
    Applies a zero-phase IIR notch filter to remove powerline noise (60 Hz in US).
    Signals shape: (n_channels, n_samples)
    """
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    if w0 >= 1.0 or w0 <= 0.0:
        return signals

    b, a = signal.iirnotch(w0, quality_factor)
    filtered = signal.filtfilt(b, a, signals, axis=-1)
    return filtered.astype(np.float32)


def normalize_signals(signals: np.ndarray,
                      method: str = 'zscore',
                      eps: float = 1e-8) -> np.ndarray:
    """
    Normalizes multi-channel EEG signals per channel.
    Methods:
      - 'zscore': (x - mean) / std
      - 'robust': (x - median) / IQR
      - 'minmax': (x - min) / (max - min)
    """
    if method == 'zscore':
        mean = np.mean(signals, axis=-1, keepdims=True)
        std = np.std(signals, axis=-1, keepdims=True)
        std[std < eps] = 1.0
        return ((signals - mean) / std).astype(np.float32)

    elif method == 'robust':
        median = np.median(signals, axis=-1, keepdims=True)
        q75, q25 = np.percentile(signals, [75, 25], axis=-1, keepdims=True)
        iqr = q75 - q25
        iqr[iqr < eps] = 1.0
        return ((signals - median) / iqr).astype(np.float32)

    elif method == 'minmax':
        s_min = np.min(signals, axis=-1, keepdims=True)
        s_max = np.max(signals, axis=-1, keepdims=True)
        rng = s_max - s_min
        rng[rng < eps] = 1.0
        return ((signals - s_min) / rng).astype(np.float32)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_eeg(signals: np.ndarray,
                   fs: float,
                   ch_names: List[str],
                   lowcut: float = 0.5,
                   highcut: float = 50.0,
                   notch_freq: float = 60.0,
                   norm_method: str = 'zscore') -> Tuple[np.ndarray, List[str]]:
    """
    Full EEG preprocessing pipeline:
    1. Construct 22-channel ACNS TCP Montage.
    2. Bandpass filtering (0.5 - 50.0 Hz).
    3. Notch filtering (60.0 Hz).
    4. Per-channel signal normalization.

    Returns:
      preprocessed_signals: (22, n_samples)
      montage_names: list of 22 channel names
    """
    # Step 1: TCP Montage Construction
    montage_signals, montage_names = build_tcp_montage(signals, ch_names)

    # Step 2: Bandpass Filter
    filt_signals = apply_bandpass_filter(montage_signals, fs=fs, lowcut=lowcut, highcut=highcut)

    # Step 3: Notch Filter
    filt_signals = apply_notch_filter(filt_signals, fs=fs, freq=notch_freq)

    # Step 4: Normalization
    norm_signals = normalize_signals(filt_signals, method=norm_method)

    return norm_signals, montage_names


def extract_sliding_windows(signals: np.ndarray,
                            fs: float,
                            events: List[Dict],
                            window_sec: float = 2.0,
                            overlap_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Segments preprocessed continuous EEG signals into overlapping fixed-duration windows.

    Args:
      signals: Shape (n_channels, n_samples)
      fs: Sampling rate (Hz)
      events: List of parsed event dictionaries from .rec or .lab file
      window_sec: Window duration in seconds
      overlap_ratio: Overlap fraction (0.0 to 0.9)

    Returns:
      windows: ndarray of shape (n_windows, n_channels, window_samples)
      labels: ndarray of shape (n_windows,) containing integer label IDs
      metadata: List of window metadata dictionaries
    """
    n_channels, n_samples = signals.shape
    window_samples = int(window_sec * fs)
    stride_samples = int(window_samples * (1.0 - overlap_ratio))
    stride_samples = max(1, stride_samples)

    windows_list = []
    labels_list = []
    metadata_list = []

    for start_idx in range(0, n_samples - window_samples + 1, stride_samples):
        stop_idx = start_idx + window_samples
        win_start_sec = start_idx / fs
        win_stop_sec = stop_idx / fs

        # Slice window matrix
        win_data = signals[:, start_idx:stop_idx]

        # Determine dominant event in this window
        win_label_id = 0  # Background default
        max_overlap = 0.0

        for ev in events:
            overlap_start = max(win_start_sec, ev['start_sec'])
            overlap_stop = min(win_stop_sec, ev['stop_sec'])
            overlap_dur = max(0.0, overlap_stop - overlap_start)

            # Prioritize non-background event classes
            if overlap_dur > max_overlap and ev['label_id'] != 0:
                max_overlap = overlap_dur
                win_label_id = ev['label_id']

        windows_list.append(win_data)
        labels_list.append(win_label_id)
        metadata_list.append({
            'start_idx': start_idx,
            'stop_idx': stop_idx,
            'start_sec': win_start_sec,
            'stop_sec': win_stop_sec,
            'label_id': win_label_id
        })

    if len(windows_list) == 0:
        return np.empty((0, n_channels, window_samples)), np.empty((0,), dtype=np.int64), []

    windows_arr = np.stack(windows_list, axis=0).astype(np.float32)
    labels_arr = np.array(labels_list, dtype=np.int64)

    return windows_arr, labels_arr, metadata_list


def compute_band_powers(signals: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
    """
    Computes absolute band powers for standard EEG frequency bands:
    - Delta: 0.5 - 4.0 Hz
    - Theta: 4.0 - 8.0 Hz
    - Alpha: 8.0 - 12.0 Hz
    - Beta: 12.0 - 30.0 Hz
    - Gamma: 30.0 - 50.0 Hz

    Args:
      signals: Shape (n_channels, n_samples)
    Returns:
      Dict mapping band name to per-channel power array shape (n_channels,)
    """
    freqs, psd = signal.welch(signals, fs=fs, nperseg=min(signals.shape[-1], int(2 * fs)), axis=-1)

    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta': (12.0, 30.0),
        'gamma': (30.0, 50.0)
    }

    band_powers = {}
    for band_name, (f_min, f_max) in bands.items():
        idx_band = np.logical_and(freqs >= f_min, freqs <= f_max)
        # Trapezoidal integration across frequency range (compatible with NumPy 2.0+)
        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        power = trapz_fn(psd[..., idx_band], freqs[idx_band], axis=-1)
        band_powers[band_name] = power.astype(np.float32)

    return band_powers


def extract_time_domain_features(signals: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extracts time-domain statistical features per channel:
    - Mean, Standard Deviation, Variance, RMS, Peak-to-Peak, Skewness, Kurtosis, Hjorth Activity & Mobility.
    Signals shape: (n_channels, n_samples)
    """
    mean = np.mean(signals, axis=-1)
    std = np.std(signals, axis=-1)
    var = np.var(signals, axis=-1)
    rms = np.sqrt(np.mean(signals**2, axis=-1))
    ptp = np.ptp(signals, axis=-1)

    # Hjorth parameters
    diff1 = np.diff(signals, axis=-1)
    diff2 = np.diff(diff1, axis=-1)
    var0 = var
    var1 = np.var(diff1, axis=-1)
    var2 = np.var(diff2, axis=-1)

    activity = var0
    mobility = np.sqrt(var1 / np.maximum(var0, 1e-8))
    complexity = np.sqrt(var2 / np.maximum(var1, 1e-8)) / np.maximum(mobility, 1e-8)

    return {
        'mean': mean,
        'std': std,
        'variance': var,
        'rms': rms,
        'peak_to_peak': ptp,
        'hjorth_activity': activity,
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity
    }
