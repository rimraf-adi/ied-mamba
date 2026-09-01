"""
Power Spectral Density (PSD) and Canonical Band-Power Extraction via Welch's Method.

Provides functions to compute PSD and integrate over canonical EEG frequency bands:
- Delta (0.5 - 4.0 Hz)
- Theta (4.0 - 8.0 Hz)
- Alpha (8.0 - 13.0 Hz)
- Beta (13.0 - 30.0 Hz)
- Gamma (30.0 - 45.0 Hz)
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple, List, Optional, Union
import torch

try:
    # Use scipy trapezoid or numpy trapezoid / trapz
    from scipy.integrate import trapezoid as trapz_fn
except ImportError:
    from numpy import trapz as trapz_fn

DEFAULT_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0),
}


def compute_welch_psd(
    signals: np.ndarray,
    fs: float = 250.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    scaling: str = 'density'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Welch Power Spectral Density per channel.
    
    Args:
        signals: Multi-channel EEG signals of shape `(..., n_samples)` or `(..., n_channels, n_samples)`.
        fs: Sampling rate in Hz.
        nperseg: Length of each segment in samples (default: int(1.0 * fs)).
        noverlap: Number of points of overlap between segments (default: 50% of nperseg).
        window: Windowing function type.
        scaling: 'density' or 'spectrum'.
        
    Returns:
        freqs: Frequency array of shape `(n_freqs,)`.
        psd: Power spectral density array of shape `(..., n_freqs)`.
    """
    if nperseg is None:
        nperseg = int(fs * 1.0)
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = signal.welch(
        signals,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
        scaling=scaling
    )
    return freqs.astype(np.float32), psd.astype(np.float32)


def extract_canonical_band_powers(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Integrates PSD across canonical frequency bands using trapezoidal integration.
    
    Args:
        freqs: Array of frequency bin centers of shape `(n_freqs,)`.
        psd: Array of PSD values of shape `(..., n_freqs)`.
        bands: Dictionary mapping band name to `(low_hz, high_hz)`.
        
    Returns:
        band_powers: Array of shape `(..., num_bands)` containing integrated power per band.
    """
    if bands is None:
        bands = DEFAULT_BANDS

    band_powers = []
    for band_name, (low_f, high_f) in bands.items():
        # Mask frequency indices within [low_f, high_f]
        idx_mask = (freqs >= low_f) & (freqs <= high_f)
        if not np.any(idx_mask):
            # Fallback for empty band selection: take nearest bin
            nearest_idx = np.argmin(np.abs(freqs - (low_f + high_f) / 2.0))
            power = psd[..., nearest_idx] * (high_f - low_f)
        else:
            freq_sub = freqs[idx_mask]
            psd_sub = psd[..., idx_mask]
            # Trapezoidal integration across the frequency dimension (axis=-1)
            power = trapz_fn(psd_sub, x=freq_sub, axis=-1)
        
        # Ensure non-negative power
        power = np.maximum(power, 1e-12)
        band_powers.append(power)

    # Stack along last axis: (..., num_bands)
    return np.stack(band_powers, axis=-1).astype(np.float32)


class PSDBandPowerExtractor:
    """
    Stateful PSD and Band-Power feature extractor for single windows or batches.
    """

    def __init__(
        self,
        fs: float = 250.0,
        welch_nperseg_sec: float = 1.0,
        welch_overlap_ratio: float = 0.5,
        bands: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.fs = fs
        self.nperseg = int(fs * welch_nperseg_sec)
        self.noverlap = int(self.nperseg * welch_overlap_ratio)
        self.bands = bands or DEFAULT_BANDS
        self.band_names = list(self.bands.keys())
        self.num_bands = len(self.band_names)

    def extract(self, signals: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Computes band powers for input signals.
        Input shape: `(n_channels, n_samples)` or `(batch_size, n_channels, n_samples)`.
        Output shape: `(n_channels, num_bands)` or `(batch_size, n_channels, num_bands)`.
        """
        is_torch = isinstance(signals, torch.Tensor)
        if is_torch:
            arr = signals.detach().cpu().numpy()
        else:
            arr = np.asarray(signals)

        freqs, psd = compute_welch_psd(
            arr,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap
        )
        band_powers = extract_canonical_band_powers(freqs, psd, self.bands)
        return band_powers

    def extract_log_psd(self, signals: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extracts log(band_powers) directly (used for log-PSD regression baseline).
        """
        bp = self.extract(signals)
        return np.log(np.maximum(bp, 1e-12))
