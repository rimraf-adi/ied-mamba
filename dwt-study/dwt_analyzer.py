"""
Discrete Wavelet Transform (DWT) Signal Processing Engine for EEG Analysis.

This module provides comprehensive DWT decomposition, sub-band feature extraction,
multi-wavelet family comparison, and sub-band reconstruction for EEG signal processing.

For an EEG signal sampled at fs = 250 Hz (Nyquist = 125 Hz):
- Level 1 (D1): 62.5 - 125.0 Hz  (High Gamma / Electrode Noise)
- Level 2 (D2): 31.25 - 62.5 Hz   (Gamma Band / Muscle Artifact)
- Level 3 (D3): 15.625 - 31.25 Hz (Beta Band / Sharp Spikes)
- Level 4 (D4): 7.8125 - 15.625 Hz (Alpha Band / Spike Component)
- Level 5 (D5): 3.90625 - 7.8125 Hz (Theta Band / Slow Wave Component)
- Level 6 (D6): 1.953125 - 3.90625 Hz (Upper Delta Band / Periodic Discharges)
- Approx. (A6): 0.0 - 1.953125 Hz (Sub-Delta / Ocular Movement, Drift)
"""

import numpy as np
import pywt
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union

# DWT Level mapping for fs = 250 Hz
DWT_SCALE_NAMES = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'A6']

SCALE_FREQ_BANDS = {
    'D1': (62.5, 125.0),
    'D2': (31.25, 62.5),
    'D3': (15.625, 31.25),
    'D4': (7.8125, 15.625),
    'D5': (3.90625, 7.8125),
    'D6': (1.953125, 3.90625),
    'A6': (0.0, 1.953125)
}

DEFAULT_WAVELET = 'db4'
SUPPORTED_WAVELETS = ['db4', 'sym4', 'coif3', 'bior3.9']


def decompose_eeg_dwt(
    signal: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = 6
) -> Dict[str, np.ndarray]:
    """
    Decomposes 1D EEG signal into multi-level DWT coefficients.
    Returns dictionary with keys ['A6', 'D6', 'D5', 'D4', 'D3', 'D2', 'D1'].
    """
    # pywt.wavedec returns [cA_n, cD_n, cD_n-1, ..., cD_1]
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    
    # Map array to named dictionary
    # coeffs[0] is cA6
    # coeffs[1] is cD6, coeffs[2] is cD5, ..., coeffs[6] is cD1
    result = {'A6': coeffs[0]}
    for idx, d_level in enumerate(range(level, 0, -1)):
        key = f"D{d_level}"
        result[key] = coeffs[idx + 1]
        
    return result


def extract_subband_features(coeff: np.ndarray) -> Dict[str, float]:
    """
    Extracts statistical, energetic, and entropic features from a DWT coefficient vector.
    """
    if len(coeff) == 0:
        return {
            'energy': 0.0, 'variance': 0.0, 'std': 0.0, 'kurtosis': 0.0,
            'skewness': 0.0, 'log_energy_entropy': 0.0, 'shannon_entropy': 0.0,
            'mav': 0.0, 'zcr': 0.0, 'max_amp': 0.0
        }
        
    energy = float(np.sum(coeff ** 2))
    variance = float(np.var(coeff))
    std = float(np.std(coeff))
    max_amp = float(np.max(np.abs(coeff)))
    mav = float(np.mean(np.abs(coeff)))
    
    # Higher order moments
    kurt = float(stats.kurtosis(coeff)) if std > 1e-9 else 0.0
    skew = float(stats.skew(coeff)) if std > 1e-9 else 0.0
    
    # Entropies
    eps = 1e-12
    # Log Energy Entropy
    log_energy_entropy = float(np.sum(np.log(coeff ** 2 + eps)))
    
    # Shannon Entropy of relative energy distribution within coefficient
    prob = (coeff ** 2) / (energy + eps)
    shannon_entropy = float(-np.sum(prob * np.log2(prob + eps)))
    
    # Zero Crossing Rate
    zcr = float(np.sum(np.diff(np.sign(coeff) != 0)) / max(1, len(coeff) - 1))
    
    return {
        'energy': energy,
        'variance': variance,
        'std': std,
        'kurtosis': kurt,
        'skewness': skew,
        'log_energy_entropy': log_energy_entropy,
        'shannon_entropy': shannon_entropy,
        'mav': mav,
        'zcr': zcr,
        'max_amp': max_amp
    }


def extract_multiscale_dwt_features(
    signal: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = 6
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Performs multi-scale DWT decomposition and computes per-scale features
    plus relative energy distribution across all scales.
    """
    coeffs = decompose_eeg_dwt(signal, wavelet=wavelet, level=level)
    
    total_energy = sum(np.sum(c ** 2) for c in coeffs.values()) + 1e-12
    
    multiscale_feats = {}
    rel_energies = {}
    
    for scale_name in DWT_SCALE_NAMES:
        c = coeffs[scale_name]
        feats = extract_subband_features(c)
        rel_energy = feats['energy'] / total_energy
        feats['rel_energy'] = float(rel_energy)
        rel_energies[scale_name] = float(rel_energy)
        multiscale_feats[scale_name] = feats
        
    return {
        'total_energy': float(total_energy),
        'scale_features': multiscale_feats,
        'rel_energies': rel_energies
    }


def reconstruct_selective_bands(
    signal: np.ndarray,
    keep_scales: List[str],
    wavelet: str = DEFAULT_WAVELET,
    level: int = 6
) -> np.ndarray:
    """
    Reconstructs 1D EEG signal keeping only specified DWT sub-bands.
    Example: keep_scales=['D3', 'D4', 'D5'] to isolate interictal spikes.
    """
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    
    # coeffs layout: [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
    mod_coeffs = []
    
    # cA6 index is 0
    if 'A6' in keep_scales:
        mod_coeffs.append(coeffs[0])
    else:
        mod_coeffs.append(np.zeros_like(coeffs[0]))
        
    # cD6 (idx 1), cD5 (idx 2), ..., cD1 (idx 6)
    for idx, d_level in enumerate(range(level, 0, -1)):
        key = f"D{d_level}"
        if key in keep_scales:
            mod_coeffs.append(coeffs[idx + 1])
        else:
            mod_coeffs.append(np.zeros_like(coeffs[idx + 1]))
            
    reconstructed = pywt.waverec(mod_coeffs, wavelet=wavelet)
    # Ensure length matches original signal length
    return reconstructed[:len(signal)]
