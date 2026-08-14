"""
Wavelet Phase-Amplitude Coupling (PAC) and Cross-Frequency Modulation Engine for EEG Analysis.

Computes the Tort Modulation Index (MI) and Phase-Amplitude distributions between
slow wavelet sub-bands (D5 Theta 3.9-7.8Hz / D6 Delta 1.95-3.9Hz) and
fast wavelet sub-bands (D2 Gamma 31.25-62.5Hz / D3 Beta 15.6-31.25Hz).
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Dict, Tuple, List, Optional
from dwt_analyzer import reconstruct_selective_bands


def compute_wavelet_pac(
    signal: np.ndarray,
    phase_band: List[str] = ['D5'],    # Theta: 3.9 - 7.8 Hz
    amp_band: List[str] = ['D2'],      # Gamma: 31.25 - 62.5 Hz
    wavelet: str = 'db4',
    level: int = 6,
    num_bins: int = 18
) -> Dict:
    """
    Computes Phase-Amplitude Coupling (PAC) between specified wavelet sub-bands.
    Returns:
    - modulation_index: Tort Modulation Index (MI) between 0 (no coupling) and 1 (perfect coupling)
    - phase_bins: array of phase bin centers in degrees (0 to 360)
    - mean_amplitudes: normalized mean amplitude distribution across phase bins
    - preferred_phase_deg: phase angle with maximum amplitude
    """
    # Step 1: Extract selective bandpass reconstructed signals
    slow_sig = reconstruct_selective_bands(signal, keep_scales=phase_band, wavelet=wavelet, level=level)
    fast_sig = reconstruct_selective_bands(signal, keep_scales=amp_band, wavelet=wavelet, level=level)
    
    # Step 2: Compute analytic signals via Hilbert transform
    analytic_slow = sp_signal.hilbert(slow_sig)
    analytic_fast = sp_signal.hilbert(fast_sig)
    
    phase_slow = np.angle(analytic_slow)  # [-pi, pi]
    amp_fast = np.abs(analytic_fast)      # Amplitude envelope
    
    # Map phase to [0, 2*pi]
    phase_slow = np.mod(phase_slow, 2 * np.pi)
    
    # Step 3: Bin phases into num_bins (e.g. 18 bins of 20 degrees each)
    bin_edges = np.linspace(0, 2 * np.pi, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_centers_deg = np.rad2deg(bin_centers)
    
    mean_amps = np.zeros(num_bins)
    for b in range(num_bins):
        mask = (phase_slow >= bin_edges[b]) & (phase_slow < bin_edges[b + 1])
        if np.any(mask):
            mean_amps[b] = np.mean(amp_fast[mask])
        else:
            mean_amps[b] = 0.0
            
    # Step 4: Normalize to probability distribution
    total_amp = np.sum(mean_amps) + 1e-12
    p = mean_amps / total_amp
    
    # Step 5: Compute Tort Modulation Index (MI) via Kullback-Leibler divergence
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    shannon_h = -np.sum(p_safe * np.log(p_safe))
    h_max = np.log(num_bins)
    kl_divergence = h_max - shannon_h
    modulation_index = float(kl_divergence / h_max)
    
    # Preferred phase
    max_bin = np.argmax(mean_amps)
    pref_phase_deg = float(bin_centers_deg[max_bin])
    
    return {
        'modulation_index': float(modulation_index),
        'preferred_phase_deg': pref_phase_deg,
        'phase_bins_deg': bin_centers_deg.tolist(),
        'mean_amplitudes': mean_amps.tolist(),
        'normalized_distribution': p.tolist(),
        'phase_band_name': "+".join(phase_band),
        'amp_band_name': "+".join(amp_band)
    }


def compute_pac_comodulogram(
    signal: np.ndarray,
    wavelet: str = 'db4',
    level: int = 6
) -> Dict:
    """
    Computes complete PAC matrix across multiple phase/amplitude wavelet band pairs:
    Phase bands: D6 (Delta), D5 (Theta), D4 (Alpha)
    Amplitude bands: D3 (Beta), D2 (Gamma), D1 (High Gamma)
    """
    phase_pairs = {'Delta (D6)': ['D6'], 'Theta (D5)': ['D5'], 'Alpha (D4)': ['D4']}
    amp_pairs = {'Beta (D3)': ['D3'], 'Gamma (D2)': ['D2'], 'High-Gamma (D1)': ['D1']}
    
    matrix = {}
    for p_name, p_scales in phase_pairs.items():
        matrix[p_name] = {}
        for a_name, a_scales in amp_pairs.items():
            res = compute_wavelet_pac(signal, phase_band=p_scales, amp_band=a_scales, wavelet=wavelet, level=level)
            matrix[p_name][a_name] = res['modulation_index']
            
    return matrix
