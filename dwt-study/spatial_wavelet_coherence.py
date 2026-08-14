"""
Spatial Wavelet Coherence & Phase-Locking Value (W-PLV) Engine for 22-Channel ACNS TCP Montage.

Computes 22x22 spatial phase synchronization connectivity matrices, Global Synchrony Indices (GSI),
and Inter/Intra-Hemispheric coupling metrics across clinical EEG annotations.
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Dict, List, Tuple
from dwt_analyzer import reconstruct_selective_bands

# 22 TCP Montage Channel labels
TCP_CHANNELS = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "A1-T3", "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "T4-A2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

LEFT_CHANNELS = [0, 1, 2, 3, 8, 9, 14, 15, 16, 17]   # Left temporal & parasagittal
RIGHT_CHANNELS = [4, 5, 6, 7, 12, 13, 18, 19, 20, 21] # Right temporal & parasagittal
MIDLINE_CHANNELS = [10, 11]                            # C3-CZ, CZ-C4


def compute_wavelet_plv_matrix(
    multichannel_signals: np.ndarray,
    target_scales: List[str] = ['D3', 'D4', 'D5'],  # IED frequency range (3.9 - 31.25 Hz)
    wavelet: str = 'db4',
    level: int = 6
) -> Dict:
    """
    Computes 22x22 Wavelet Phase-Locking Value (W-PLV) connectivity matrix.
    multichannel_signals shape: (22, n_samples)
    """
    n_ch, n_samples = multichannel_signals.shape
    assert n_ch == 22, f"Expected 22 channels, got {n_ch}"
    
    # Step 1: Bandpass reconstruct each channel using selected wavelet sub-bands
    filtered_signals = np.zeros((n_ch, n_samples))
    phases = np.zeros((n_ch, n_samples))
    
    for ch in range(n_ch):
        rec_sig = reconstruct_selective_bands(
            multichannel_signals[ch],
            keep_scales=target_scales,
            wavelet=wavelet,
            level=level
        )
        filtered_signals[ch] = rec_sig
        analytic = sp_signal.hilbert(rec_sig)
        phases[ch] = np.angle(analytic)
        
    # Step 2: Compute pairwise W-PLV matrix
    plv_matrix = np.zeros((n_ch, n_ch))
    
    for i in range(n_ch):
        for j in range(i, n_ch):
            if i == j:
                plv_matrix[i, j] = 1.0
            else:
                # W-PLV = |1/N * sum(exp(i * (phi_i - phi_j)))|
                phase_diff = phases[i] - phases[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_matrix[i, j] = float(plv)
                plv_matrix[j, i] = float(plv)
                
    # Step 3: Compute Regional Connectivity & Asymmetry Metrics
    # Global Synchrony Index (mean off-diagonal PLV)
    mask = ~np.eye(n_ch, dtype=bool)
    gsi = float(np.mean(plv_matrix[mask]))
    
    # Left hemisphere intra-connectivity
    left_sub = plv_matrix[np.ix_(LEFT_CHANNELS, LEFT_CHANNELS)]
    left_mask = ~np.eye(len(LEFT_CHANNELS), dtype=bool)
    left_intra_plv = float(np.mean(left_sub[left_mask]))
    
    # Right hemisphere intra-connectivity
    right_sub = plv_matrix[np.ix_(RIGHT_CHANNELS, RIGHT_CHANNELS)]
    right_mask = ~np.eye(len(RIGHT_CHANNELS), dtype=bool)
    right_intra_plv = float(np.mean(right_sub[right_mask]))
    
    # Inter-hemispheric cross-connectivity
    inter_sub = plv_matrix[np.ix_(LEFT_CHANNELS, RIGHT_CHANNELS)]
    inter_plv = float(np.mean(inter_sub))
    
    # Lateralization Asymmetry Index: (Left - Right) / (Left + Right)
    asymmetry_index = float((left_intra_plv - right_intra_plv) / (left_intra_plv + right_intra_plv + 1e-12))
    
    return {
        'plv_matrix': plv_matrix.tolist(),
        'channel_names': TCP_CHANNELS,
        'global_synchrony_index': gsi,
        'left_intra_plv': left_intra_plv,
        'right_intra_plv': right_intra_plv,
        'inter_hemispheric_plv': inter_plv,
        'asymmetry_index': asymmetry_index,
        'target_scales': target_scales
    }
