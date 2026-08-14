"""
16+ Mother Wavelet Morphological Matching & Cross-Correlation Benchmark.

Evaluates 16 mother wavelets across Daubechies (db), Symlets (sym), Coiflets (coif),
and Biorthogonal (bior) families against clinical spike-and-slow-wave (spsw) waveforms.
Computes Normalized Cross-Correlation, Energy Compaction Ratio (ECR), Reconstruction SNR,
and provides a definitive ranking for clinical EEG IED detection.
"""

import numpy as np
import pywt
from scipy import signal as sp_signal
from typing import Dict, List, Tuple

BENCHMARK_WAVELETS = [
    # Daubechies
    'db2', 'db4', 'db6', 'db8', 'db10',
    # Symlets
    'sym2', 'sym4', 'sym6', 'sym8',
    # Coiflets
    'coif1', 'coif3', 'coif5',
    # Biorthogonal
    'bior1.3', 'bior2.4', 'bior3.9', 'bior6.8'
]


def evaluate_wavelet_morphology(
    spike_signals: np.ndarray,  # shape: (N_spikes, N_samples)
    wavelet_name: str,
    level: int = 6
) -> Dict:
    """
    Evaluates a single mother wavelet against a collection of clinical spike epochs.
    Computes:
    - mean_cross_corr: Normalized cross-correlation between wavelet function psi(t) and spike
    - mean_ecr: Energy Compaction Ratio (energy in top 10% coefficients)
    - mean_snr: Reconstruction SNR after 80% coefficient thresholding (denoising capacity)
    """
    w = pywt.Wavelet(wavelet_name)
    wf_res = w.wavefun(level=7)
    if len(wf_res) == 5:
        phi_d, psi, phi_r, psi_r, x_grid = wf_res
    elif len(wf_res) == 3:
        phi, psi, x_grid = wf_res
    else:
        psi = wf_res[0]
    
    cross_corrs = []
    ecrs = []
    snrs = []
    
    # Target length of spike transient window (~100 ms around spike peak = ~25 samples @ 250Hz)
    psi_norm = psi / (np.linalg.norm(psi) + 1e-12)
    
    for sig in spike_signals:
        # 1. Normalized cross-correlation
        sig_norm = sig / (np.linalg.norm(sig) + 1e-12)
        
        # Resample wavelet psi to match spike length
        psi_resampled = sp_signal.resample(psi_norm, len(sig_norm))
        psi_resampled = psi_resampled / (np.linalg.norm(psi_resampled) + 1e-12)
        
        # Cross correlation
        corr = np.max(np.abs(sp_signal.correlate(sig_norm, psi_resampled, mode='same')))
        cross_corrs.append(float(corr))
        
        # 2. Multi-level DWT Energy Compaction Ratio (ECR)
        coeffs = pywt.wavedec(sig, wavelet=wavelet_name, level=level)
        all_c = np.concatenate(coeffs)
        abs_c_sorted = np.sort(np.abs(all_c))[::-1]
        
        total_energy = np.sum(abs_c_sorted ** 2) + 1e-12
        k_10pct = max(1, int(0.10 * len(abs_c_sorted)))
        top_energy = np.sum(abs_c_sorted[:k_10pct] ** 2)
        ecr = float(top_energy / total_energy)
        ecrs.append(ecr)
        
        # 3. Denoising Reconstruction SNR (reconstruct keeping top 20% coefficients)
        thresh_val = abs_c_sorted[int(0.20 * len(abs_c_sorted))] if len(abs_c_sorted) > 5 else 0.0
        thresh_coeffs = [pywt.threshold(c, value=thresh_val, mode='hard') for c in coeffs]
        rec_sig = pywt.waverec(thresh_coeffs, wavelet=wavelet_name)[:len(sig)]
        
        noise_err = sig - rec_sig
        sig_power = np.sum(sig ** 2) + 1e-12
        err_power = np.sum(noise_err ** 2) + 1e-12
        snr = float(10.0 * np.log10(sig_power / err_power))
        snrs.append(snr)
        
    mean_corr = float(np.mean(cross_corrs))
    mean_ecr = float(np.mean(ecrs))
    mean_snr = float(np.mean(snrs))
    
    return {
        'wavelet': wavelet_name,
        'family': wavelet_name[:4].rstrip('0123456789.'),
        'vanishing_moments_psi': getattr(w, 'vanishing_moments_psi', 0),
        'filter_length': w.dec_len,
        'is_orthogonal': w.orthogonal,
        'is_biorthogonal': w.biorthogonal,
        'is_symmetric': w.symmetry in ['symmetric', 'near symmetric'],
        'mean_cross_correlation': mean_corr,
        'mean_energy_compaction_ratio': mean_ecr,
        'mean_reconstruction_snr_db': mean_snr
    }


def run_full_wavelet_morphology_benchmark(spike_signals: np.ndarray) -> List[Dict]:
    """
    Benchmarks all 16 mother wavelets and ranks them by composite performance score.
    """
    results = []
    for wv in BENCHMARK_WAVELETS:
        metrics = evaluate_wavelet_morphology(spike_signals, wv)
        results.append(metrics)
        
    # Calculate normalized composite score
    max_corr = max(r['mean_cross_correlation'] for r in results) + 1e-12
    max_ecr = max(r['mean_energy_compaction_ratio'] for r in results) + 1e-12
    max_snr = max(r['mean_reconstruction_snr_db'] for r in results) + 1e-12
    
    for r in results:
        norm_corr = r['mean_cross_correlation'] / max_corr
        norm_ecr = r['mean_energy_compaction_ratio'] / max_ecr
        norm_snr = r['mean_reconstruction_snr_db'] / max_snr
        
        # Composite score: 40% Cross-correlation, 30% ECR, 30% SNR
        composite_score = 0.40 * norm_corr + 0.30 * norm_ecr + 0.30 * norm_snr
        r['composite_score'] = float(composite_score)
        
    # Sort descending by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # Assign ranks
    for rank, r in enumerate(results, 1):
        r['rank'] = rank
        
    return results
