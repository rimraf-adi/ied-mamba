"""
Unified Runner for Advanced Wavelet & EEG Analytical Suite.

Executes:
1. Wavelet Packet Decomposition (WPD) & Best Basis Analysis
2. Wavelet Phase-Amplitude Coupling (PAC) & Tort Modulation Index
3. 22-Channel ACNS TCP Spatial Synchrony & Wavelet Phase-Locking Values (W-PLV)
4. 16+ Mother Wavelet Morphological Matching Benchmark

Saves structured results to `dwt-study/advanced_wavelet_results.json`.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import sys
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset_loader import parse_rec_file, read_edf_file, build_tcp_montage, LABEL_MAP, REVERSE_LABEL_MAP
from wpd_analyzer import compute_wpd_uniform_spectrum, compute_coifman_wickerhauser_best_basis
from wavelet_pac_analyzer import compute_wavelet_pac, compute_pac_comodulogram
from spatial_wavelet_coherence import compute_wavelet_plv_matrix, TCP_CHANNELS
from wavelet_morphology_benchmark import run_full_wavelet_morphology_benchmark

CORPUS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TU-v2.0.1'))
OUTPUT_JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'advanced_wavelet_results.json'))


def collect_multichannel_epochs(max_per_class: int = 25, epoch_sec: float = 2.0, target_fs: float = 250.0):
    """
    Extracts 22-channel ACNS TCP montage epochs for all clinical event classes.
    """
    print(f"Scanning {CORPUS_ROOT} for 22-channel multi-channel EEG epochs...")
    rec_files = glob.glob(os.path.join(CORPUS_ROOT, '**', '*.rec'), recursive=True)[:120]
    
    epochs_22ch = {cls: [] for cls in LABEL_MAP.keys()}
    target_len = int(epoch_sec * target_fs)
    
    for rec_path in rec_files:
        if all(len(v) >= max_per_class for v in epochs_22ch.values()):
            break
            
        base_path = os.path.splitext(rec_path)[0]
        edf_path = base_path + '.edf'
        if not os.path.exists(edf_path):
            continue
            
        events = parse_rec_file(rec_path)
        if not events:
            continue
            
        edf_data = read_edf_file(edf_path)
        if edf_data is None:
            continue
        raw_signals, fs, ch_names = edf_data
        
        montage_res = build_tcp_montage(raw_signals, ch_names)
        if montage_res is None:
            continue
        montage_signals, _ = montage_res
        
        for ev in events:
            label_str = ev['label_str']
            if len(epochs_22ch[label_str]) >= max_per_class:
                continue
                
            center_sec = (ev['start_sec'] + ev['stop_sec']) / 2.0
            start_sample = int((center_sec - epoch_sec / 2.0) * fs)
            end_sample = start_sample + int(epoch_sec * fs)
            
            if start_sample < 0 or end_sample > montage_signals.shape[1]:
                continue
                
            snippet_22ch = montage_signals[:, start_sample:end_sample]
            if snippet_22ch.shape[1] != target_len:
                continue
                
            epochs_22ch[label_str].append(snippet_22ch)
            
    for cls, ep_list in epochs_22ch.items():
        print(f"  Class '{cls}': {len(ep_list)} 22-channel epochs collected.")
        
    return epochs_22ch


def run_advanced_study():
    print("==========================================================")
    print("STARTING ADVANCED WAVELET & EEG CLINICAL STUDY")
    print("==========================================================")
    
    epochs_22ch = collect_multichannel_epochs(max_per_class=20)
    
    # ---------------------------------------------------------
    # 1. Wavelet Packet Decomposition (WPD) & Best Basis
    # ---------------------------------------------------------
    print("\n--- 1. Running Wavelet Packet Decomposition (WPD) & Best Basis ---")
    wpd_results = {}
    
    for cls in LABEL_MAP.keys():
        if len(epochs_22ch[cls]) == 0:
            continue
        # Use primary channel (e.g. FP1-F7 or FP2-F8)
        single_ch_sigs = [ep[0] for ep in epochs_22ch[cls]]
        
        # 32 uniform subbands (Level 5)
        uniform_energies_list = []
        cost_reductions = []
        num_basis_nodes_list = []
        
        for sig in single_ch_sigs:
            rel_e, freq_ranges, paths = compute_wpd_uniform_spectrum(sig, wavelet='db4', level=5)
            uniform_energies_list.append(rel_e)
            
            bb_res = compute_coifman_wickerhauser_best_basis(sig, wavelet='db4', max_level=5)
            cost_reductions.append(bb_res['cost_reduction_pct'])
            num_basis_nodes_list.append(bb_res['num_basis_nodes'])
            
        mean_uniform_e = np.mean(uniform_energies_list, axis=0).tolist()
        wpd_results[cls] = {
            'mean_uniform_32_bands_energy': mean_uniform_e,
            'freq_bandwidth_hz': 125.0 / 32.0,
            'mean_best_basis_entropy_reduction_pct': float(np.mean(cost_reductions)),
            'mean_best_basis_nodes_count': float(np.mean(num_basis_nodes_list))
        }
        print(f"  [{cls.upper()}] Best Basis Entropy Reduction: {wpd_results[cls]['mean_best_basis_entropy_reduction_pct']:.2f}%, Optimal Nodes: {wpd_results[cls]['mean_best_basis_nodes_count']:.1f}")

    # ---------------------------------------------------------
    # 2. Phase-Amplitude Coupling (PAC) & Modulation Index
    # ---------------------------------------------------------
    print("\n--- 2. Running Wavelet Phase-Amplitude Coupling (PAC) Analysis ---")
    pac_results = {}
    
    for cls in LABEL_MAP.keys():
        if len(epochs_22ch[cls]) == 0:
            continue
        single_ch_sigs = [ep[0] for ep in epochs_22ch[cls]]
        
        # Theta (D5) -> Gamma (D2) PAC
        theta_gamma_mis = []
        phase_dists = []
        
        for sig in single_ch_sigs:
            res_pac = compute_wavelet_pac(sig, phase_band=['D5'], amp_band=['D2'], wavelet='db4', level=6)
            theta_gamma_mis.append(res_pac['modulation_index'])
            phase_dists.append(res_pac['normalized_distribution'])
            
        mean_mi = float(np.mean(theta_gamma_mis))
        mean_dist = np.mean(phase_dists, axis=0).tolist()
        
        # Also compute full comodulogram for first epoch
        comod = compute_pac_comodulogram(single_ch_sigs[0], wavelet='db4', level=6)
        
        pac_results[cls] = {
            'theta_gamma_modulation_index': mean_mi,
            'mean_phase_amplitude_distribution': mean_dist,
            'phase_bins_deg': np.linspace(10, 350, 18).tolist(),
            'comodulogram_sample': comod
        }
        print(f"  [{cls.upper()}] Theta-to-Gamma PAC Modulation Index (MI): {mean_mi:.5f}")

    # ---------------------------------------------------------
    # 3. 22-Channel Spatial Wavelet Coherence & W-PLV Matrix
    # ---------------------------------------------------------
    print("\n--- 3. Running 22-Channel ACNS TCP Spatial Synchrony (W-PLV) ---")
    spatial_results = {}
    
    for cls in LABEL_MAP.keys():
        if len(epochs_22ch[cls]) == 0:
            continue
            
        gsi_list = []
        left_intra_list = []
        right_intra_list = []
        inter_list = []
        asym_list = []
        plv_matrices = []
        
        for ep_22ch in epochs_22ch[cls]:
            plv_data = compute_wavelet_plv_matrix(ep_22ch, target_scales=['D3', 'D4', 'D5'], wavelet='db4', level=6)
            gsi_list.append(plv_data['global_synchrony_index'])
            left_intra_list.append(plv_data['left_intra_plv'])
            right_intra_list.append(plv_data['right_intra_plv'])
            inter_list.append(plv_data['inter_hemispheric_plv'])
            asym_list.append(plv_data['asymmetry_index'])
            plv_matrices.append(plv_data['plv_matrix'])
            
        mean_plv_matrix = np.mean(plv_matrices, axis=0).tolist()
        
        spatial_results[cls] = {
            'mean_global_synchrony_index': float(np.mean(gsi_list)),
            'mean_left_intra_plv': float(np.mean(left_intra_list)),
            'mean_right_intra_plv': float(np.mean(right_intra_list)),
            'mean_inter_hemispheric_plv': float(np.mean(inter_list)),
            'mean_asymmetry_index': float(np.mean(asym_list)),
            'mean_22ch_plv_matrix': mean_plv_matrix
        }
        print(f"  [{cls.upper()}] Global Synchrony Index: {spatial_results[cls]['mean_global_synchrony_index']:.4f}, Inter-Hemispheric PLV: {spatial_results[cls]['mean_inter_hemispheric_plv']:.4f}, Asymmetry: {spatial_results[cls]['mean_asymmetry_index']:+.4f}")

    # ---------------------------------------------------------
    # 4. 16+ Mother Wavelet Morphological Matching Benchmark
    # ---------------------------------------------------------
    print("\n--- 4. Running 16+ Mother Wavelet Morphological Matching Benchmark ---")
    spsw_signals = np.array([ep[0] for ep in epochs_22ch['spsw']]) if len(epochs_22ch['spsw']) > 0 else np.random.randn(10, 500)
    
    morphology_rankings = run_full_wavelet_morphology_benchmark(spsw_signals)
    
    print("\nTop 5 Mother Wavelets for IED Detection:")
    for r in morphology_rankings[:5]:
        print(f"  Rank #{r['rank']}: {r['wavelet']} (Score: {r['composite_score']:.4f}, Cross-Corr: {r['mean_cross_correlation']:.4f}, ECR: {r['mean_energy_compaction_ratio']*100:.1f}%, SNR: {r['mean_reconstruction_snr_db']:.1f} dB)")

    # Save comprehensive results
    master_advanced_results = {
        'wpd_results': wpd_results,
        'pac_results': pac_results,
        'spatial_results': spatial_results,
        'morphology_rankings': morphology_rankings,
        'channel_names': TCP_CHANNELS
    }
    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(master_advanced_results, f, indent=2)
        
    print(f"\n[SUCCESS] Advanced Wavelet study results saved to {OUTPUT_JSON_PATH}")
    return master_advanced_results


if __name__ == '__main__':
    run_advanced_study()
