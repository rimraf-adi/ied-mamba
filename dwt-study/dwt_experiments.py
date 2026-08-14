"""
DWT Empirical Study & Feature Extraction Engine.

Scans the TUH EEG Event Corpus (TU-v2.0.1), extracts annotated EEG epochs,
runs multi-level DWT feature extraction across wavelet families, performs ANOVA F-test
discriminability analysis, and evaluates scale-by-scale machine learning classification.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import sys
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Add parent directory to path to import dataset_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset_loader import parse_rec_file, read_edf_file, build_tcp_montage, LABEL_MAP, REVERSE_LABEL_MAP
from dwt_analyzer import (
    decompose_eeg_dwt, extract_multiscale_dwt_features, extract_subband_features,
    DWT_SCALE_NAMES, SCALE_FREQ_BANDS, SUPPORTED_WAVELETS
)

CORPUS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TU-v2.0.1'))
RESULTS_JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dwt_study_results.json'))


def collect_annotated_eeg_epochs(
    max_epochs_per_class: int = 40,
    epoch_duration_sec: float = 2.0,
    target_fs: float = 250.0
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Scans dataset directories, extracts centered 2-second single-channel EEG signal epochs for each event class.
    Returns:
    - X_epochs: np.ndarray of shape (N_samples, N_timepoints)
    - y_labels: np.ndarray of shape (N_samples,)
    - metadata: List of dicts containing session_id, channel_idx, label_str, etc.
    """
    print(f"Scanning corpus at {CORPUS_ROOT} for EEG event epochs...")
    rec_files = glob.glob(os.path.join(CORPUS_ROOT, '**', '*.rec'), recursive=True)
    rec_files = rec_files[:100]  # Capped for fast execution
    
    epochs_by_class = {code: [] for code in LABEL_MAP.keys()}
    metadata_list = []
    
    target_len = int(epoch_duration_sec * target_fs)
    
    # Process files until quota met
    for rec_path in rec_files:
        # Check if all classes met quota
        if all(len(v) >= max_epochs_per_class for v in epochs_by_class.values()):
            break
            
        base_path = os.path.splitext(rec_path)[0]
        edf_path = base_path + '.edf'
        
        if not os.path.exists(edf_path):
            continue
            
        events = parse_rec_file(rec_path)
        if not events:
            continue
            
        # Check if this rec file has any needed class
        needed = any(len(epochs_by_class[ev['label_str']]) < max_epochs_per_class for ev in events)
        if not needed:
            continue
            
        # Load EDF signal & TCP Montage
        edf_data = read_edf_file(edf_path)
        if edf_data is None:
            continue
            
        raw_signals, fs, ch_names = edf_data
        
        montage_res = build_tcp_montage(raw_signals, ch_names)
        if montage_res is None:
            continue
            
        montage_signals, montage_names = montage_res
        
        for ev in events:
            label_str = ev['label_str']
            if len(epochs_by_class[label_str]) >= max_epochs_per_class:
                continue
                
            ch_idx = ev['channel_idx']
            if ch_idx < 0 or ch_idx >= len(montage_names):
                ch_idx = 0
                
            ch_signal = montage_signals[ch_idx]
            ch_key = montage_names[ch_idx]
            
            # Extract signal snippet centered at event center
            start_sec = ev['start_sec']
            stop_sec = ev['stop_sec']
            center_sec = (start_sec + stop_sec) / 2.0
            
            start_sample = int((center_sec - epoch_duration_sec / 2.0) * fs)
            end_sample = start_sample + int(epoch_duration_sec * fs)
            
            if start_sample < 0 or end_sample > len(ch_signal):
                continue
                
            snippet = ch_signal[start_sample:end_sample]
            
            # Resample if fs != 250 Hz
            if abs(fs - target_fs) > 1.0:
                from scipy import signal as sp_signal
                snippet = sp_signal.resample(snippet, target_len)
            elif len(snippet) != target_len:
                snippet = snippet[:target_len] if len(snippet) > target_len else np.pad(snippet, (0, target_len - len(snippet)))
                
            epochs_by_class[label_str].append(snippet)
            metadata_list.append({
                'label_str': label_str,
                'label_id': LABEL_MAP[label_str],
                'channel_name': ch_key,
                'session': os.path.basename(os.path.dirname(rec_path))
            })

    X_list = []
    y_list = []
    meta_final = []
    
    for label_str, snippets in epochs_by_class.items():
        print(f"  Class '{label_str}': {len(snippets)} epochs extracted.")
        for snip in snippets:
            X_list.append(snip)
            y_list.append(LABEL_MAP[label_str])
            
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, metadata_list


def run_dwt_empirical_study():
    """
    Main execution function for DWT empirical study on TUH EEG dataset.
    """
    X, y, meta = collect_annotated_eeg_epochs(max_epochs_per_class=40)
    
    print("\n--- 1. Computing Multi-Scale DWT Features (db4) ---")
    dwt_results_by_class = {cls: {scale: {'rel_energy': [], 'shannon_entropy': [], 'kurtosis': [], 'std': [], 'mav': []} for scale in DWT_SCALE_NAMES} for cls in LABEL_MAP.keys()}
    
    all_scale_features = [] # Row per epoch, columns per scale feature
    
    for idx in range(len(X)):
        signal_epoch = X[idx]
        label_id = y[idx]
        label_str = REVERSE_LABEL_MAP[label_id]
        
        multiscale_info = extract_multiscale_dwt_features(signal_epoch, wavelet='db4', level=6)
        scale_feats = multiscale_info['scale_features']
        
        row_feats = {'label_id': label_id, 'label_str': label_str}
        
        for scale in DWT_SCALE_NAMES:
            sf = scale_feats[scale]
            dwt_results_by_class[label_str][scale]['rel_energy'].append(sf['rel_energy'])
            dwt_results_by_class[label_str][scale]['shannon_entropy'].append(sf['shannon_entropy'])
            dwt_results_by_class[label_str][scale]['kurtosis'].append(sf['kurtosis'])
            dwt_results_by_class[label_str][scale]['std'].append(sf['std'])
            dwt_results_by_class[label_str][scale]['mav'].append(sf['mav'])
            
            # Store in tabular format for ML classification & ANOVA
            for feat_key in ['rel_energy', 'shannon_entropy', 'kurtosis', 'std', 'mav', 'log_energy_entropy']:
                row_feats[f"{scale}_{feat_key}"] = sf[feat_key]
                
        all_scale_features.append(row_feats)
        
    df_features = pd.DataFrame(all_scale_features)
    
    # Calculate summary statistics per class per scale
    scale_summary_stats = {}
    for cls in LABEL_MAP.keys():
        scale_summary_stats[cls] = {}
        for scale in DWT_SCALE_NAMES:
            scale_summary_stats[cls][scale] = {
                'mean_rel_energy': float(np.mean(dwt_results_by_class[cls][scale]['rel_energy'])),
                'std_rel_energy': float(np.std(dwt_results_by_class[cls][scale]['rel_energy'])),
                'mean_shannon_entropy': float(np.mean(dwt_results_by_class[cls][scale]['shannon_entropy'])),
                'mean_kurtosis': float(np.mean(dwt_results_by_class[cls][scale]['kurtosis'])),
                'mean_std': float(np.mean(dwt_results_by_class[cls][scale]['std'])),
                'mean_mav': float(np.mean(dwt_results_by_class[cls][scale]['mav']))
            }

    print("\n--- 2. Computing Statistical Discriminability (ANOVA F-Scores) per Scale ---")
    feat_cols = [c for c in df_features.columns if c not in ['label_id', 'label_str']]
    X_mat = df_features[feat_cols].values
    y_vec = df_features['label_id'].values
    
    f_vals, p_vals = f_classif(X_mat, y_vec)
    
    anova_scores = {}
    for col, f_v, p_v in zip(feat_cols, f_vals, p_vals):
        anova_scores[col] = {'f_score': float(f_v), 'p_value': float(p_v)}
        
    # Scale-level aggregated ANOVA F-score
    scale_anova_agg = {}
    for scale in DWT_SCALE_NAMES:
        scale_cols = [c for c in feat_cols if c.startswith(scale + '_')]
        scale_f_avg = np.mean([anova_scores[c]['f_score'] for c in scale_cols])
        scale_anova_agg[scale] = float(scale_f_avg)

    print("\n--- 3. Testing Mother Wavelet Families Comparison ---")
    wavelet_family_results = {}
    for wv in SUPPORTED_WAVELETS:
        family_energies = {cls: {s: [] for s in DWT_SCALE_NAMES} for cls in LABEL_MAP.keys()}
        for idx in range(min(150, len(X))):
            signal_epoch = X[idx]
            label_str = REVERSE_LABEL_MAP[y[idx]]
            ms_info = extract_multiscale_dwt_features(signal_epoch, wavelet=wv, level=6)
            for s in DWT_SCALE_NAMES:
                family_energies[label_str][s].append(ms_info['scale_features'][s]['rel_energy'])
        
        wv_summary = {}
        for cls in LABEL_MAP.keys():
            wv_summary[cls] = {s: float(np.mean(family_energies[cls][s])) for s in DWT_SCALE_NAMES}
        wavelet_family_results[wv] = wv_summary

    print("\n--- 4. Machine Learning Classification Performance by Scale ---")
    # Evaluate individual scales and scale subsets
    scale_subsets = {
        'D1 (62.5-125Hz)': ['D1'],
        'D2 (31.25-62.5Hz)': ['D2'],
        'D3 (15.625-31.25Hz)': ['D3'],
        'D4 (7.8125-15.625Hz)': ['D4'],
        'D5 (3.906-7.8125Hz)': ['D5'],
        'D6 (1.953-3.906Hz)': ['D6'],
        'A6 (0-1.953Hz)': ['A6'],
        'High-Freq Noise (D1+D2)': ['D1', 'D2'],
        'IED Band (D3+D4+D5)': ['D3', 'D4', 'D5'],
        'Slow Discharges (D5+D6+A6)': ['D5', 'D6', 'A6'],
        'All Scales (D1-D6+A6)': DWT_SCALE_NAMES
    }
    
    cv_results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for subset_name, scales_included in scale_subsets.items():
        subset_cols = [c for c in feat_cols if any(c.startswith(s + '_') for s in scales_included)]
        X_sub = df_features[subset_cols].values
        
        f1_scores = []
        acc_scores = []
        
        for train_idx, test_idx in skf.split(X_sub, y_vec):
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_sub[train_idx], y_vec[train_idx])
            preds = clf.predict(X_sub[test_idx])
            
            f1_scores.append(f1_score(y_vec[test_idx], preds, average='macro'))
            acc_scores.append(accuracy_score(y_vec[test_idx], preds))
            
        cv_results[subset_name] = {
            'mean_f1': float(np.mean(f1_scores)),
            'std_f1': float(np.std(f1_scores)),
            'mean_accuracy': float(np.mean(acc_scores)),
            'std_accuracy': float(np.std(acc_scores))
        }
        print(f"  Subset '{subset_name}': F1-Score = {cv_results[subset_name]['mean_f1']:.4f} (+/- {cv_results[subset_name]['std_f1']:.4f})")

    # Compile master JSON results
    final_output = {
        'freq_bands': {k: list(v) for k, v in SCALE_FREQ_BANDS.items()},
        'scale_summary_stats': scale_summary_stats,
        'anova_scale_fscores': scale_anova_agg,
        'anova_feature_detail': anova_scores,
        'wavelet_family_results': wavelet_family_results,
        'classification_by_scale': cv_results,
        'num_epochs_processed': len(X),
        'class_counts': {cls: int(np.sum(y == LABEL_MAP[cls])) for cls in LABEL_MAP.keys()}
    }
    
    with open(RESULTS_JSON_PATH, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\n[SUCCESS] Empirical DWT results saved to {RESULTS_JSON_PATH}")
    return final_output, X, y, meta


if __name__ == '__main__':
    run_dwt_empirical_study()
