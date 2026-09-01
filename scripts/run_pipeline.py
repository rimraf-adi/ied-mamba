r"""
End-to-End Pipeline Runner for TUH EEG Event Corpus.

Runs:
1. Dataset Loader verification (.edf, .rec, .lab, .htk parsers & ACNS TCP Montage builder).
2. Signal Preprocessing & Windowing verification.
3. PyTorch EEGEventDataset & DataLoader batching benchmark.
4. Exploratory Data Analysis & plot generation saved to `d:\ied\output_analysis/`.
"""

import os
import sys
import time

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from torch.utils.data import DataLoader

from dataset_loader import parse_rec_file, parse_lab_file, parse_htk_file, read_edf_file, build_tcp_montage, EEGEventDataset
from preprocessing import preprocess_eeg, extract_sliding_windows, compute_band_powers, extract_time_domain_features
from exploratory_analysis import scan_corpus_events, generate_class_distribution_report, generate_spatial_heatmap, generate_psd_class_comparison, plot_eeg_trace_with_events

CORPUS_ROOT = r"d:\ied\TU-v2.0.1"
OUTPUT_DIR = r"d:\ied\output_analysis"

def main():
    print("=" * 80)
    print("[+] TUH EEG Event Corpus Pipeline & Preprocessing Verification")
    print("=" * 80)

    start_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Verify Dataset Loader Parsers
    # -------------------------------------------------------------
    print("\n[1/4] Testing Dataset Loader Parsers (.edf, .rec, .lab, .htk)...")
    sample_eval_dir = os.path.join(CORPUS_ROOT, 'edf', 'eval', '000')
    sample_edf = os.path.join(sample_eval_dir, 'bckg_000_a_.edf')
    sample_rec = os.path.join(sample_eval_dir, 'bckg_000_a_.rec')
    sample_lab = os.path.join(sample_eval_dir, 'bckg_000_a__ch000.lab')
    sample_htk = os.path.join(sample_eval_dir, 'bckg_000_a__ch000.htk')

    signals, fs, ch_names = read_edf_file(sample_edf)
    print(f"  OK Read EDF Signal Shape: {signals.shape} | Sampling Rate: {fs} Hz | Channels: {len(ch_names)}")

    montage_signals, montage_names = build_tcp_montage(signals, ch_names)
    print(f"  OK Built ACNS TCP Montage Shape: {montage_signals.shape} | Channels: {len(montage_names)}")

    rec_events = parse_rec_file(sample_rec)
    print(f"  OK Parsed .rec File: {len(rec_events)} event records found.")

    lab_events = parse_lab_file(sample_lab)
    print(f"  OK Parsed .lab File: {len(lab_events)} microsecond event records found.")

    if os.path.exists(sample_htk):
        htk_feats, htk_period, htk_kind = parse_htk_file(sample_htk)
        print(f"  OK Parsed .htk Feature File: Matrix Shape {htk_feats.shape} | Period: {htk_period} ns")

    # -------------------------------------------------------------
    # Step 2: Test Preprocessing Pipeline & Feature Extraction
    # -------------------------------------------------------------
    print("\n[2/4] Testing Signal Preprocessing, Normalization & Windowing...")
    prep_signals, _ = preprocess_eeg(signals, fs=fs, ch_names=ch_names, lowcut=0.5, highcut=50.0, notch_freq=60.0)
    print(f"  OK Preprocessed EEG Signals Shape: {prep_signals.shape}")

    windows, labels, meta = extract_sliding_windows(prep_signals, fs=fs, events=rec_events, window_sec=2.0, overlap_ratio=0.5)
    print(f"  OK Extracted Sliding Windows Shape: {windows.shape} | Labels Shape: {labels.shape}")

    band_powers = compute_band_powers(prep_signals, fs=fs)
    print(f"  OK Computed Band Powers: Delta={band_powers['delta'].shape}, Alpha={band_powers['alpha'].shape}, Beta={band_powers['beta'].shape}")

    time_feats = extract_time_domain_features(prep_signals)
    print(f"  OK Extracted Time-Domain Features: RMS={time_feats['rms'].shape}, Hjorth Activity={time_feats['hjorth_activity'].shape}")

    # -------------------------------------------------------------
    # Step 3: Test PyTorch EEGEventDataset & DataLoader
    # -------------------------------------------------------------
    print("\n[3/4] Testing PyTorch Dataset & DataLoader Integration...")
    ds = EEGEventDataset(root_dir=CORPUS_ROOT, split='eval', window_sec=2.0, stride_sec=1.0, max_files=5)
    print(f"  OK Created PyTorch EEGEventDataset with {len(ds)} total window samples.")

    if len(ds) > 0:
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        x_batch, y_batch, meta_batch = next(iter(loader))
        print(f"  OK PyTorch DataLoader Sample Batch Shape - Input (X): {x_batch.shape} | Target (Y): {y_batch.shape}")
        print(f"  OK Target Labels Sample Batch: {y_batch.tolist()}")

    # -------------------------------------------------------------
    # Step 4: Run Exploratory Data Analysis & Generate Visualizations
    # -------------------------------------------------------------
    print("\n[4/4] Executing Corpus Analysis & Saving Plots to output_analysis/...")
    events_df = scan_corpus_events(CORPUS_ROOT)
    print(f"  OK Scanned {len(events_df)} total event records across corpus.")

    if not events_df.empty:
        summary_df = generate_class_distribution_report(events_df, OUTPUT_DIR)
        generate_spatial_heatmap(events_df, OUTPUT_DIR)
        generate_psd_class_comparison(CORPUS_ROOT, OUTPUT_DIR, max_files_per_class=5)

        # Plot sample EEG trace with event overlay
        trace_plot_path = os.path.join(OUTPUT_DIR, 'sample_eeg_event_trace.png')
        plot_eeg_trace_with_events(sample_edf, sample_rec, trace_plot_path, start_sec=0.0, duration_sec=15.0)
        print(f"  OK Saved EEG Trace with Event Overlays to: {trace_plot_path}")

    elapsed = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"[+] Pipeline Completed Successfully in {elapsed:.2f} seconds!")
    print(f"[+] Analysis artifacts and plots saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
