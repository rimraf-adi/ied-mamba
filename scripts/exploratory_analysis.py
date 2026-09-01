"""
Exploratory Data Analysis (EDA) & Visualization Suite for TUH EEG Event Corpus (Light Theme).

Computes:
- Event count distributions across splits (train/eval) and classes (spsw, gped, pled, eyem, artf, bckg).
- Event duration distribution metrics (mean, median, std, min, max, quantiles).
- Spatial event occurrence heatmaps across 22 TCP montage channels.
- Power Spectral Density (PSD) analysis per event class.
- Multi-channel EEG trace visualization with event annotation overlays.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from typing import Dict, List, Tuple, Optional

from dataset_loader import (
    parse_rec_file, parse_lab_file, read_edf_file, build_tcp_montage,
    LABEL_MAP, REVERSE_LABEL_MAP, TCP_MONTAGE_DEFINITIONS
)

# Set global publication quality light aesthetics and fixed margins
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
    'figure.autolayout': False
})

CLASS_COLORS = {
    'bckg': '#6c757d',  # Slate Gray
    'spsw': '#d95f02',  # Deep Orange / Red
    'gped': '#e66101',  # Orange
    'pled': '#e6ab02',  # Yellow/Gold
    'eyem': '#1b9e77',  # Teal / Emerald
    'artf': '#7570b3'   # Purple / Blue
}


def scan_corpus_events(corpus_root: str) -> pd.DataFrame:
    """
    Scans the entire TUH EEG Event Corpus directory and compiles a master DataFrame of all events.
    Columns: [split, session_id, file_name, channel_idx, channel_name, start_sec, stop_sec, duration_sec, label_str, label_id]
    """
    records = []

    for split in ['train', 'eval']:
        split_dir = os.path.join(corpus_root, 'edf', split)
        if not os.path.exists(split_dir):
            continue

        rec_files = glob.glob(os.path.join(split_dir, '**', '*.rec'), recursive=True)

        for rec_path in rec_files:
            rel_path = os.path.relpath(rec_path, split_dir)
            session_id = os.path.dirname(rel_path)
            file_name = os.path.basename(rec_path)

            events = parse_rec_file(rec_path)
            for ev in events:
                ch_idx = ev['channel_idx']
                ch_name = TCP_MONTAGE_DEFINITIONS[ch_idx][1] if 0 <= ch_idx < len(TCP_MONTAGE_DEFINITIONS) else f"CH_{ch_idx}"
                records.append({
                    'split': split,
                    'session_id': session_id,
                    'file_name': file_name,
                    'channel_idx': ch_idx,
                    'channel_name': ch_name,
                    'start_sec': ev['start_sec'],
                    'stop_sec': ev['stop_sec'],
                    'duration_sec': ev['duration_sec'],
                    'label_str': ev['label_str'],
                    'label_id': ev['label_id']
                })

    df = pd.DataFrame(records)
    return df


def generate_class_distribution_report(events_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Generates class distribution summary statistics and plots count bar chart (Light Theme & Fixed Margins).
    """
    os.makedirs(output_dir, exist_ok=True)

    summary = events_df.groupby(['split', 'label_str']).agg(
        event_count=('duration_sec', 'count'),
        total_duration_sec=('duration_sec', 'sum'),
        mean_duration_sec=('duration_sec', 'mean'),
        median_duration_sec=('duration_sec', 'median'),
        std_duration_sec=('duration_sec', 'std')
    ).reset_index()

    print("\n--- Event Class Distribution & Duration Summary ---")
    print(summary.to_string(index=False))

    # Plot 1: Event Count per Split & Class (Light Theme)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=events_df,
        x='label_str',
        hue='split',
        palette={'train': '#0284c7', 'eval': '#f59e0b'},
        estimator=len,
        errorbar=None,
        ax=ax
    )
    ax.set_title('TUH EEG Event Corpus - Event Counts per Split & Class', pad=15)
    ax.set_xlabel('Event Class', labelpad=10)
    ax.set_ylabel('Number of Event Instances', labelpad=10)
    ax.legend(title='Dataset Split', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.12)
    plot_path = os.path.join(output_dir, 'event_counts_per_class.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # Plot 2: Event Duration Distribution Boxplot (Light Theme)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=events_df,
        x='label_str',
        y='duration_sec',
        hue='split',
        palette={'train': '#0284c7', 'eval': '#f59e0b'},
        ax=ax
    )
    ax.set_yscale('log')
    ax.set_title('Event Duration Distribution (Log Scale)', pad=15)
    ax.set_xlabel('Event Class', labelpad=10)
    ax.set_ylabel('Duration (seconds)', labelpad=10)
    ax.legend(title='Dataset Split', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.12)
    duration_plot_path = os.path.join(output_dir, 'event_duration_boxplot.png')
    plt.savefig(duration_plot_path, bbox_inches='tight')
    plt.close()

    return summary


def generate_spatial_heatmap(events_df: pd.DataFrame, output_dir: str):
    """
    Generates a 2D heatmap showing event occurrences across the 22 TCP montage channels (Light Theme & Fixed Margins).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Cross-tabulate channel x label_str
    pivot_df = pd.crosstab(events_df['channel_name'], events_df['label_str'])

    # Reorder channels according to standard TCP Montage order
    tcp_order = [t[1] for t in TCP_MONTAGE_DEFINITIONS]
    pivot_df = pivot_df.reindex(tcp_order).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(pivot_df, annot=True, fmt='g', cmap='Blues', cbar_kws={'label': 'Event Count'}, ax=ax, linewidths=0.5, linecolor='#e2e8f0')
    ax.set_title('Spatial Occurrence of Events Across TCP Montage Channels', pad=15)
    ax.set_xlabel('Event Class', labelpad=10)
    ax.set_ylabel('TCP Montage Channel', labelpad=15)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.10)
    heatmap_path = os.path.join(output_dir, 'spatial_channel_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()


def plot_eeg_trace_with_events(edf_path: str,
                              rec_path: str,
                              output_path: str,
                              start_sec: float = 0.0,
                              duration_sec: float = 15.0):
    """
    Plots multi-channel EEG signals with color-coded event overlays (Light Theme & Fixed Margins).
    """
    signals, fs, ch_names = read_edf_file(edf_path)
    montage_signals, montage_names = build_tcp_montage(signals, ch_names)
    events = parse_rec_file(rec_path)

    n_channels = len(montage_names)
    start_sample = int(start_sec * fs)
    stop_sample = int((start_sec + duration_sec) * fs)
    stop_sample = min(stop_sample, montage_signals.shape[1])

    t = np.linspace(start_sec, stop_sample / fs, stop_sample - start_sample)

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 16), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ch_idx in range(n_channels):
        ax = axes[ch_idx]
        sig_chunk = montage_signals[ch_idx, start_sample:stop_sample]

        ax.plot(t, sig_chunk, color='#0284c7', linewidth=0.9)
        # Generous labelpad (50) so channel labels fit without clipping
        ax.set_ylabel(montage_names[ch_idx], rotation=0, labelpad=50, va='center', fontsize=9, color='#0f172a')
        ax.set_yticks([])
        ax.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
        ax.set_facecolor('#ffffff')

        # Highlight events on this channel
        for ev in events:
            if ev['channel_idx'] == ch_idx:
                ev_start = ev['start_sec']
                ev_stop = ev['stop_sec']

                if ev_stop >= start_sec and ev_start <= (start_sec + duration_sec):
                    cls_str = ev['label_str']
                    color = CLASS_COLORS.get(cls_str, '#ef4444')
                    ax.axvspan(max(start_sec, ev_start), min(start_sec + duration_sec, ev_stop),
                               color=color, alpha=0.35)

    axes[-1].set_xlabel('Time (seconds)', labelpad=10, fontsize=11, color='#0f172a')
    fig.suptitle(f'EEG Signal Traces with Event Overlays ({os.path.basename(edf_path)})', fontsize=14, y=0.99, color='#0f172a')
    plt.subplots_adjust(left=0.14, right=0.96, top=0.97, bottom=0.05)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def generate_psd_class_comparison(corpus_root: str, output_dir: str, max_files_per_class: int = 10):
    """
    Computes and plots Power Spectral Density (PSD) for each event class (Light Theme & Fixed Margins).
    """
    os.makedirs(output_dir, exist_ok=True)
    psd_by_class = {cls: [] for cls in LABEL_MAP.keys()}

    train_dir = os.path.join(corpus_root, 'edf', 'train')
    if not os.path.exists(train_dir):
        return

    edf_files = glob.glob(os.path.join(train_dir, '**', '*.edf'), recursive=True)

    for edf_path in edf_files:
        rec_path = os.path.splitext(edf_path)[0] + '.rec'
        events = parse_rec_file(rec_path)
        if not events:
            continue

        try:
            signals, fs, ch_names = read_edf_file(edf_path)
            montage_signals, _ = build_tcp_montage(signals, ch_names)
        except Exception:
            continue

        for ev in events:
            cls_str = ev['label_str']
            if len(psd_by_class[cls_str]) >= max_files_per_class:
                continue

            ch_idx = ev['channel_idx']
            if 0 <= ch_idx < montage_signals.shape[0]:
                start_sample = int(ev['start_sec'] * fs)
                stop_sample = int(ev['stop_sec'] * fs)
                segment = montage_signals[ch_idx, start_sample:stop_sample]

                if len(segment) >= int(0.5 * fs):
                    freqs, psd = signal.welch(segment, fs=fs, nperseg=min(len(segment), int(2*fs)))
                    psd_by_class[cls_str].append((freqs, psd))

    fig, ax = plt.subplots(figsize=(11, 6))
    for cls_str, psd_list in psd_by_class.items():
        if not psd_list:
            continue

        common_freqs = np.linspace(0.5, 50.0, 200)
        interp_psds = []

        for f, p in psd_list:
            interp_p = np.interp(common_freqs, f, p)
            interp_psds.append(interp_p)

        mean_psd = np.mean(interp_psds, axis=0)
        color = CLASS_COLORS.get(cls_str, '#000000')
        ax.semilogy(common_freqs, mean_psd, label=f"{cls_str.upper()} (n={len(psd_list)})", color=color, linewidth=2)

    ax.set_title('Power Spectral Density (PSD) Comparison by Event Class', pad=15)
    ax.set_xlabel('Frequency (Hz)', labelpad=10)
    ax.set_ylabel('Power Spectral Density (V^2 / Hz)', labelpad=10)
    ax.set_xlim(0.5, 50.0)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.12)
    psd_path = os.path.join(output_dir, 'event_class_psd_comparison.png')
    plt.savefig(psd_path, bbox_inches='tight')
    plt.close()
