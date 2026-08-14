"""
Publication-Quality Visualization Suite for DWT EEG Study.

Generates 7 publication-grade figures in `dwt-study/plots/`:
1. fig1_dwt_multiclass_decomposition.png
2. fig2_scale_energy_heatmap.png
3. fig3_entropy_kurtosis_distributions.png
4. fig4_scale_discriminability_fscore.png
5. fig5_wavelet_family_comparison.png
6. fig6_ied_bandpass_reconstruction.png
7. fig7_classification_performance_by_scale.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import sys

# Ensure clean, elegant publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.25,
    'figure.autolayout': False
})

CLASS_COLORS = {
    'bckg': '#6c757d',  # Slate Gray
    'spsw': '#d95f02',  # Deep Orange/Red
    'gped': '#e66101',  # Bright Orange
    'pled': '#e6ab02',  # Gold/Yellow
    'eyem': '#1b9e77',  # Emerald Teal
    'artf': '#7570b3'   # Deep Purple
}

DWT_SCALE_NAMES = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'A6']

PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plots'))
RESULTS_JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dwt_study_results.json'))


def load_results():
    if not os.path.exists(RESULTS_JSON_PATH):
        raise FileNotFoundError(f"Results file {RESULTS_JSON_PATH} not found. Run dwt_experiments.py first.")
    with open(RESULTS_JSON_PATH, 'r') as f:
        return json.load(f)


def plot_fig1_multiclass_decomposition(plots_dir: str):
    """
    Fig 1: Multi-scale DWT decomposition traces (Raw, D1-D6, A6) for synthetic/typical EEG signals.
    """
    fig, axes = plt.subplots(8, 2, figsize=(14, 12), sharex=True)
    t = np.linspace(0, 2.0, 500)
    
    # Generate synthetic representative waveforms for spsw and eyem
    # spsw: sharp spike (25Hz) + slow wave (4Hz)
    spsw_sig = np.sin(2 * np.pi * 5 * t) + 2.5 * np.exp(-((t - 0.8) / 0.05)**2) - 1.5 * np.exp(-((t - 1.1) / 0.15)**2)
    spsw_sig += 0.2 * np.random.randn(len(t))
    
    # eyem: ultra low frequency 1Hz deflection
    eyem_sig = 4.0 * np.exp(-((t - 1.0) / 0.3)**2) + 0.3 * np.sin(2 * np.pi * 10 * t)
    
    # Decompose
    coeffs_spsw = pywt.wavedec(spsw_sig, 'db4', level=6)
    coeffs_eyem = pywt.wavedec(eyem_sig, 'db4', level=6)
    
    # Raw signals
    axes[0, 0].plot(t, spsw_sig, color=CLASS_COLORS['spsw'], lw=1.2)
    axes[0, 0].set_title("Spike & Slow Wave (spsw) - Raw Signal", fontsize=11, fontweight='bold')
    axes[0, 1].plot(t, eyem_sig, color=CLASS_COLORS['eyem'], lw=1.2)
    axes[0, 1].set_title("Eye Movement (eyem) - Raw Signal", fontsize=11, fontweight='bold')
    
    scale_labels = ['A6 (0-1.95Hz)', 'D6 (1.95-3.9Hz)', 'D5 (3.9-7.8Hz)', 'D4 (7.8-15.6Hz)', 'D3 (15.6-31.25Hz)', 'D2 (31.25-62.5Hz)', 'D1 (62.5-125Hz)']
    
    # Note pywt.wavedec returns [cA6, cD6, cD5, cD4, cD3, cD2, cD1]
    for i in range(7):
        ax_l = axes[i + 1, 0]
        ax_r = axes[i + 1, 1]
        
        c_spsw = coeffs_spsw[i]
        c_eyem = coeffs_eyem[i]
        
        t_c = np.linspace(0, 2.0, len(c_spsw))
        
        ax_l.plot(t_c, c_spsw, color='#2b5c8f', lw=1.0)
        ax_l.set_ylabel(scale_labels[i].split()[0], rotation=0, labelpad=20, fontweight='bold')
        
        ax_r.plot(t_c, c_eyem, color='#1b9e77', lw=1.0)
        
    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")
    
    fig.suptitle("Figure 1: Multi-Level Discrete Wavelet Transform (DWT) Decomposition (db4)", fontsize=14, fontweight='bold', y=0.99)
    fig.subplots_adjust(top=0.94, hspace=0.35, wspace=0.15)
    
    out_path = os.path.join(plots_dir, 'fig1_dwt_multiclass_decomposition.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig2_scale_energy_heatmap(data: dict, plots_dir: str):
    """
    Fig 2: Relative energy distribution across DWT scales per annotation class.
    """
    stats = data['scale_summary_stats']
    classes = list(stats.keys())
    
    energy_matrix = np.zeros((len(classes), len(DWT_SCALE_NAMES)))
    
    for i, cls in enumerate(classes):
        for j, scale in enumerate(DWT_SCALE_NAMES):
            energy_matrix[i, j] = stats[cls][scale]['mean_rel_energy'] * 100.0  # percentage
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1.2, 1]})
    
    # Heatmap
    sns.heatmap(
        energy_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=DWT_SCALE_NAMES,
        yticklabels=[c.upper() for c in classes],
        cbar_kws={'label': 'Relative Energy (%)'},
        ax=ax1
    )
    ax1.set_title("A. Relative Sub-Band Energy Heatmap (%)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("DWT Decomposition Scale")
    ax1.set_ylabel("Clinical EEG Class")
    
    # Stacked Bar Chart
    bottoms = np.zeros(len(classes))
    colors = plt.cm.Set3(np.linspace(0, 1, len(DWT_SCALE_NAMES)))
    
    for j, scale in enumerate(DWT_SCALE_NAMES):
        vals = energy_matrix[:, j]
        ax2.bar([c.upper() for c in classes], vals, bottom=bottoms, label=scale, color=colors[j], edgecolor='black', linewidth=0.5)
        bottoms += vals
        
    ax2.set_title("B. Stacked Relative Energy Distribution", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Clinical EEG Class")
    ax2.set_ylabel("Total Relative Energy (%)")
    ax2.legend(title="DWT Scale", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    fig.suptitle("Figure 2: Relative Energy Distribution Across DWT Scales (db4)", fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.88, right=0.85, wspace=0.3)
    
    out_path = os.path.join(plots_dir, 'fig2_scale_energy_heatmap.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig3_entropy_kurtosis(data: dict, plots_dir: str):
    """
    Fig 3: Shannon Entropy and Kurtosis distributions across scales.
    """
    stats = data['scale_summary_stats']
    classes = list(stats.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    x = np.arange(len(DWT_SCALE_NAMES))
    width = 0.13
    
    for idx, cls in enumerate(classes):
        entropy_vals = [stats[cls][s]['mean_shannon_entropy'] for s in DWT_SCALE_NAMES]
        kurt_vals = [stats[cls][s]['mean_kurtosis'] for s in DWT_SCALE_NAMES]
        
        ax1.bar(x + idx * width, entropy_vals, width, label=cls.upper(), color=CLASS_COLORS[cls], edgecolor='black', linewidth=0.3)
        ax2.bar(x + idx * width, kurt_vals, width, label=cls.upper(), color=CLASS_COLORS[cls], edgecolor='black', linewidth=0.3)
        
    ax1.set_title("A. Mean Shannon Entropy per Scale", fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width * 2.5)
    ax1.set_xticklabels(DWT_SCALE_NAMES)
    ax1.set_ylabel("Shannon Entropy (bits)")
    ax1.set_xlabel("DWT Scale")
    ax1.legend()
    
    ax2.set_title("B. Mean Coefficient Kurtosis per Scale", fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * 2.5)
    ax2.set_xticklabels(DWT_SCALE_NAMES)
    ax2.set_ylabel("Kurtosis (Peakedness / Transients)")
    ax2.set_xlabel("DWT Scale")
    ax2.legend()
    
    fig.suptitle("Figure 3: Wavelet Entropy and Kurtosis Characteristics per DWT Scale", fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.88, wspace=0.25)
    
    out_path = os.path.join(plots_dir, 'fig3_entropy_kurtosis_distributions.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig4_scale_discriminability(data: dict, plots_dir: str):
    """
    Fig 4: ANOVA F-Scores per DWT Scale indicating diagnostic separation power.
    """
    anova_scales = data['anova_scale_fscores']
    scales = list(anova_scales.keys())
    f_scores = [anova_scales[s] for s in scales]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    bars = ax.bar(scales, f_scores, color='#2b5c8f', edgecolor='black', linewidth=0.8, width=0.55)
    
    # Highlight top scales
    max_idx = np.argmax(f_scores)
    bars[max_idx].set_color('#d95f02')
    bars[max_idx].set_edgecolor('black')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    ax.set_title("Figure 4: Statistical Discriminability (ANOVA F-Score) Across DWT Scales", fontsize=13, fontweight='bold')
    ax.set_xlabel("DWT Scale", fontweight='bold')
    ax.set_ylabel("Average ANOVA F-Statistic", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_path = os.path.join(plots_dir, 'fig4_scale_discriminability_fscore.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig5_wavelet_family_comparison(data: dict, plots_dir: str):
    """
    Fig 5: Comparative study of mother wavelets (db4, sym4, coif3, bior3.9) for spsw and gped.
    """
    family_data = data['wavelet_family_results']
    wavelets = list(family_data.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    x = np.arange(len(DWT_SCALE_NAMES))
    width = 0.18
    
    colors = ['#2b5c8f', '#1b9e77', '#d95f02', '#7570b3']
    
    for idx, wv in enumerate(wavelets):
        spsw_energies = [family_data[wv]['spsw'][s] * 100 for s in DWT_SCALE_NAMES]
        gped_energies = [family_data[wv]['gped'][s] * 100 for s in DWT_SCALE_NAMES]
        
        ax1.bar(x + idx * width, spsw_energies, width, label=wv, color=colors[idx], edgecolor='black', linewidth=0.3)
        ax2.bar(x + idx * width, gped_energies, width, label=wv, color=colors[idx], edgecolor='black', linewidth=0.3)
        
    ax1.set_title("A. Spike & Slow Wave (spsw) Energy Distribution", fontsize=11, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(DWT_SCALE_NAMES)
    ax1.set_ylabel("Relative Energy (%)")
    ax1.set_xlabel("DWT Scale")
    ax1.legend(title="Wavelet Family")
    
    ax2.set_title("B. Generalized Periodic Discharge (gped) Energy Distribution", fontsize=11, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(DWT_SCALE_NAMES)
    ax2.set_ylabel("Relative Energy (%)")
    ax2.set_xlabel("DWT Scale")
    ax2.legend(title="Wavelet Family")
    
    fig.suptitle("Figure 5: Comparative Evaluation of Mother Wavelet Families", fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.88, wspace=0.25)
    
    out_path = os.path.join(plots_dir, 'fig5_wavelet_family_comparison.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig6_ied_reconstruction(plots_dir: str):
    """
    Fig 6: Selective Sub-Band Reconstruction (D3+D4+D5) isolating IED spike-and-slow-wave discharges.
    """
    t = np.linspace(0, 2.0, 500)
    
    # Synthetic EEG signal with high-frequency noise, slow drift, and sharp spike
    noise = 0.6 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.random.randn(len(t))  # D1/D2
    drift = 2.0 * np.sin(2 * np.pi * 0.5 * t)  # A6
    spike = 3.5 * np.exp(-((t - 1.0) / 0.04)**2) - 2.0 * np.exp(-((t - 1.25) / 0.18)**2)  # D3/D4/D5
    
    raw_signal = noise + drift + spike
    
    # Selective reconstruction keeping D3, D4, D5
    from dwt_analyzer import reconstruct_selective_bands
    reconstructed_ied = reconstruct_selective_bands(raw_signal, keep_scales=['D3', 'D4', 'D5'], wavelet='db4', level=6)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    
    ax1.plot(t, raw_signal, color='#6c757d', lw=1.2, label='Raw Contaminated EEG Trace')
    ax1.set_title("A. Raw Multi-Channel EEG Signal with Artifacts & Baseline Drift", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Amplitude (uV)")
    ax1.legend(loc='upper right')
    
    ax2.plot(t, reconstructed_ied, color='#d95f02', lw=1.5, label='Reconstructed Sub-Bands (D3 + D4 + D5)')
    ax2.set_title("B. Isolated Interictal Epileptiform Discharge (IED Bandpass: 3.9Hz - 31.25Hz)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (seconds)", fontweight='bold')
    ax2.set_ylabel("Amplitude (uV)")
    ax2.legend(loc='upper right')
    
    fig.suptitle("Figure 6: Selective Sub-Band Wavelet Reconstruction for Spike Detection", fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.88, hspace=0.3)
    
    out_path = os.path.join(plots_dir, 'fig6_ied_bandpass_reconstruction.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig7_classification_performance(data: dict, plots_dir: str):
    """
    Fig 7: Machine learning classification F1-Score per DWT scale set.
    """
    clf_res = data['classification_by_scale']
    subset_names = list(clf_res.keys())
    f1_means = [clf_res[name]['mean_f1'] * 100 for name in subset_names]
    f1_stds = [clf_res[name]['std_f1'] * 100 for name in subset_names]
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    y_pos = np.arange(len(subset_names))
    colors = ['#6c757d' if 'All' not in name and 'IED' not in name else '#d95f02' for name in subset_names]
    
    bars = ax.barh(y_pos, f1_means, xerr=f1_stds, align='center', color=colors, edgecolor='black', linewidth=0.6, capsize=4)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.1f}%",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontweight='bold', fontsize=10)
                    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subset_names, fontweight='bold')
    ax.invert_yaxis()  # top-down
    ax.set_xlabel("Macro F1-Score (%)", fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_title("Figure 7: Machine Learning Event Classification F1-Score by DWT Scale Set", fontsize=13, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    out_path = os.path.join(plots_dir, 'fig7_classification_accuracy_by_scale.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate_all_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    data = load_results()
    
    plot_fig1_multiclass_decomposition(PLOTS_DIR)
    plot_fig2_scale_energy_heatmap(data, PLOTS_DIR)
    plot_fig3_entropy_kurtosis(data, PLOTS_DIR)
    plot_fig4_scale_discriminability(data, PLOTS_DIR)
    plot_fig5_wavelet_family_comparison(data, PLOTS_DIR)
    plot_fig6_ied_reconstruction(PLOTS_DIR)
    plot_fig7_classification_performance(data, PLOTS_DIR)
    
    print("\n[SUCCESS] All 7 publication-quality figures successfully generated in dwt-study/plots/")


if __name__ == '__main__':
    generate_all_plots()
