"""
Publication Plot Generator for Advanced Wavelet & EEG Analyses.

Generates 6 advanced figures in `dwt-study/plots/`:
- fig8_wpd_binary_tree_and_uniform_bands.png
- fig9_wpd_best_basis_entropy.png
- fig10_wavelet_phase_amplitude_coupling.png
- fig11_spatial_wavelet_coherence_22ch.png
- fig12_mother_wavelet_morphology_ranking.png
- fig13_cross_correlation_spike_matching.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
from scipy import signal as sp_signal

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
    'bckg': '#6c757d',
    'spsw': '#d95f02',
    'gped': '#e66101',
    'pled': '#e6ab02',
    'eyem': '#1b9e77',
    'artf': '#7570b3'
}

PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plots'))
ADV_JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'advanced_wavelet_results.json'))


def load_adv_results():
    if not os.path.exists(ADV_JSON_PATH):
        raise FileNotFoundError(f"Missing results file: {ADV_JSON_PATH}")
    with open(ADV_JSON_PATH, 'r') as f:
        return json.load(f)


def plot_fig8_wpd_uniform_bands(data: dict, plots_dir: str):
    """
    Fig 8: 32 Uniform Sub-Bands Energy Spectrum (WPD Level 5) comparing spsw, artf, and bckg.
    """
    wpd_res = data['wpd_results']
    classes = ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg']
    
    n_bands = 32
    bw = 125.0 / n_bands  # 3.90625 Hz
    freq_centers = np.arange(n_bands) * bw + bw / 2.0
    
    fig, ax = plt.subplots(figsize=(14, 5.5))
    
    for cls in ['spsw', 'gped', 'artf', 'bckg']:
        if cls in wpd_res:
            energies = np.array(wpd_res[cls]['mean_uniform_32_bands_energy']) * 100.0
            ax.plot(freq_centers, energies, marker='o', lw=1.8, markersize=4, label=cls.upper(), color=CLASS_COLORS[cls])
            
    ax.set_title("Figure 8: Wavelet Packet Decomposition (WPD Level 5) - 32 Uniform Sub-Bands (3.9 Hz Resolution)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Center Frequency (Hz)", fontweight='bold')
    ax.set_ylabel("Relative Energy (%)", fontweight='bold')
    ax.set_xlim(0, 125)
    
    # Highlight clinical bands
    ax.axvspan(0, 4, color='#e0f2fe', alpha=0.5, label='Delta (0-4Hz)')
    ax.axvspan(4, 8, color='#fef3c7', alpha=0.5, label='Theta (4-8Hz)')
    ax.axvspan(8, 16, color='#fee2e2', alpha=0.5, label='Alpha/Spike (8-16Hz)')
    ax.axvspan(16, 32, color='#f3e8ff', alpha=0.5, label='Beta (16-32Hz)')
    ax.axvspan(32, 125, color='#f1f5f9', alpha=0.5, label='Gamma/EMG (>32Hz)')
    
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.subplots_adjust(top=0.90, right=0.82)
    
    out_path = os.path.join(plots_dir, 'fig8_wpd_binary_tree_and_uniform_bands.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig9_wpd_best_basis(data: dict, plots_dir: str):
    """
    Fig 9: Coifman-Wickerhauser Best Basis Entropy Reduction & Optimal Node Count across classes.
    """
    wpd_res = data['wpd_results']
    classes = list(wpd_res.keys())
    
    reductions = [wpd_res[c]['mean_best_basis_entropy_reduction_pct'] for c in classes]
    node_counts = [wpd_res[c]['mean_best_basis_nodes_count'] for c in classes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    colors = [CLASS_COLORS.get(c, '#2b5c8f') for c in classes]
    
    # Ax1: Entropy reduction percentage
    bars1 = ax1.bar([c.upper() for c in classes], reductions, color=colors, edgecolor='black', linewidth=0.6, width=0.55)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    ax1.set_title("A. Coifman-Wickerhauser Entropy Reduction (%)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Shannon Entropy Reduction vs Root (%)", fontweight='bold')
    ax1.set_xlabel("Clinical EEG Class", fontweight='bold')
    
    # Ax2: Optimal basis node count
    bars2 = ax2.bar([c.upper() for c in classes], node_counts, color=colors, edgecolor='black', linewidth=0.6, width=0.55)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    ax2.set_title("B. Optimal Best Basis Leaf Nodes Count", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Optimal Number of Basis Nodes", fontweight='bold')
    ax2.set_xlabel("Clinical EEG Class", fontweight='bold')
    
    fig.suptitle("Figure 9: Coifman-Wickerhauser Best Basis Analysis per Clinical Event Class", fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.88, wspace=0.25)
    
    out_path = os.path.join(plots_dir, 'fig9_wpd_best_basis_entropy.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig10_wavelet_pac(data: dict, plots_dir: str):
    """
    Fig 10: Phase-Amplitude Coupling (PAC) Modulation Index (MI) & Phase-Amplitude Distributions.
    """
    pac_res = data['pac_results']
    classes = list(pac_res.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Bar chart of Modulation Index
    mis = [pac_res[c]['theta_gamma_modulation_index'] * 1000.0 for c in classes]  # Scale by 1000 for display
    colors = [CLASS_COLORS.get(c, '#2b5c8f') for c in classes]
    bars = ax1.bar([c.upper() for c in classes], mis, color=colors, edgecolor='black', linewidth=0.6, width=0.55)
    for b in bars:
        h = b.get_height()
        ax1.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    ax1.set_title("A. Theta (D5) -> Gamma (D2) Modulation Index (x10^-3)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Tort Modulation Index (x10^-3)", fontweight='bold')
    ax1.set_xlabel("Clinical EEG Class", fontweight='bold')
    
    # Phase distribution polar or line plot
    bins_deg = pac_res['spsw']['phase_bins_deg']
    for cls in ['spsw', 'gped', 'bckg']:
        if cls in pac_res:
            dist = np.array(pac_res[cls]['mean_phase_amplitude_distribution'])
            ax2.plot(bins_deg, dist, marker='s', lw=2.0, label=cls.upper(), color=CLASS_COLORS[cls])
            
    ax2.set_title("B. Fast Gamma Amplitude across Slow Theta Phase Bins", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Theta Phase (degrees, 0-360)", fontweight='bold')
    ax2.set_ylabel("Normalized Gamma Amplitude", fontweight='bold')
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.set_xticklabels(['0° (Trough)', '90° (Ascent)', '180° (Peak)', '270° (Descent)', '360°'])
    ax2.legend()
    
    fig.suptitle("Figure 10: Wavelet Phase-Amplitude Coupling (PAC) in Epileptiform vs Normal EEG", fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.88, wspace=0.25)
    
    out_path = os.path.join(plots_dir, 'fig10_wavelet_phase_amplitude_coupling.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig11_spatial_coherence_22ch(data: dict, plots_dir: str):
    """
    Fig 11: 22x22 ACNS TCP Spatial Wavelet Phase-Locking Value (W-PLV) Connectivity Heatmaps.
    """
    spatial_res = data['spatial_results']
    ch_names = data['channel_names']
    
    # Pick GPED (bilateral) vs PLED (lateralized) vs BCKG
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    
    classes_to_show = [('gped', 'GPED (Generalized Periodic)'), ('pled', 'PLED (Periodic Lateralized)'), ('bckg', 'BCKG (Background Rhythm)')]
    
    for idx, (cls, title) in enumerate(classes_to_show):
        ax = axes[idx]
        if cls in spatial_res:
            matrix = np.array(spatial_res[cls]['mean_22ch_plv_matrix'])
            sns.heatmap(
                matrix,
                cmap='viridis',
                vmin=0.1,
                vmax=0.9,
                xticklabels=ch_names if idx == 1 else False,
                yticklabels=ch_names if idx == 0 else False,
                cbar=(idx == 2),
                cbar_kws={'label': 'Wavelet Phase-Locking Value (W-PLV)'},
                ax=ax
            )
            gsi = spatial_res[cls]['mean_global_synchrony_index']
            asym = spatial_res[cls]['mean_asymmetry_index']
            ax.set_title(f"{title}\nGSI: {gsi:.3f} | Asym: {asym:+.2f}", fontsize=11, fontweight='bold')
            if idx == 1:
                ax.tick_params(axis='x', rotation=90, labelsize=7)
            if idx == 0:
                ax.tick_params(axis='y', rotation=0, labelsize=7)
                
    fig.suptitle("Figure 11: 22-Channel ACNS TCP Spatial Wavelet Phase-Locking Matrices (IED Sub-Bands D3+D4+D5)", fontsize=13, fontweight='bold')
    fig.subplots_adjust(top=0.86, wspace=0.15)
    
    out_path = os.path.join(plots_dir, 'fig11_spatial_wavelet_coherence_22ch.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig12_morphology_ranking(data: dict, plots_dir: str):
    """
    Fig 12: 16+ Mother Wavelet Morphological Benchmark Rankings (Composite Score, Cross-Corr, ECR).
    """
    rankings = data['morphology_rankings']
    
    wavelets = [r['wavelet'] for r in rankings]
    scores = [r['composite_score'] for r in rankings]
    corrs = [r['mean_cross_correlation'] for r in rankings]
    ecrs = [r['mean_energy_compaction_ratio'] * 100 for r in rankings]
    snrs = [r['mean_reconstruction_snr_db'] for r in rankings]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    y_pos = np.arange(len(wavelets))
    
    # Ax1: Composite Quality Score
    colors = ['#d95f02' if i < 3 else '#2b5c8f' for i in range(len(wavelets))]
    bars = ax1.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=0.6)
    for b in bars:
        w = b.get_width()
        ax1.annotate(f"{w:.3f}", xy=(w, b.get_y() + b.get_height()/2), xytext=(4, 0), textcoords="offset points", ha='left', va='center', fontweight='bold', fontsize=9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(wavelets, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlabel("Composite Quality Score", fontweight='bold')
    ax1.set_title("A. Ranked 16+ Mother Wavelets for IED Spikes", fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1.15)
    
    # Ax2: Multi-metric comparison (Scatter: Cross-Corr vs Energy Compaction)
    for r in rankings:
        wv_fam = r['family']
        fam_color = '#d95f02' if 'db' in wv_fam else ('#1b9e77' if 'sym' in wv_fam else ('#7570b3' if 'coif' in wv_fam else '#e6ab02'))
        ax2.scatter(r['mean_cross_correlation'], r['mean_energy_compaction_ratio'] * 100, s=r['mean_reconstruction_snr_db'] * 8, color=fam_color, edgecolors='black', alpha=0.85)
        ax2.annotate(r['wavelet'], xy=(r['mean_cross_correlation'], r['mean_energy_compaction_ratio'] * 100), xytext=(4, 2), textcoords="offset points", fontsize=8.5, fontweight='bold')
        
    ax2.set_title("B. Cross-Correlation vs Energy Compaction (Bubble size = SNR dB)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Normalized Cross-Correlation with Clinical Spike", fontweight='bold')
    ax2.set_ylabel("Energy Compaction Ratio (% in top 10% coeffs)", fontweight='bold')
    
    fig.suptitle("Figure 12: Comprehensive 16+ Mother Wavelet Morphological Matching Benchmark", fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.88, wspace=0.25)
    
    out_path = os.path.join(plots_dir, 'fig12_mother_wavelet_morphology_ranking.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig13_cross_correlation_matching(plots_dir: str):
    """
    Fig 13: Continuous Wavelet psi(t) Shape Overlay against Prototypical Spike-and-Slow-Wave.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    t = np.linspace(-0.2, 0.4, 300)
    
    # Prototypical spike + slow wave
    spike_peak = 3.5 * np.exp(-((t) / 0.03)**2)
    slow_wave = -2.0 * np.exp(-((t - 0.15) / 0.12)**2)
    spike_model = (spike_peak + slow_wave)
    spike_model /= np.max(np.abs(spike_model))
    
    wavelets_to_plot = ['db4', 'sym4', 'coif3', 'bior3.9']
    
    for idx, wv_name in enumerate(wavelets_to_plot):
        ax = axes[idx // 2, idx % 2]
        w = pywt.Wavelet(wv_name)
        wf_res = w.wavefun(level=7)
        if len(wf_res) == 5:
            phi_d, psi, phi_r, psi_r, x_grid = wf_res
        elif len(wf_res) == 3:
            phi, psi, x_grid = wf_res
        else:
            psi = wf_res[0]
        
        # Resample psi to match
        psi_resampled = sp_signal.resample(psi, len(t))
        psi_resampled /= np.max(np.abs(psi_resampled))
        
        # Align peak
        lag = np.argmax(sp_signal.correlate(spike_model, psi_resampled, mode='same')) - len(t)//2
        psi_aligned = np.roll(psi_resampled, lag)
        
        corr = np.max(np.abs(sp_signal.correlate(spike_model / np.linalg.norm(spike_model), psi_aligned / np.linalg.norm(psi_aligned), mode='same')))
        
        ax.plot(t, spike_model, color='#d95f02', lw=2.0, label='Clinical Spike-and-Slow-Wave')
        ax.plot(t, psi_aligned, color='#2b5c8f', lw=1.5, linestyle='--', label=f'Wavelet Function ψ(t) [{wv_name}]')
        ax.set_title(f"{wv_name.upper()} (Cross-Correlation: {corr:.3f})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Normalized Amplitude")
        ax.legend(fontsize=8.5, loc='upper right')
        
    fig.suptitle("Figure 13: Morphological Alignment of Mother Wavelets ψ(t) with Clinical Epileptic Spikes", fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.20)
    
    out_path = os.path.join(plots_dir, 'fig13_cross_correlation_spike_matching.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate_all_advanced_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    data = load_adv_results()
    
    plot_fig8_wpd_uniform_bands(data, PLOTS_DIR)
    plot_fig9_wpd_best_basis(data, PLOTS_DIR)
    plot_fig10_wavelet_pac(data, PLOTS_DIR)
    plot_fig11_spatial_coherence_22ch(data, PLOTS_DIR)
    plot_fig12_morphology_ranking(data, PLOTS_DIR)
    plot_fig13_cross_correlation_matching(PLOTS_DIR)
    
    print("\n[SUCCESS] All 6 advanced figures generated in dwt-study/plots/")


if __name__ == '__main__':
    generate_all_advanced_plots()
