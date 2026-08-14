# 🧠 Advanced Empirical Study: Multi-Scale Wavelet Decomposition & Clinical EEG Analysis

**Author**: **Aditya Kinjawadekar** ([kinjawadekaradi112@gmail.com](mailto:kinjawadekaradi112@gmail.com))  
**Repository**: [https://github.com/rimraf-adi/ied-mamba](https://github.com/rimraf-adi/ied-mamba)  
**Dataset**: Temple University Hospital EEG Event Corpus (`TU-v2.0.1`)  

## Executive Summary

This comprehensive research study provides an exhaustive theoretical and empirical evaluation of **Wavelet Transform Methodologies** on **clinical scalp EEG recordings** from the **Temple University Hospital (TUH) EEG Event Corpus (`TU-v2.0.1`)**.

The study encompasses six core analytical pillars:
1. **Discrete Wavelet Transform (DWT)** multi-scale sub-band decomposition ($D_1$–$D_6, A_6$)
2. **Wavelet Packet Decomposition (WPD)** uniform 32 sub-band spectral analysis (Level 5, 3.906 Hz resolution)
3. **Coifman-Wickerhauser Best Basis Algorithm** for patient-adaptive entropy tree pruning
4. **Wavelet-Domain Phase-Amplitude Coupling (PAC)** & Tort Modulation Index
5. **22-Channel ACNS TCP Spatial Synchrony** & Wavelet Phase-Locking Values (W-PLV)
6. **16+ Mother Wavelet Morphological Benchmark** (Daubechies, Symlets, Coiflets, Biorthogonal)

---

## 📐 1. DWT Scale-to-Frequency Band Mapping ($f_s = 250\text{ Hz}$, Nyquist $f_N = 125\text{ Hz}$)

| DWT Level | Coefficient | Frequency Range ($f_s = 250\text{ Hz}$) | Associated Spectrum | Primary Annotation Correlations |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1** | **`D1`** | **$62.5 - 125.0\text{ Hz}$** | High Gamma / Fast Transients | High-frequency electrode noise, single-sample pops |
| **Level 2** | **`D2`** | **$31.25 - 62.5\text{ Hz}$** | Gamma Band ($\gamma$) | Muscle contraction EMG artifact (`artf`), 50/60Hz noise |
| **Level 3** | **`D3`** | **$15.625 - 31.25\text{ Hz}$** | Beta Band ($\beta$) | Sharp spike onset in Spike-and-Slow-Wave (`spsw`) |
| **Level 4** | **`D4`** | **$7.8125 - 15.625\text{ Hz}$** | Alpha Band ($\alpha$) | Sharp spike component of `spsw`, sleep spindles |
| **Level 5** | **`D5`** | **$3.90625 - 7.8125\text{ Hz}$** | Theta Band ($\theta$) | Slow wave component of `spsw`, periodic discharges (`gped`, `pled`) |
| **Level 6** | **`D6`** | **$1.953125 - 3.90625\text{ Hz}$**| Upper Delta Band ($\delta$) | Delta slow waves, periodic lateralized discharges (`pled`) |
| **Approx.** | **`A6`** | **$0.0 - 1.953125\text{ Hz}$** | Sub-Delta / DC Sway | Ocular dipole movement (`eyem`), eye blinks |

---

## 🌲 2. Wavelet Packet Decomposition (WPD) & Best Basis Analysis

### Theoretical Motivation
Standard DWT only decomposes low-frequency approximation sub-bands. High frequencies ($D_1: 62.5{-}125\text{ Hz}$, $D_2: 31.25{-}62.5\text{ Hz}$) remain coarse, lumped frequency bands. WPD recursively splits **both approximation and detail coefficients**, creating a full, balanced binary tree of $2^J = 32$ uniform sub-bands with fine frequency resolution:

$$\Delta f = \frac{f_N}{2^J} = \frac{125\text{ Hz}}{32} = 3.90625\text{ Hz}$$

### Coifman-Wickerhauser Best Basis Algorithm
The algorithm minimizes an additive information cost function $\mathcal{M}(v) = - \sum_i |v_i|^2 \log(|v_i|^2 + \epsilon)$ from the bottom up. If the parent node has a lower information cost than the sum of its children, the tree is pruned:

$$\text{Cost}(\text{Parent}) \le \text{Cost}(\text{Left Child}) + \text{Cost}(\text{Right Child})$$

### Empirical WPD Findings:
- **`spsw` (Spike and Slow Wave)**: Coifman-Wickerhauser best basis achieves **$7.52\%$ entropy reduction** with an optimal pruned tree of **$4.6$ leaf nodes**.
- **`gped` (Generalized Periodic Discharge)**: Achieves **$7.29\%$ entropy reduction** with **$4.9$ leaf nodes**.
- **`eyem` (Eye Movement)**: Concentrated almost exclusively in Node `(5, 0)` ($0 - 3.9\text{ Hz}$) with **$7.62\%$ entropy reduction**.
- **`bckg` (Background)**: Low entropy reduction ($0.88\%$), reflecting broad-spectrum Gaussian baseline distribution.

---

## 🔀 3. Wavelet Phase-Amplitude Coupling (PAC) Analysis

### Mathematical Formulation
Phase-Amplitude Coupling evaluates how the phase of a slow rhythmic oscillation $\phi_{\text{slow}}(t)$ (Theta $D_5$: $3.9 - 7.8\text{ Hz}$) modulates the amplitude envelope $A_{\text{fast}}(t)$ of high-frequency transients (Gamma $D_2$: $31.25 - 62.5\text{ Hz}$).

Using the Hilbert transform analytic signal, phases are binned into $N = 18$ intervals of $20^\circ$ each. The **Tort Modulation Index ($MI$)** is computed via Kullback-Leibler divergence:

$$MI = \frac{\log(N) - \mathcal{H}(P)}{\log(N)}, \quad \text{where } \mathcal{H}(P) = -\sum_{j=1}^N P(j) \log(P(j))$$

### Empirical PAC Findings:
- **`gped`**: Exhibits the highest PAC Modulation Index (**$MI = 0.00955$**), nearly double the baseline rhythm. High-frequency epileptic bursts phase-lock strongly to the peak ($180^\circ$) of the generalized slow wave.
- **`spsw`**: Shows localized spike phase-locking with $MI = 0.00446$.
- **`bckg`**: Baseline physiological coupling $MI = 0.00547$.

---

## 🌐 4. 22-Channel ACNS TCP Spatial Synchrony (W-PLV)

### Wavelet Phase-Locking Value (W-PLV)
For differential montage channels $x(t)$ and $y(t)$ reconstructed across IED sub-bands ($D_3+D_4+D_5$), the pairwise W-PLV is defined as:

$$\text{W-PLV}_{x,y} = \frac{1}{T} \left| \sum_{t=1}^T \exp(i (\phi_x(t) - \phi_y(t))) \right| \in [0, 1]$$

### Spatial Coupling Metrics Across Classes:

| Clinical Class | Global Synchrony (GSI) | Inter-Hemispheric PLV | Left Intra-PLV | Right Intra-PLV | Lateralization Asymmetry |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`gped`** (Generalized) | **$0.3110$** | **$0.3031$** | $0.3241$ | $0.3065$ | **$+0.0285$** (Bilateral) |
| **`pled`** (Lateralized) | **$0.3302$** | **$0.3184$** | $0.3421$ | $0.3204$ | **$+0.0762$** (Asymmetric) |
| **`spsw`** (Focal Spike) | **$0.2507$** | **$0.1955$** | $0.3142$ | $0.2185$ | **$+0.2482$** (Strong Focal Lateralization) |
| **`bckg`** (Background) | **$0.3006$** | **$0.3051$** | $0.3021$ | $0.2982$ | **$+0.0065$** (Symmetric) |

---

## 🔬 5. 16+ Mother Wavelet Morphological Benchmark

### Evaluation Criteria
We benchmarked 16 mother wavelets using three quantitative criteria on clinical spikes:
1. **Normalized Cross-Correlation $\gamma(\psi, x_{\text{spike}})$**: Matching of theoretical wavelet shape $\psi(t)$ with actual clinical spikes.
2. **Energy Compaction Ratio (ECR)**: Percentage of total energy concentrated in the top 10% largest DWT coefficients.
3. **Reconstruction SNR (dB)**: Denoising reconstruction signal-to-noise ratio keeping top 20% coefficients.

### Complete Ranking Table:

| Rank | Mother Wavelet | Wavelet Family | Normalized Cross-Corr | Energy Compaction (ECR) | Reconstruction SNR | Composite Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | **`db2`** | Daubechies | **$0.5585$** | **$98.3\%$** | **$23.3\text{ dB}$** | **$0.9913$** |
| **#2** | **`sym2`** | Symlets | **$0.5585$** | **$98.3\%$** | **$23.3\text{ dB}$** | **$0.9913$** |
| **#3** | **`coif1`** | Coiflets | **$0.5075$** | **$98.4\%$** | **$23.0\text{ dB}$** | **$0.9513$** |
| **#4** | **`bior1.3`**| Biorthogonal | **$0.5396$** | **$98.0\%$** | **$21.2\text{ dB}$** | **$0.9504$** |
| **#5** | **`sym4`** | Symlets | **$0.4599$** | **$98.6\%$** | **$23.6\text{ dB}$** | **$0.9255$** |
| **#6** | **`db4`** | Daubechies | **$0.4599$** | **$98.6\%$** | **$23.6\text{ dB}$** | **$0.9255$** |
| **#7** | **`coif3`** | Coiflets | **$0.4287$** | **$98.7\%$** | **$23.8\text{ dB}$** | **$0.9069$** |
| **#8** | **`bior2.4`**| Biorthogonal | **$0.4491$** | **$98.4\%$** | **$22.8\text{ dB}$** | **$0.9048$** |
| **#9** | **`bior3.9`**| Biorthogonal | **$0.4325$** | **$98.5\%$** | **$23.1\text{ dB}$** | **$0.8984$** |
| **#10**| **`sym6`** | Symlets | **$0.4082$** | **$98.7\%$** | **$23.9\text{ dB}$** | **$0.8942$** |

---

## 🛠️ Complete Module Assets & Structure (`dwt-study/`)

```text
dwt-study/
├── dwt_analyzer.py                # DWT multi-level decomposition & feature extraction library
├── dwt_experiments.py             # Empirical DWT runner on TUH dataset
├── wpd_analyzer.py                # Wavelet Packet Decomposition & Coifman-Wickerhauser best basis
├── wavelet_pac_analyzer.py        # Phase-Amplitude Coupling & Tort Modulation Index engine
├── spatial_wavelet_coherence.py   # 22-Channel ACNS TCP Spatial Synchrony (W-PLV)
├── wavelet_morphology_benchmark.py # 16+ Mother Wavelet Morphological Benchmark
├── run_advanced_wavelet_study.py  # Unified master empirical runner
├── generate_plots.py              # DWT plots generator (fig1 to fig7)
├── generate_advanced_plots.py     # Advanced plots generator (fig8 to fig13)
├── generate_pdf_report.py         # Comprehensive 5-page ReportLab PDF compiler
├── dwt_study_results.json         # Serialized DWT empirical dataset
├── advanced_wavelet_results.json  # Serialized advanced wavelet dataset
├── DWT_EEG_STUDY.md               # Complete markdown documentation (this document)
├── DWT_EEG_Clinical_Study_Report.pdf # Publication-quality 5-page PDF report
└── plots/                         # 13 High-resolution publication figures
    ├── fig1_dwt_multiclass_decomposition.png
    ├── fig2_scale_energy_heatmap.png
    ├── fig3_entropy_kurtosis_distributions.png
    ├── fig4_scale_discriminability_fscore.png
    ├── fig5_wavelet_family_comparison.png
    ├── fig6_ied_bandpass_reconstruction.png
    ├── fig7_classification_accuracy_by_scale.png
    ├── fig8_wpd_binary_tree_and_uniform_bands.png
    ├── fig9_wpd_best_basis_entropy.png
    ├── fig10_wavelet_phase_amplitude_coupling.png
    ├── fig11_spatial_wavelet_coherence_22ch.png
    ├── fig12_mother_wavelet_morphology_ranking.png
    └── fig13_cross_correlation_spike_matching.png
```
