# Ordinal PSD Band-Power Ranking: Methodology & Framework Details

This document comprehensively explains the theoretical background, task formulation, architectures, and the rigorous evaluation protocol built in this repository.

---

## 1. Core Problem and Theoretical Rationale

### The Limitations of Direct Magnitude Regression
In typical Self-Supervised Learning (SSL) on physiological signals like EEG, models often regress raw signal magnitudes or absolute Power Spectral Density (PSD) as a pretext task. However, this suffers from several biological biases:
1. **$1/f$ Power Law Bias**: Low-frequency bands (Delta, Theta) naturally possess exponentially higher amplitudes than high-frequency bands (Beta, Gamma). A standard MSE loss causes models to blindly minimize Delta-band errors while completely ignoring high-frequency structural changes crucial for detecting seizures or focal slowing.
2. **Subject & Hardware Scaling Variance**: The absolute amplitude of an EEG channel varies drastically between patients due to skull thickness, impedance, or amplifier gain settings. 

### The Ordinal Ranking Solution
To force the model to learn the structural and relational topography of the brain rather than memorizing absolute patient-specific noise, we formulate spectral learning as an **Ordinal Ranking Task**. Instead of predicting "What is the absolute Alpha power in channel T3?", the model predicts "Is the Alpha power in T3 greater than in T4?". 
Ranking forces the encoder to build robust, invariant representations of relative structural gradients.

---

## 2. Multi-Variant Pretext Task Formulation

We extract standard Welch PSD and integrate it across 5 canonical clinical frequency bands:
- $\delta$ (0.5 - 4 Hz): Slow waves, sleep, pathology
- $\theta$ (4 - 8 Hz): Drowsiness, early slowing
- $\alpha$ (8 - 13 Hz): Awake, relaxed, occipital dominance
- $\beta$ (13 - 30 Hz): Active thinking, motor activity
- $\gamma$ (30 - 45 Hz): High cognitive processing

Based on these bands, we evaluate two orthogonal variants of the pretext task.

### Variant A: Spatial Topography (Cross-Channel)
- **Goal**: Rank the EEG channels for a specific frequency band at a single point in time.
- **Why**: Certain rhythms are highly localized. For example, Alpha rhythms are predominantly occipital (back of the head). By forcing the model to reconstruct this spatial gradient, it intrinsically maps the 2D topography of the scalp.
- **Target Construction**: The model computes the descending rank order of all 22 channels (e.g., `O1` > `C3` > `Fp1`) independently for all 5 bands.

### Variant C: Spectral Profile (Cross-Band)
- **Goal**: Rank the multiple frequency bands for a single channel in a single time window.
- **Why**: Recognizing whether a specific brain region is dominated by slow-wave $\delta$ (abnormal focal slowing) versus fast $\beta$ activity (awake state) requires understanding the internal spectral profile of a single electrode. It forces the encoder to recognize the *shape* of the frequency spectrum independently of amplitude.
- **Target Construction**: The model looks across the frequency band dimension, reconstructing the descending rank of the 5 canonical bands (e.g., $\delta > \theta > \alpha > \beta > \gamma$) for each channel.

---

## 3. High-Capacity Model Architectures

The framework uses a high-capacity embedding dimension of `512` to ensure sufficient representational space for production-scale EEG foundation modeling. The primary backbone evaluated in this suite is:

### Mamba (Selective State Space Model)
- **Architecture**: Employs the `mamba_ssm` (Mamba-2 SSD) blocks preceded by a multi-channel 1D temporal convolution front-end. It uses hardware-aware parallel state-space recurrence (`d_state=64`, `expand=2`).
- **Characteristics**: Achieves equivalent or superior quality to Transformers but operates with linear time complexity $O(N)$, drastically reducing memory overhead on long, continuous EEG recordings.

---

## 4. Differentiable Ranking Loss Functions

Since absolute ranks are non-differentiable step functions, the pipeline integrates a differentiable surrogate:

**Pairwise RankNet (with Tie Margin)**
- Decomposes the ranking task into $N(N-1)/2$ pairwise comparisons.
- Evaluates a log-sigmoid cross-entropy loss: $L_{ij} = -\bar{P}_{ij} \log P_{ij} - (1 - \bar{P}_{ij}) \log(1 - P_{ij})$.
- **Tie Margin ($\delta_{\text{margin}}$)**: A critical innovation to prevent the model from fighting over microscopic noise differences. If the true power difference between channel $i$ and $j$ is $< \delta_{\text{margin}}$, the pair is masked out and excluded from the loss.

---

## 5. Downstream Evaluation & Methodological Rigor

To measure the clinical utility of the pretext tasks, the pre-trained encoder is Fine-Tuned on the **TUEV Event Classification Dataset**. The pipeline is highly specialized to counteract both patient leakage and extreme class imbalance.

### Methodological Protections
Early iterations of this study identified two critical flaws commonly found in medical time-series literature, which have been definitively corrected in this framework:
1. **Strict Patient-Wise Isolation**: Randomly shuffling patient epochs across train/validation splits results in catastrophic "patient leakage" (allowing the model to memorize patients instead of clinical patterns). We enforce the official TUEV `train` and `eval` splits to guarantee 100% patient isolation.
2. **Prior-Calibrated Sampling**: Using `WeightedRandomSampler` on a naturally imbalanced medical dataset (1.7% positives) forces the model to learn a distorted 50/50 prior, causing boundary collapse when evaluating on the real validation prior. We use strict natural-prior sampling combined with an active **Optimal Binary Threshold Sweep** (Precision-Recall calibration) to properly evaluate classification metrics.

### Stage 1: Detection Task (Binary)
- **Objective**: Classify segments as Seizure/Event (`1`) vs. Background/Normal (`0`).
- **Classes**: 2 (98.3% Background, 1.7% Event)
- **Evaluation Logic**: We use `FocalLoss` ($\gamma=2.0$) to heavily penalize errors on minority events without distorting the batch distribution. During validation, we sweep all thresholds $[0.01, 0.99]$ to find the operating threshold that mathematically maximizes the Macro F1 score on the true prior.

### Stage 2: Typing Task (5-Class Multiclass)
- **Objective**: Conditioned on a segment being an event, classify the specific morphological subtype.
- **Classes**:
  - `SPSW`: Spike and Slow Wave
  - `GPED`: Generalized Periodic Epileptiform Discharges
  - `PLED`: Periodic Lateralized Epileptiform Discharges
  - `EYEM`: Eye Movement (Artifact)
  - `ARTF`: Chewing/Muscle Artifact

---

## 6. Hyperparameters and Experimental Settings

The study standardizes all signal processing, architectural, and optimization parameters to ensure fair comparisons.

### Signal Processing & Time Instants
- **Sampling Rate ($f_s$)**: 250 Hz (Standard TUEV resampling rate).
- **Time Instants (Window Length)**: **4.0 seconds** (1,000 samples).
- **Stride**: **2.0-second stride** (500 samples), iterating continuously across the raw unannotated signal.
- **Montage**: 22 channels (Standard ACNS TCP derivation).
- **PSD Extraction**: Welch's method with 1.0-second segments (`nperseg=250`) and 50% overlap.

### Multiprocessing Deadlock Prevention
To prevent OS-level deadlocks in Windows `spawn` multiprocessing workers, `memmap` numpy file handlers are strictly decoupled from the Dataset's pickling state via explicit `__getstate__` and `__setstate__` overrides. The memmap files are lazily loaded by worker threads upon their first access.

### Backbone Architecture Dimensions
- **Base Embedding Dimension (`embed_dim`)**: 512
- **Number of Encoder Layers**: 4
- **Dropout**: 0.1
- **Mamba (SSM) Specifics**: 
  - State Dimension (`d_state`): 64
  - Expansion Factor (`expand`): 2 (yielding an inner dimension of 1024)
  - Convolution Kernel (`d_conv`): 4

### Loss Formulation & Optimization
- **Tie-Margin ($\delta_{\text{margin}}$)**: $1 \times 10^{-3}$.
- **Epochs**: Pretraining 50 Epochs | Downstream Fine-Tuning 50 Epochs.
- **Early Stopping Patience**: 10 Epochs.
- **Optimizer**: AdamW (`weight_decay=1e-2`).
- **Data Export**: Outputs are streamed to an aggregate CSV and to detailed JSON artifacts nested by variant.
