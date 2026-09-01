# Ordinal PSD Band-Power Ranking: Methodology & Framework Details

This document comprehensively explains the theoretical background, task formulation, architectures, and evaluation logic built in this directory.

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

Based on these bands, we define three orthogonal variants of the pretext task.

### 🧠 Conceptual Dry Run

Let's assume a simplified scenario to understand how the tensors and targets work. 
Imagine an EEG segment with **3 channels** (e.g., `O1`, `C3`, `Fp1`), **2 bands** ($\alpha, \delta$), and **2 time windows** ($T_1, T_2$).

**Raw PSD Extracted (Log Scale for Illustration)**:
- $T_1$: 
  - `O1`: $\alpha=10.5, \delta=2.0$
  - `C3`: $\alpha=5.1, \delta=4.0$
  - `Fp1`: $\alpha=2.0, \delta=8.5$
- $T_2$: 
  - `O1`: $\alpha=11.2, \delta=1.5$
  - `C3`: $\alpha=6.0, \delta=3.8$
  - `Fp1`: $\alpha=1.5, \delta=9.0$

How do the three variants create self-supervised ranking targets out of this matrix?

---

### Variant A: Spatial Topography (Cross-Channel)
- **Goal**: Rank the EEG channels for a specific frequency band at a single point in time.
- **Why**: Certain rhythms are highly localized. For example, Alpha rhythms are predominantly occipital (back of the head). By forcing the model to reconstruct this spatial gradient, it intrinsically maps the 2D topography of the scalp.
- **Dry Run Example (Band = $\alpha$, Window = $T_1$)**:
  - The PSD values are: `O1` (10.5), `C3` (5.1), `Fp1` (2.0)
  - The model does **not** predict the raw numbers (10.5, 5.1, 2.0).
  - Instead, the target is the rank descending order: `O1` > `C3` > `Fp1` (Rank indices: `0, 1, 2`).
  - If the model predicts a score of `4.0` for `O1`, `2.1` for `C3`, and `-1.0` for `Fp1`, its predicted ranking perfectly matches the true rank, yielding a loss of zero, even though its raw predicted numbers don't match the PSD values.

### Variant B: Temporal Drift (Cross-Time / Multi-Window Scenario)
- **Goal**: Rank a continuous sequence of non-overlapping time windows for a fixed channel and band.
- **Why**: Clinical conditions like seizures (sz) or periodic discharges (PLEDs) evolve over time. Predicting which window holds the most power forces the model to capture non-stationary temporal dynamics and state transitions, instead of treating each 4-second window as an isolated snapshot.
- **The Multi-Window Mechanics (Channel = `O1`, Band = $\alpha$)**:
  - Suppose we feed the model a sequence of $K=4$ consecutive windows: $T_1, T_2, T_3, T_4$.
  - The extracted PSD values for this sequence are: $T_1 (10.5), T_2 (11.2), T_3 (9.1), T_4 (12.0)$.
  - **Target Construction**: The model looks across the *temporal sequence dimension*. The target ranking is sorted descending: $T_4 > T_2 > T_1 > T_3$.
  - **Model Execution**: The backbone processes all 4 windows, pooling the temporal sequence latent vectors. The scoring head outputs a single scalar score per window. If the predicted scores are $S_{T_1}=0.5, S_{T_2}=1.2, S_{T_3}=-0.3, S_{T_4}=2.5$, the model correctly deduced the temporal drift sequence (the predicted ranking $2.5 > 1.2 > 0.5 > -0.3$ perfectly matches $T_4 > T_2 > T_1 > T_3$).

### Variant C: Spectral Profile (Cross-Band / Multi-Band Scenario)
- **Goal**: Rank the multiple frequency bands for a single channel in a single time window.
- **Why**: Recognizing whether a specific brain region is dominated by slow-wave $\delta$ (abnormal focal slowing) versus fast $\beta$ activity (awake state) requires understanding the internal spectral profile of a single electrode. It forces the encoder to recognize the *shape* of the frequency spectrum independently of amplitude.
- **The Multi-Band Mechanics (Channel = `Fp1`, Window = $T_1$)**:
  - The model considers all 5 canonical bands simultaneously: $\delta, \theta, \alpha, \beta, \gamma$.
  - Suppose the raw PSD values are: $\delta (8.5), \theta (4.1), \alpha (2.0), \beta (1.5), \gamma (0.2)$.
  - **Target Construction**: The model looks across the *frequency band dimension*. The target ranking descending order is: $\delta > \theta > \alpha > \beta > \gamma$.
  - **Model Execution**: The backbone generates the latent embedding for the `Fp1` channel. This latent vector is fed into 5 distinct band-specific scoring heads (or conditioned on band embeddings). The model must output 5 scores. To achieve zero loss, the model must assign the highest scalar score to the $\delta$ head and the lowest to the $\gamma$ head, thus successfully reconstructing the relative multi-band spectral shape.

---

## 3. High-Capacity Model Architectures

The framework uses an embed dimension of `256`, standard for production-scale EEG foundation models (like BIOT or LaBraM). The pipeline supports two state-of-the-art backbones:

### 1. Spatial-Temporal Transformer
- **Architecture**: A multi-channel 1D temporal convolution front-end projects raw EEG samples into a sequence of latent tokens. These tokens are processed by 4 stacked Multi-Head Self-Attention layers (`num_heads=8`, `dim_feedforward=512`).
- **Characteristics**: Global receptive field via self-attention, but computationally scales quadratically $O(N^2)$ with sequence length.

### 2. Mamba (Selective State Space Model)
- **Architecture**: Employs the `mamba_ssm` (Mamba-2 SSD) blocks. It uses the same temporal convolution front-end but replaces self-attention with hardware-aware parallel state-space recurrence (`d_state=64`, `expand=2`).
- **Characteristics**: Matches or exceeds Transformer quality but operates with linear time complexity $O(N)$, significantly reducing memory overhead on long EEG continuous recordings.

---

## 4. Differentiable Ranking Loss Functions

Since absolute ranks are non-differentiable step functions, the pipeline integrates several differentiable surrogates:

1. **Pairwise RankNet (with Tie Margin)**
   - Decomposes the ranking task into $N(N-1)/2$ pairwise comparisons.
   - Evaluates a log-sigmoid cross-entropy loss: $L_{ij} = -\bar{P}_{ij} \log P_{ij} - (1 - \bar{P}_{ij}) \log(1 - P_{ij})$.
   - **Tie Margin ($\delta_{\text{margin}}$)**: A critical innovation to prevent the model from fighting over microscopic noise differences. If the true power difference between channel $i$ and $j$ is $< \delta_{\text{margin}}$, the pair is masked out and excluded from the loss.

2. **Listwise Approaches (ListNet & ListMLE)**
   - **ListNet**: Treats the ranking scores as a probability distribution over the top-1 element (via Softmax) and uses Cross-Entropy against the true distribution.
   - **ListMLE**: Utilizes the Plackett-Luce model for permutations, evaluating the negative log-likelihood of the true descending rank permutation.

3. **Soft-Spearman**
   - Applies a continuous, differentiable sort operator (via soft-rank projections) to compute a differentiable approximation of the Spearman $\rho$ correlation, which acts directly as the loss.

---

## 5. Downstream Evaluation on TUEV

To measure the quality of representations learned by the pretext tasks, the pre-trained encoder is frozen and attached to a **Linear Probe** evaluation head on the **TUEV Event Classification Dataset**.

- **Classes (6)**: 
  - `SPSW`: Spike and Slow Wave (Seizure biomarker)
  - `GPED`: Generalized Periodic Epileptiform Discharges
  - `PLED`: Periodic Lateralized Epileptiform Discharges
  - `EYEM`: Eye Movement (Artifact)
  - `ARTF`: Chewing/Muscle Artifact
  - `BCKG`: Background / Normal Activity

- **Metrics**: 
  - Because clinical events are rare, standard accuracy is misleading. 
  - We log **Macro F1-Score**, **Balanced Accuracy**, and **AUROC** as primary validation metrics. 
  - The pipeline compares the Ordinal Ranking performance against a direct Log-PSD regression baseline, random untrained weights, and a shuffled-rank negative control.

---

## 6. Hyperparameters and Experimental Settings

The study standardizes all signal processing, architectural, and optimization parameters to ensure fair comparisons between the pretext tasks and backbones.

### Signal Processing & Time Instants
- **Sampling Rate ($f_s$)**: 250 Hz (Standard TUEV resampling rate).
- **Time Instants (Window Length)**: Each fundamental time window ($T$) spans **4.0 seconds**, which equates to exactly **1,000 samples** per channel.
- **Variant B Sequence Length**: For temporal drift ranking, the model processes a sequence of **4 consecutive windows** ($K=4$), representing 16 total seconds of continuous EEG context.
- **Montage**: 22 channels (Standard ACNS TCP derivation).
- **PSD Extraction**: Welch's method with 1.0-second segments (`nperseg=250`) and 50% overlap.

### Backbone Architecture Dimensions
All backbones are unified to maintain identical representational capacity.
- **Base Embedding Dimension (`embed_dim`)**: 256
- **Number of Encoder Layers**: 4
- **Dropout**: 0.1
- **Mamba (SSM) Specifics**: 
  - State Dimension (`d_state`): 64
  - Expansion Factor (`expand`): 2 (yielding an inner dimension of 512)
  - Convolution Kernel (`d_conv`): 4
- **Transformer Specifics**:
  - Attention Heads (`num_heads`): 8
  - Feed-Forward Dimension (`dim_feedforward`): 512

### Loss Formulation & Optimization
- **Tie-Margin ($\delta_{\text{margin}}$)**: $1 \times 10^{-3}$ (Used in Pairwise RankNet to gate noise).
- **Pretraining Epochs**: 30 Epochs.
- **Downstream Linear Probe Epochs**: 15 Epochs (Frozen Encoder).
- **Optimizer**: AdamW (`weight_decay=1e-2`).
- **Learning Rates**: Pretraining at $1 \times 10^{-4}$, Linear Probe at $1 \times 10^{-3}$.
