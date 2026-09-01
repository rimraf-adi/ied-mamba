# Ordinal PSD Band-Power Ranking Pretext Task for Self-Supervised EEG Pretraining on TUEV

## 1. Overview & Theoretical Motivation

In self-supervised EEG representation learning, spectral content is one of the most critical neurophysiological signals. However, standard methods that regress raw or log-scaled Power Spectral Density (PSD) suffer from two fundamental pitfalls:
1. **1/f Amplitude & Low-Frequency Dominance**: Low-frequency bands (Delta $0.5-4.0$ Hz, Theta $4.0-8.0$ Hz) inherently carry substantially greater power than high-frequency bands (Beta, Gamma). MSE / regression losses are overwhelmingly dominated by fitting low frequencies.
2. **Subject & Recording Scale Bias**: Absolute amplitude varies wildly across subjects, skull thicknesses, electrode impedances, and recording hardware. Regressing magnitude forces the encoder to memorize scale rather than underlying spectral *structure*.

### The Solution: Ordinal Band-Power Ranking
This repository implements **Ordinal Band-Power Ranking Pretraining**, which reframes spectral pretraining as learning **relative order (permutations)**:
- **Variant A (Spatial / Cross-Channel)**: For a given window $t$ and band $b$, rank the 22 EEG channels by power. Forces the encoder to capture spatial topography (e.g. occipital alpha dominance).
- **Variant B (Temporal / Cross-Time)**: For a given channel $c$ and band $b$, rank a sequence of time windows. Captures non-stationarity and clinical state transitions.
- **Variant C (Spectral Profile / Cross-Band)**: For a given window $t$ and channel $c$, rank the 5 canonical bands $(\delta, \theta, \alpha, \beta, \gamma)$. Captures the physiological shape of the power spectrum.

---

## 2. Directory Structure

```
band-power-ranking/
├── configs/
│   └── config.py                     # Centralized configuration dataclasses
├── src/
│   ├── dataset/
│   │   ├── augmentations.py          # Rank-preserving augmentations (Gaussian noise, time jitter, channel dropout)
│   │   ├── tuev_pretrain_dataset.py  # 4s windowing, Welch PSD caching, target batching
│   │   └── tuev_downstream_dataset.py# 6-class event classification dataset (SPSW, GPED, PLED, EYEM, ARTF, BCKG)
│   ├── spectral/
│   │   ├── psd_extractor.py          # Welch PSD & trapezoidal integration across canonical bands
│   │   └── rank_target_builder.py    # Target builder for Variants A, B, C with margin tie-handling
│   ├── models/
│   │   ├── backbone.py               # Mamba (SSM / Mamba-2) & Spatial-Temporal Transformer backbones
│   │   ├── ranking_heads.py          # Band-embedding conditioned ranking scoring heads
│   │   └── baselines.py              # Log-PSD regression baseline & Downstream classifier heads
│   ├── losses/
│   │   ├── ranking_losses.py         # Pairwise RankNet, ListNet, ListMLE, Soft-Spearman losses
│   │   └── metrics.py                # Spearman rho, Kendall tau, Pairwise accuracy, NDCG
│   └── training/
│       ├── pretrainer.py             # Pretraining loop with metric logging and negative controls
│       └── evaluator.py              # Linear Probe & Fine-Tuning evaluation runner
├── tests/
│   ├── test_spectral_and_ranking.py  # Unit tests on synthetic sinusoids (2Hz delta, 10Hz alpha, 20Hz beta)
│   ├── test_losses.py                # Unit tests for differentiable ranking losses
│   └── test_models.py                # Forward/backward gradient tests for Mamba & Transformer
├── experiments/
│   ├── run_pretraining.py            # CLI script for SSL pretraining
│   ├── run_downstream_eval.py        # CLI script for downstream TUEV 6-class evaluation
│   ├── run_ablation_study.py         # Automated ablation suite across Mamba & Transformer
│   └── generate_report.py            # Generates publication plots and markdown ablation report
└── README.md
```

---

## 3. Supported Backbones

- **Mamba / Structured State Space Models (`mamba_ssm` & SSD)**: Linear-time sequence modeling for long EEG multi-channel recordings.
- **Transformer**: Multi-Head Self-Attention with temporal token attention and cross-channel spatial attention.
- **ConvNet**: Multi-scale 1D ResNet baseline.

---

## 4. Quickstart Guide (using `uv`)

### 1. Run Unit Tests
```bash
uv run python -m unittest discover -s band-power-ranking/tests -p "test_*.py"
```

### 2. Run Self-Supervised Pretraining
- **Train Mamba with Multi-Variant Ordinal Ranking**:
  ```bash
  uv run python band-power-ranking/experiments/run_pretraining.py --model-type mamba --variants A+B+C --loss-type pairwise --epochs 15
  ```
- **Train Transformer with Multi-Variant Ordinal Ranking**:
  ```bash
  uv run python band-power-ranking/experiments/run_pretraining.py --model-type transformer --variants A+B+C --loss-type listnet --epochs 15
  ```
- **Train Log-PSD Direct Regression Baseline**:
  ```bash
  uv run python band-power-ranking/experiments/run_pretraining.py --model-type mamba --is-baseline-logpsd --epochs 15
  ```

### 3. Downstream 6-Class TUEV Evaluation
- **Linear Probe (Frozen Encoder)**:
  ```bash
  uv run python band-power-ranking/experiments/run_downstream_eval.py --model-type mamba --ckpt-path checkpoints/pretrain/pretrained_encoder.pt --mode linear_probe
  ```
- **Full Fine-Tuning**:
  ```bash
  uv run python band-power-ranking/experiments/run_downstream_eval.py --model-type mamba --ckpt-path checkpoints/pretrain/pretrained_encoder.pt --mode fine_tune
  ```

### 4. Run the Full Automated Ablation Suite
Executes ablations across Mamba and Transformer backbones, loss functions, tie margins, and baselines:
```bash
uv run python band-power-ranking/experiments/run_ablation_study.py
uv run python band-power-ranking/experiments/generate_report.py
```
