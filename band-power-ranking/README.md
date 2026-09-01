# Ordinal PSD Band-Power Ranking for Self-Supervised EEG Pretraining

This module implements a self-supervised learning (SSL) framework for EEG representation learning on the TUH EEG Event Corpus (TUEV). It replaces standard log-PSD magnitude regression with ordinal ranking tasks to mitigate $1/f$ power law dominance and subject-specific amplitude scaling biases.

## Pretext Task Formulations
The framework extracts Welch PSD across 5 canonical bands ($\delta, \theta, \alpha, \beta, \gamma$) and frames pretraining as an ordinal ranking problem across three orthogonal dimensions:
- **Variant A (Spatial / Cross-Channel)**: Rank 22 EEG channels for a specific frequency band at a single time step.
- **Variant B (Temporal / Cross-Time)**: Rank a sequence of non-overlapping time windows for a fixed channel and band.
- **Variant C (Spectral / Cross-Band)**: Rank the 5 frequency bands for a single channel in a single time window.

## Architectures Supported
- **Mamba**: Hardware-aware selective state space model (`mamba_ssm`), providing $O(N)$ linear time complexity.
- **Transformer**: Standard Multi-Head Self-Attention spatial-temporal encoder.

## Differentiable Ranking Losses
The module implements several differentiable surrogate functions for step-wise rank target prediction:
1. Pairwise RankNet (with noise-gating tie margins $\delta_{margin}$)
2. ListNet
3. ListMLE
4. Soft-Spearman Correlation

## Execution
```bash
# Pretraining
uv run python experiments/run_pretraining.py --model mamba --task ordinal_all

# Downstream Evaluation (Linear Probe or Fine-Tuning)
uv run python experiments/run_downstream_eval.py --model mamba --ckpt checkpoints/pretrain/encoder.pt

# Automated Ablation & Hyperparameter Sweep
uv run python experiments/run_ablation_study.py
```

## Referencing
Detailed mathematical derivations for the pretext variants and model layer dimensions can be found in `METHODOLOGY.md`.
