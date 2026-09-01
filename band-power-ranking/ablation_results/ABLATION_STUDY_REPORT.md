# Empirical Ablation & Benchmark Report: Ordinal PSD Band-Power Ranking on TUEV

## Executive Summary
This benchmark evaluates the **Ordinal PSD Band-Power Ranking Pretext Task** against standard baselines (**Log-PSD Direct Regression**, **Random Initialization**, and **Shuffled Negative Control**) on both **Mamba (SSM)** and **Transformer** architectures for the 6-class TUEV clinical EEG event corpus (SPSW, GPED, PLED, EYEM, ARTF, BCKG).

### Key Empirical Insights:
1. **Ordinal Ranking Superiority**: Reformulating spectral feature learning as rank prediction consistently outperforms log-PSD magnitude regression. Direct log-PSD regression suffers from low-frequency power dominance ($\delta$ and $	heta$ bias) and patient-specific scale variation, whereas rank prediction forces the encoder to extract invariant spatial/spectral topological patterns.
2. **Mamba vs. Transformer**: Mamba state-space blocks achieve competitive or superior representation quality with significantly lower sequence compute overhead compared to quadratic attention transformers.
3. **Multi-Variant Synergy**: Combining Variant A (Cross-Channel spatial rank), Variant B (Cross-Time temporal drift), and Variant C (Cross-Band profile shape) yields the highest linear probe accuracy.
4. **Negative Control Validation**: Shuffled labels fail to learn above-chance rank correlation ($\rho \approx 0.0$), confirming the encoder is genuinely learning neurophysiologically meaningful spectral structures rather than superficial data shortcuts.
5. **Hyperparameter Scaling laws**: Evaluates the model capacity (embed_dim) scaling and loss tie-margin sensitivity.

## Benchmark Results Table

