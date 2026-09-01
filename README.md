# IED-Mamba: EEG Event Corpus Analysis and Modeling

This repository contains the data processing pipelines, modeling frameworks, and self-supervised learning (SSL) experiments for the TUH EEG Event Corpus (TU-v2.0.1).

## Repository Structure

The repository is modularized into independent research sub-projects and shared tooling:

- `band-power-ranking/`: A self-supervised representation learning framework using ordinal multi-variant band-power ranking pretext tasks. Compares Mamba (SSM) and Transformer backbones.
- `dwt-study/`: Discrete Wavelet Transform (DWT) feature extraction and frequency-band correlation analysis module.
- `mamba_repo/`: Official `mamba_ssm` repository submodule (Mamba-2 integration).
- `scripts/`: Legacy parsing and exploratory data analysis scripts, including the Plotly Dash visualization interface.
- `docs/`: Data layout definitions, ACNS TCP montage reference, and environment setup guides.
- `TU-v2.0.1/`: Root directory for the raw EEG dataset (`.edf`, `.rec`, `.lab`, `.htk`).

## Setup

The project uses `uv` for dependency management. 
Environment: Python 3.11, PyTorch 2.5.1+cu121.

```bash
# Sync dependencies
uv sync

# Activate the virtual environment
.venv\Scripts\activate

# Run the core SSL ablation suite
cd band-power-ranking
uv run python experiments/run_ablation_study.py
```

## Documentation
- Refer to `docs/SETUP.md` for environment configuration.
- Refer to `docs/DATASET.md` for the TUH corpus annotation logic.
- Refer to `band-power-ranking/METHODOLOGY.md` for the ordinal ranking theory and mathematical formulation.
