# 🧠 IED Analysis & TUH EEG Event Corpus Processing Pipeline

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed_by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-orange.svg)](https://pytorch.org/)
[![Plotly Dash](https://img.shields.io/badge/Plotly_Dash-4.4.1-brightgreen.svg)](https://dash.plotly.com/)

A modular Python software suite and interactive **Plotly Dash** web dashboard for parsing, preprocessing, feature extraction, analyzing, and PyTorch deep-learning model integration on the **TUH EEG Event Corpus** (`TU-v2.0.1` / Interictal Epileptiform Discharge dataset).

---

## ✨ Features

- **Multi-Format Parsers ([dataset_loader.py](dataset_loader.py))**:
  - European Data Format (`.edf`) raw EEG signal loader (MNE Python / PyEDFLib with fallback binary reader).
  - Second-precision recording annotation parser (`.rec`).
  - 10-microsecond channel annotation parser (`.lab`).
  - HTK differential energy feature matrix parser (`.htk`).

- **Standard ACNS TCP Montage Builder**:
  - Automatically constructs all 22 differential bipolar channels (`FP1-F7`, `F7-T3`, ..., `P4-O2`) from raw scalp reference electrodes.

- **Preprocessing & Normalization ([preprocessing.py](preprocessing.py))**:
  - Zero-phase Butterworth Bandpass filter (0.5 – 50.0 Hz).
  - Powerline IIR Notch filter (60.0 Hz noise rejection).
  - Signal Normalization: Z-score (`zscore`), Robust IQR (`robust`), and Min-Max (`minmax`).
  - Sliding-window segmentation with overlap ratio and event label rasterization.
  - Feature calculations: EEG frequency band powers (`Delta`, `Theta`, `Alpha`, `Beta`, `Gamma`) and time-domain metrics (RMS, Variance, Hjorth parameters).

- **PyTorch Integration**:
  - Custom `EEGEventDataset` and `DataLoader` for training deep learning models (e.g., Mamba, CNN, Transformers).

- **Exploratory Data Analysis ([exploratory_analysis.py](exploratory_analysis.py))**:
  - Aggregates over 113,000 event annotations across `train` and `eval` splits (`spsw`, `gped`, `pled`, `eyem`, `artf`, `bckg`).
  - Generates distribution bar charts, duration metrics, 22-channel spatial heatmaps, and spectral PSD plots.

- **Interactive Plotly Dash Dashboard ([dashboard.py](dashboard.py))**:
  - Modern high-contrast Light Theme interface accessible at `http://127.0.0.1:8050/`.
  - Multi-channel stacked signal visualizer with zoom/pan and event overlays.
  - Interactive HTK feature heatmap inspector.
  - Side-by-side `.rec` vs `.lab` annotation inspector.
  - Live PyTorch `DataLoader` batch shape & class balance simulator.

---

## 🚀 Quick Start (Managed with `uv`)

### 1. Installation & Environment Setup
```bash
# Clone repository
git clone https://github.com/rimraf-adi/ied-mamba.git
cd ied-mamba

# Create virtual environment & install dependencies using uv
uv venv .venv
uv add numpy scipy pandas mne pyedflib torch matplotlib seaborn scikit-learn tqdm plotly dash
```

### 2. Run Pipeline & Generate Analysis Reports
```bash
uv run python run_pipeline.py
```

### 3. Launch Interactive Plotly Dash Dashboard
```bash
uv run python dashboard.py
```
Open your web browser at **`http://127.0.0.1:8050/`**

---

## 📁 Repository Architecture

```text
├── dataset_loader.py       # EDF, REC, LAB, HTK parsers & PyTorch EEGEventDataset
├── preprocessing.py        # Signal filtering, TCP montage, sliding windows, band powers
├── exploratory_analysis.py # Corpus scanning, heatmaps, PSD plots, trace renderer
├── dashboard.py            # Plotly Dash web dashboard (http://127.0.0.1:8050/)
├── run_pipeline.py         # End-to-end pipeline runner & test verification
├── pyproject.toml          # Project configuration & dependencies (uv)
├── uv.lock                 # Lockfile for reproducible builds
└── README.md               # Project documentation
```

---

## 📜 License
MIT License.
