# 🛠️ TUH EEG Event Corpus & Mamba SSM Setup Guide & Problem Resolution Log

This document provides a complete technical guide for setting up the **TUH EEG Event Corpus (TU-v2.0.1)** processing pipeline, the **Plotly Dash interactive signal visualizer**, and the official **`state-spaces/mamba` (`mamba_ssm`) CUDA State Space Model** using the **`uv`** package manager on Windows with NVIDIA GPU acceleration.

---

## 📌 Project Quick Reference

| Parameter | Project Setting / Requirement |
| :--- | :--- |
| **Package Manager** | `uv` (strictly used throughout) |
| **Python Version** | Python `3.11` (`.python-version` pinned to `3.11`) |
| **GPU Hardware** | NVIDIA RTX A5000 (24GB VRAM) |
| **CUDA Toolkit / Driver** | CUDA 12.3 / PyTorch CUDA 12.1 (`torch==2.5.1+cu121`) |
| **State Space Model** | Official `state-spaces/mamba` (`mamba_ssm` v2.3.2) & `causal-conv1d` (v1.6.2) |
| **Git Remote Origin** | `https://github.com/rimraf-adi/ied-mamba.git` (Branch: `main`) |

---

## ⚠️ Challenges & Technical Problems Encountered

During the initialization, environment setup, signal visualization, and CUDA compilation of `state-spaces/mamba`, several operating system, compiler, and package management challenges were encountered and resolved:

### 1. Python 3.13 Incompatibility with PyTorch CUDA & `mamba-ssm` C++ Extensions
- **Problem**: `uv venv` initially selected the system-default CPython 3.13. However, PyTorch CUDA index wheels (`+cu121`) and CUDA C++ extensions (`causal-conv1d`, `mamba-ssm`) only publish pre-compiled wheels and C++ compilation targets for Python `3.9` through `3.12`.
- **Root Cause**: Python 3.13 ABI (`cp313`) is not yet supported by official CUDA C++ extensions.
- **Solution**:
  - Updated `pyproject.toml` with `requires-python = ">=3.11,<3.13"`.
  - Created `.python-version` file pinned to `3.11`.
  - Re-created virtual environment using `uv venv --clear --python 3.11`.

---

### 2. Windows MSVC C++ Compiler (`cl.exe`) & `nvcc` Environment Setup
- **Problem**: Attempting to install `mamba-ssm` and `causal-conv1d` via standard `pip install` resulted in compilation failures:
  `UserWarning: It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set`
  and `NameError: name 'bare_metal_version' is not defined`.
- **Root Cause**: On Windows, building PyTorch C++/CUDA extensions requires activating Visual Studio 2022 MSVC compiler environment (`cl.exe`) and setting explicit `setutils` SDK flags.
- **Solution**:
  - Activated MSVC environment via `vcvars64.bat`:
    `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`
  - Configured build flags prior to compilation:
    ```cmd
    set DISTUTILS_USE_SDK=1
    set CAUSAL_CONV1D_FORCE_BUILD=TRUE
    set MAMBA_FORCE_BUILD=TRUE
    ```
  - Successfully compiled both `causal-conv1d==1.6.2.post1` and `mamba_ssm==2.3.2.post1` directly from source.

---

### 3. Missing `triton` Dependency on Windows
- **Problem**: Importing `mamba_ssm` raised `ModuleNotFoundError: No module named 'triton'`.
- **Root Cause**: Official `mamba_ssm` relies on OpenAI Triton for fused layer normalization and selective scan operations. Standard `triton` on PyPI is Linux-only.
- **Solution**: Installed Windows-native Triton wheel `triton-windows==3.7.1.post27` via `uv pip install triton-windows`.

---

### 4. Stale Background Server Processes on Port 8050
- **Problem**: Dashboard updates and Light Theme styling were not rendering in the browser despite code changes; `http://127.0.0.1:8050/` continued serving cached dark layouts.
- **Root Cause**: Multiple background Python processes (`PIDs 6808, 47572, 40540, 43160`) spawned in earlier turns remained bound to port `8050`.
- **Solution**:
  - Implemented mandatory process termination before starting server instances:
    `taskkill /F /IM python.exe`
  - Updated process lifecycle management to prevent process appending.

---

### 5. Windows Console Unicode Encoding Crashes (`UnicodeEncodeError`)
- **Problem**: Scripts crashed on Windows console with:
  `UnicodeEncodeError: 'charmap' codec can't encode character '\u26a1'` when printing emojis (🧠, ⚡, ✅) or unicode progress indicators.
- **Root Cause**: Windows PowerShell stdout defaults to `cp1252` encoding instead of `utf-8`.
- **Solution**: Configured UTF-8 stdout reconfiguration across all entry points:
  ```python
  import sys
  try:
      sys.stdout.reconfigure(encoding='utf-8')
  except Exception:
      pass
  ```

---

### 6. Dash 2.17+ API Deprecation & Canvas Rendering Bottlenecks
- **Problem**: `app.run_server(...)` raised `ObsoleteAttributeException` in Dash 2.17+; rendering 22-channel multi-second SVG traces caused severe UI lag when dragging sliders.
- **Root Cause**: Dash replaced `app.run_server` with `app.run`; SVG `<path>` DOM nodes overload browser layout engines.
- **Solution**:
  - Replaced `app.run_server` with `app.run(debug=False, port=8050)`.
  - Replaced SVG `go.Scatter` with WebGL GPU hardware acceleration (`go.Scattergl`).
  - Implemented `PRELOADED_FILE_CACHE` RAM dictionary and background preloader thread for sub-millisecond 60 FPS slider transitions.

---

## 🚀 Environment Setup & Reproduction Steps

Follow these steps to set up the environment from scratch:

### 1. Initialize Python 3.11 Virtual Environment with `uv`
```bash
# Pin Python version
echo 3.11 > .python-version

# Create virtual environment
uv venv --clear --python 3.11
```

### 2. Install PyTorch CUDA 12.1 & Core Dependencies
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install triton-windows mambapy dash mne pyedflib numpy scipy pandas matplotlib seaborn scikit-learn tqdm
```

### 3. Build & Install Official `state-spaces/mamba` (If building C++ extensions)
```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
set CAUSAL_CONV1D_FORCE_BUILD=TRUE
set MAMBA_FORCE_BUILD=TRUE
uv pip install --no-build-isolation causal-conv1d mamba-ssm
```

---

## 🧪 Verification Commands

### 1. Test Official Mamba SSM CUDA Acceleration
```bash
uv run python test_mamba_cuda.py
```
*Expected Output:*
```text
🚀 Official state-spaces/mamba (mamba_ssm v2.3.2) GPU Verification Test
CUDA Available:       True
GPU Device:           NVIDIA RTX A5000
causal_conv1d ver:    1.6.2.post1
mamba_ssm ver:        2.3.2.post1
SUCCESS: Official state-spaces/mamba (mamba_ssm Mamba2) is functional on CUDA!
```

### 2. Run Interactive Signal Visualizer Dashboard
```bash
uv run python dashboard.py
```
Access at **`http://127.0.0.1:8050/`** in your browser.
