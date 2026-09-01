# 📖 TUH EEG Event Corpus (TU-v2.0.1) & IED Analysis Help Guide

This comprehensive guide provides detailed explanations of all clinical terminology, dataset file formats, signal processing metrics, channel montages, and statistical parameters used throughout this project and the interactive Plotly Dash dashboard.

---

## 📌 1. Clinical Event Classes & Terminology

The **TUH EEG Event Corpus** contains annotated clinical EEG recordings categorized into 6 primary event classes:

| Event Code | Class Name | Full Clinical Description | Medical Significance |
| :--- | :--- | :--- | :--- |
| **`spsw`** (1) | **Spike and Slow Wave** | Sharp transient discharges (<70 ms) followed by a broad slow wave (200–500 ms). | Classic hallmark of focal or generalized epileptiform activity (interictal epileptiform discharges - IEDs). |
| **`gped`** (2) | **Generalized Periodic Epileptiform Discharge** | Periodic waveforms appearing synchronously over both brain hemispheres. | Associated with acute encephalopathy, status epilepticus, severe metabolic dysfunction, or anoxia. |
| **`pled`** (3) | **Periodic Lateralized Epileptiform Discharge** | Repetitive sharp or spike wave complexes localized to one brain hemisphere. | Signifies focal destructive brain lesions, acute cerebral infarction, or herpes simplex encephalitis. |
| **`eyem`** (4) | **Eye Movement** | Low-frequency (<4 Hz) high-amplitude frontal artifacts caused by ocular dipoles. | Non-epileptic biological artifact resulting from eye blinks or lateral eye movements. |
| **`artf`** (5) | **Artifact** | High-frequency or high-amplitude electrical noise (muscle contraction, pop, electrode movement). | Non-cerebral signal interference originating from movement, cardiac activity (EKG), or cable noise. |
| **`bckg`** (6) | **Background** | Normal baseline cerebral rhythm (Alpha, Beta, Theta, Delta waves). | Normal continuous background activity without transient events or paroxysmal discharges. |

---

## 📁 2. File Formats & Data Structure

```text
TU-v2.0.1/
└── edf/
    ├── train/   # 359 training session files (cross-referenced to patient index)
    └── eval/    # 159 disjoint evaluation session files
```

Each recording session directory contains four key file types:

1. **`.edf` (European Data Format)**:
   - Contains raw, continuous multi-channel scalp electroencephalogram (EEG) signals sampled at `fs` (typically 250 Hz or 400 Hz).
   - Stores physical calibration gains, digital min/max, physical min/max, and transducer channel labels.

2. **`.rec` (Recording-Level Event Annotations)**:
   - CSV-like text format with second-level timestamps for global recording annotations.
   - Format: `channel_index, start_seconds, stop_seconds, event_code`
   - Example: `19, 28.6, 29.6, 6` (Channel 19, 28.6s to 29.6s, Background).

3. **`.lab` (Channel-Level Event Annotations)**:
   - High-resolution annotation file per channel with 10-microsecond precision timestamps ($1 \times 10^{-5}$ seconds).
   - Format: `start_10us  stop_10us  label_name`
   - Example: `15760000  15860000  artf` ($157.60$s to $158.60$s, Artifact).

4. **`.htk` (HTK Feature Files)**:
   - Hidden Markov Model Toolkit (HTK) binary feature files containing pre-extracted differential energy features per channel.
   - Includes 12-byte binary header followed by 32-bit float feature matrices.

---

## ⚡ 3. ACNS TCP Montage Standard (22 Bipolar Channels)

Clinical EEGs record signals against a reference electrode (e.g. `REF` or `LE`). To isolate localized electrical activity, channels are transformed into the **American Clinical Neurophysiology Society (ACNS) Temporal Central Parasagittal (TCP) Montage** of 22 differential channel pairs:

$$\text{Montage Channel Signal} = V_{\text{Electrode}_1} - V_{\text{Electrode}_2}$$

### Standard 22 Differential Channel Pairs:
- **Left Temporal Chain**: `FP1-F7`, `F7-T3`, `T3-T5`, `T5-O1`
- **Right Temporal Chain**: `FP2-F8`, `F8-T4`, `T4-T6`, `T6-O2`
- **Parasagittal & Center Chain**: `A1-T3`, `T3-C3`, `C3-CZ`, `CZ-C4`, `C4-T4`, `T4-A2`
- **Left Parasagittal Chain**: `FP1-F3`, `F3-C3`, `C3-P3`, `P3-O1`
- **Right Parasagittal Chain**: `FP2-F4`, `F4-C4`, `C4-P4`, `P4-O2`

---

## 📊 4. Signal Processing Metrics & Formulas

### 1. EEG Frequency Bands & Power Spectral Density (PSD)
Computes signal energy across standard physiological frequency spectrum via Welch's Periodogram:
- **Delta ($\delta$)**: 0.5 – 4.0 Hz (Deep sleep, slow-wave cerebral activity).
- **Theta ($\theta$)**: 4.0 – 8.0 Hz (Drowsiness, focal slowing).
- **Alpha ($\alpha$)**: 8.0 – 12.0 Hz (Relaxed wakeful posterior rhythm).
- **Beta ($\beta$)**: 12.0 – 30.0 Hz (Active cognitive state, medication effect).
- **Gamma ($\gamma$)**: 30.0 – 50.0 Hz (High-frequency cognitive binding).

Absolute band power is calculated via trapezoidal integration of PSD $S(f)$:

$$P_{\text{band}} = \int_{f_{\text{min}}}^{f_{\text{max}}} S(f) \, df$$

### 2. Hjorth Parameters
Time-domain statistical descriptors for signal complexity:
- **Activity**: Measure of total signal power / variance ($\sigma^2_x$).
$$\text{Activity} = \operatorname{Var}(x(t))$$

- **Mobility**: Mean frequency estimate of the signal spectrum.
$$\text{Mobility} = \sqrt{\frac{\operatorname{Var}\left(\frac{dx}{dt}\right)}{\operatorname{Var}(x(t))}}$$

- **Complexity**: Measure of frequency deviation / change in signal shape.
$$\text{Complexity} = \frac{\text{Mobility}\left(\frac{dx}{dt}\right)}{\text{Mobility}(x(t))}$$

### 3. Normalization Techniques
- **Z-Score Normalization**: Zero-mean, unit-variance transformation:
$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

- **Robust Scaling**: Median and Interquartile Range (IQR) scaling, highly resilient to extreme noise spikes:
$$x_{\text{robust}} = \frac{x - \operatorname{Median}(x)}{\operatorname{IQR}(x)}$$

---

## 🎛️ 5. Dashboard Controls & Performance Tips

1. **`updatemode='mouseup'`**: Sliders update graphics only when the handle is released, eliminating drag latency.
2. **LRU Caching**: Previously loaded EDF files and annotations are cached in RAM for instant sub-millisecond updates.
3. **PyTorch Loader Config**: Live preview of output tensor shapes `(Batch, Channels, WindowSamples)` and extracted window count.
