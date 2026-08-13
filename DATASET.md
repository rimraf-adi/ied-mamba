# 🧠 TUH EEG Event Corpus (TU-v2.0.1) Dataset Specification & Statistics

This document provides a comprehensive technical overview of the **Temple University Hospital (TUH) EEG Event Corpus (`TU-v2.0.1`)**, including dataset structure, file formats, class distribution statistics, clinical event explanations, and ACNS montage signal processing specifications.

---

## 📌 1. Dataset Overview & Scope

- **Corpus Name**: TUH EEG Event Corpus
- **Version**: `TU-v2.0.1`
- **Location on Disk**: `d:\ied\TU-v2.0.1`
- **Primary Focus**: Interictal Epileptiform Discharge (IED) detection, periodic discharge localization, and biological/non-biological artifact separation across multi-channel clinical scalp EEG recordings.

---

## 📊 2. Global Corpus Statistics

Across **518 total clinical session directories** divided into disjoint **train** and **eval** subsets, **113,353 total event annotations** were parsed and analyzed:

### Subset Split Breakdown:
- **Training Set (`train/`)**: **359 patient session files**
- **Evaluation Set (`eval/`)**: **159 patient session files**

### Event Class Distribution Table:

| Event Code | Internal ID | Clinical Class Name | Total Event Count | Percentage | Average Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`bckg`** | `6` | **Baseline Background** | **85,630** | **75.54%** | $12.45 \text{ seconds}$ |
| **`spsw`** | `1` | **Spike and Slow Wave** | **11,885** | **10.48%** | $1.82 \text{ seconds}$ |
| **`artf`** | `5` | **Noise Artifact** | **10,757** | **9.49%** | $2.15 \text{ seconds}$ |
| **`eyem`** | `4` | **Eye Movement** | **3,892** | **3.43%** | $1.45 \text{ seconds}$ |
| **`gped`** | `2` | **Generalized Periodic Discharge** | **802** | **0.71%** | $4.62 \text{ seconds}$ |
| **`pled`** | `3` | **Periodic Lateralized Discharge**| **387** | **0.34%** | $5.10 \text{ seconds}$ |
| **TOTAL** | — | — | **113,353** | **100.00%** | — |

---

## 🔬 3. Detailed Clinical Event Explanations

### 1. `spsw` (Spike and Slow Wave)
- **Morphology**: A sharp transient voltage deflection lasting 20–70 ms followed by a broad slow wave lasting 200–500 ms.
- **Clinical Significance**: Hallmark of focal or generalized **Interictal Epileptiform Discharges (IEDs)** and active seizure foci.

### 2. `gped` (Generalized Periodic Discharge)
- **Morphology**: Repetitive, periodic waveforms appearing synchronously over both cerebral hemispheres with uniform morphology.
- **Clinical Significance**: Associated with severe acute encephalopathy, status epilepticus, metabolic dysfunction, or cerebral anoxia.

### 3. `pled` (Periodic Lateralized Discharge)
- **Morphology**: Periodic sharp wave or spike-wave complexes restricted to one brain hemisphere.
- **Clinical Significance**: Indicates acute focal brain injury, cerebral infarction (stroke), viral encephalitis, or mass lesions.

### 4. `eyem` (Eye Movement)
- **Morphology**: Low-frequency (<4 Hz), high-amplitude frontal signal deflections created by the corneo-retinal ocular dipole during eye blinks or lateral eye movement.
- **Clinical Significance**: Non-epileptic biological artifact.

### 5. `artf` (Artifact)
- **Morphology**: High-frequency muscle contraction (EMG) noise, sharp single-sample electrode pops, or low-frequency cable sway.
- **Clinical Significance**: Non-cerebral electrical interference originating outside the brain.

### 6. `bckg` (Background)
- **Morphology**: Normal continuous baseline EEG rhythms (Alpha 8-12Hz, Beta 12-30Hz, Theta 4-8Hz, Delta 0.5-4Hz).
- **Clinical Significance**: Baseline brain electrical activity in the absence of paroxysmal discharges or transient artifacts.

---

## 📁 4. File Formats & Technical Specifications

Each session directory contains four corresponding binary/text files sharing a common base path:

```text
TU-v2.0.1/edf/eval/01_tcp_ar/002/00000254/s005_2010_11_15/
├── 00000254_s005_t000.edf   # Continuous Multi-Channel EEG Signals
├── 00000254_s005_t000.rec   # Second-Precision Recording Annotations
├── 00000254_s005_t000_ch0.lab # 10-Microsecond Channel Annotations
└── 00000254_s005_t000_ch0.htk # Binary HTK Differential Energy Features
```

### 1. `.edf` (European Data Format)
- Contains continuous multi-channel scalp EEG signals sampled at frequency $f_s$ (250 Hz or 400 Hz).
- Stores channel labels, transducer types, digital min/max, and physical gain scaling parameters.

### 2. `.rec` (Recording Annotations)
- Text format storing interval annotations across channels.
- Format: `channel_idx, start_seconds, stop_seconds, class_code`
- Code mapping: `1: spsw`, `2: gped`, `3: pled`, `4: eyem`, `5: artf`, `6: bckg`.

### 3. `.lab` (Channel Annotations)
- High-precision text files storing timestamps with $10 \mu s$ ($1 \times 10^{-5}\text{ s}$) resolution per channel.
- Format: `start_10us  stop_10us  class_label`

### 4. `.htk` (HTK Feature Matrices)
- Binary Hidden Markov Model Toolkit (HTK) feature files containing differential energy feature vectors.
- Header: 12-byte binary header (`nSamples`, `samplePeriod`, `sampleSize`, `parmKind`) followed by float32 feature matrices.

---

## ⚡ 5. ACNS TCP Montage Standard (22 Channels)

To eliminate reference electrode noise, raw scalp channels are transformed into the **American Clinical Neurophysiology Society (ACNS) Temporal Central Parasagittal (TCP) Montage** of 22 differential channel pairs:

$$V_{\text{Montage Channel}} = V_{\text{Electrode}_1} - V_{\text{Electrode}_2}$$

### Standard 22 Differential Channel Pairs:

1. `FP1-F7` (Left Frontal-Temporal)
2. `F7-T3` (Left Anterior-Mid Temporal)
3. `T3-T5` (Left Mid-Posterior Temporal)
4. `T5-O1` (Left Posterior-Occipital)
5. `FP2-F8` (Right Frontal-Temporal)
6. `F8-T4` (Right Anterior-Mid Temporal)
7. `T4-T6` (Right Mid-Posterior Temporal)
8. `T6-O2` (Right Posterior-Occipital)
9. `A1-T3` (Left Ear-Temporal)
10. `T3-C3` (Left Mid Temporal-Central)
11. `C3-CZ` (Left Central-Midline)
12. `CZ-C4` (Midline-Right Central)
13. `C4-T4` (Right Central-Mid Temporal)
14. `T4-A2` (Right Mid Temporal-Ear)
15. `FP1-F3` (Left Frontal-Parasagittal)
16. `F3-C3` (Left Mid Frontal-Central)
17. `C3-P3` (Left Central-Parietal)
18. `P3-O1` (Left Parietal-Occipital)
19. `FP2-F4` (Right Frontal-Parasagittal)
20. `F4-C4` (Right Mid Frontal-Central)
21. `C4-P4` (Right Central-Parietal)
22. `P4-O2` (Right Parietal-Occipital)
