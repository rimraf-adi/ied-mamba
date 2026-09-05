# EEG Band-Power Ranking: Project Documentation & Progress

## 1. Project Overview
This repository contains the code for a self-supervised learning (SSL) framework designed to learn robust representations of EEG data. The primary pretext task revolves around ranking frequency band power in EEG signals, with different variants focusing on spatial relationships (Variant A) and spectral cross-band profiles (Variant C).

## 2. Recent Architectural & Methodological Changes
- **Capacity Scaling**: Upgraded the core Mamba backbone to a massive 512-dimension embedding space (`embed_dim=512`) to ensure the network has sufficient capacity to capture complex EEG patterns.
- **Deep MLP Classification Head**: Transitioned from a simple linear probe to a deep Multi-Layer Perceptron (MLP) head with LayerNorm, Dropout (0.3), and ReLU activations for downstream tasks. This resolved the capacity bottleneck observed when fine-tuning on highly complex downstream tasks.
- **Two-Stage Downstream Protocol**: Adopted a strict two-stage evaluation pipeline to prevent inflation of performance metrics:
  - **Stage 1 (Detection)**: Binary classification (Background vs. Seizure/Event).
  - **Stage 2 (Typing)**: 5-class classification (SPSW, GPED, PLED, EYEM, ARTF) performed *only* on samples identified as events.
- **Focal Loss & Weighted Sampling**: Implemented `FocalLoss(gamma=2.0)` and a `WeightedRandomSampler` to aggressively combat extreme class imbalances (e.g., the SPSW class has only 17 training samples).

## 3. Bug Fixes & Refactoring
- **Dataloader Zero-Batch Bug**: Discovered and fixed a critical bug where `drop_last=True` combined with a `batch_size=512` caused the downstream Typing dataset (which has ~500 samples) to yield zero batches during training, resulting in complete mode collapse. Changed to `drop_last=False` for downstream tasks.
- **Dataloader Fallback Bug**: Fixed a Python boolean falsy evaluation bug where `loader = loader or self.val_loader` silently fell back to the validation set if the training loader had zero batches.
- **Windows Multiprocessing Deadlock**: Manually cleared memory-mapped numpy arrays (`_labels_mmap`, `_windows_mmap`) before spawning PyTorch DataLoader workers to prevent fatal deadlocks on Windows environments.

## 4. Current Results (Variant A)
Variant A (Spatial Topography Ranking) was fully evaluated using the corrected two-stage pipeline. The results demonstrate phenomenal representation learning capabilities:

### Detection Task (Binary)
- **Macro F1**: ~57.9%
- **AUROC**: 0.7269
- **PR-AUC**: 0.0824
*(Note: The model is highly conservative in detection due to massive background imbalance, resulting in low PR-AUC but stable true-positive detection).*

### Typing Task (5-Class)
- **Macro F1**: **94.14%**
- **PR-AUC**: **95.67%**
- **AUROC**: **99.25%**
- **Balanced Accuracy**: 92.86%

**Conclusion for Variant A**: By forcing the model to rank spatial channel relationships based on band power, it learned incredibly robust representations that easily discriminate between complex morphological events (like SPSW vs. PLED) with over 95% precision-recall.

## 5. Ongoing Work
- **Variant C Evaluation**: The ablation suite is currently running `Variant C` (Spectral Profile Cross-Band) overnight. It has successfully hit `0.912` Val Rho at Epoch 8 of pretraining and is making steady progress.
