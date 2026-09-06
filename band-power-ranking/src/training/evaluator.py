"""
Downstream Evaluation Protocol for TUEV 6-Class EEG Event Classification.

Computes:
- Balanced Accuracy
- Macro & Weighted F1-scores
- Class-wise AUROC & AUPRC
- Multi-class Confusion Matrix
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_tensor = self.alpha.to(targets.device)
            at = alpha_tensor.gather(0, targets)
            focal_loss = focal_loss * at
        return focal_loss.mean()
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from typing import Dict, Any, Optional, List, Tuple

from ..models.baselines import DownstreamClassificationModel
from ..dataset.tuev_downstream_dataset import CLASS_NAMES


class TUEVEvaluator:
    """
    Evaluation runner for Linear Probe and Full Fine-Tuning on TUEV.
    """

    def __init__(
        self,
        backbone: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        embed_dim: int = 128,
        num_classes: int = 6,
        mode: str = "linear_probe",  # 'linear_probe' or 'fine_tune'
        num_epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        patience: int = 10,
        device: Optional[torch.device] = None,
        save_dir: str = "checkpoints/downstream"
    ):
        self.mode = mode
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        freeze = (mode == "linear_probe")
        self.classifier_model = DownstreamClassificationModel(
            backbone=backbone,
            embed_dim=embed_dim,
            num_classes=num_classes,
            freeze_backbone=freeze,
            use_mlp_head=True
        ).to(self.device)

        # Compute Inverse Class Frequencies for Alpha Weighting
        # IMPORTANT: Must remap labels the same way __getitem__ does, so alpha matches the actual classes.
        try:
            raw_labels = np.load(self.train_loader.dataset.mmap_label_path, mmap_mode='r')
            # Apply the same remapping as __getitem__
            task_mode = getattr(self.train_loader.dataset, 'task_mode', 'joint')
            if task_mode == "detection":
                remapped = (raw_labels > 0).astype(np.int64)
            elif task_mode == "typing":
                valid_mask = raw_labels > 0
                remapped = raw_labels[valid_mask] - 1
            else:
                remapped = raw_labels
            class_counts = np.bincount(remapped, minlength=num_classes)
            class_weights = 1.0 / (class_counts + 1e-8)
            alpha = torch.tensor(class_weights, dtype=torch.float32)
            alpha = alpha / alpha.sum()
            print(f"  [Alpha Weights] Class counts: {class_counts.tolist()} | Alpha: {[f'{a:.4f}' for a in alpha.tolist()]}")
        except Exception as e:
            print(f"  [Alpha Weights] Failed to compute: {e}")
            alpha = None

        # Use Focal Loss for heavy class imbalance
        self.criterion = FocalLoss(gamma=2.0, alpha=alpha)

        # Bug #3 Fix: Discriminative Learning Rates
        # Backbone LR = lr/50 to prevent catastrophic forgetting of pretrained representations.
        # MLP Head LR = lr (full speed, randomly initialized).
        backbone_params = [p for p in self.classifier_model.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.classifier_model.classifier.parameters() if p.requires_grad]
        
        if freeze:
            # Linear probe: only head params, single LR
            self.optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=weight_decay)
        else:
            # Fine-tune: discriminative LR
            param_groups = [
                {'params': backbone_params, 'lr': lr / 50},  # 2e-5 by default
                {'params': head_params, 'lr': lr}             # 1e-3
            ]
            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

    def train_epoch(self) -> float:
        self.classifier_model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.classifier_model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        self.classifier_model.eval()
        if loader is None:
            loader = self.val_loader
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.classifier_model(x)
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        # Optimal Threshold Sweep for Binary Classification
        threshold_sweep_data = {}
        if self.num_classes == 2:
            y_prob_positive = y_prob[:, 1]
            best_f1 = 0.0
            best_thresh = 0.5
            best_preds = y_pred
            for thresh in np.linspace(0.40, 0.60, 21):
                thresh_preds = (y_prob_positive >= thresh).astype(int)
                from sklearn.metrics import precision_score, recall_score
                t_prec = precision_score(y_true, thresh_preds, zero_division=0)
                t_rec = recall_score(y_true, thresh_preds, zero_division=0)
                t_f1 = f1_score(y_true, thresh_preds, average='macro', zero_division=0)
                
                threshold_sweep_data[f"{thresh:.2f}"] = {
                    "precision": float(t_prec),
                    "recall": float(t_rec),
                    "macro_f1": float(t_f1)
                }
                
                if t_f1 > best_f1:
                    best_f1 = t_f1
                    best_thresh = thresh
                    best_preds = thresh_preds
            y_pred = best_preds
            print(f"  [Threshold Sweep] Optimal Binary Threshold: {best_thresh:.2f}")

        # Compute metrics
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))

        # One-hot encode targets for AUROC / AUPRC
        present_classes = np.unique(y_true)
        auroc_macro = 0.5
        auprc_macro = 0.0

        if len(present_classes) > 1:
            try:
                if self.num_classes == 2:
                    y_prob_positive = y_prob[:, 1]
                    auroc_macro = roc_auc_score(y_true, y_prob_positive)
                    auprc_macro = average_precision_score(y_true, y_prob_positive)
                else:
                    y_true_oh = np.eye(self.num_classes)[y_true]
                    auroc_macro = roc_auc_score(y_true_oh, y_prob, multi_class='ovr', average='macro')
                    auprc_macro = average_precision_score(y_true_oh, y_prob, average='macro')
            except Exception:
                pass

        return {
            'balanced_accuracy': float(bal_acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'auroc': float(auroc_macro),
            'auprc': float(auprc_macro),
            'confusion_matrix': conf_mat.tolist(),
            'present_classes': present_classes.tolist(),
            'threshold_sweep': threshold_sweep_data
        }

    def run(self) -> Dict[str, Any]:
        print(f"--- Starting Downstream {self.mode.upper()} Evaluation ({self.num_epochs} epochs) ---")
        best_val_f1 = -1.0
        best_metrics = {}
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch()
            self.scheduler.step()

            # Evaluate every epoch for accurate patience tracking
            val_metrics = self.evaluate()
            f1 = val_metrics['macro_f1']
            b_acc = val_metrics['balanced_accuracy']

            print(
                f"[{self.mode} Epoch {epoch:02d}/{self.num_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Bal Acc: {b_acc*100:.2f}% | "
                f"Macro F1: {f1:.4f} | "
                f"PR-AUC: {val_metrics['auprc']:.4f} | "
                f"AUROC: {val_metrics['auroc']:.4f}"
            )

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_metrics = val_metrics
                patience_counter = 0
                
                # Also record train metrics at the best validation checkpoint
                train_metrics = self.evaluate(loader=self.train_loader)
                best_metrics['train_macro_f1'] = train_metrics['macro_f1']
                best_metrics['train_auroc'] = train_metrics['auroc']
                best_metrics['train_confusion_matrix'] = train_metrics['confusion_matrix']
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best Val Macro F1: {best_val_f1:.4f} | Best Train Macro F1: {best_metrics.get('train_macro_f1', 0.0):.4f}")
                    break

        return best_metrics
