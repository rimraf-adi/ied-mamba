"""
Self-Supervised Pretraining Engine for Ordinal Ranking and Log-PSD Baseline on EEG.
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any, List

from ..losses.ranking_losses import MultiVariantRankingLoss
from ..losses.metrics import compute_ranking_metrics_dict
from ..models.ranking_heads import MultiVariantRankingModel
from ..models.baselines import LogPSDRegressionModel


class EEGPretrainer:
    """
    Pretraining Engine managing training loops, multi-task ranking losses, metric tracking, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_type: str = "pairwise",
        variant_a_weight: float = 1.0,
        variant_b_weight: float = 1.0,
        variant_c_weight: float = 1.0,
        tie_margin: float = 1e-3,
        lr: float = 5e-4,
        weight_decay: float = 1e-2,
        num_epochs: int = 15,
        warmup_epochs: int = 2,
        grad_clip_norm: float = 1.0,
        device: Optional[torch.device] = None,
        is_log_psd_baseline: bool = False,
        shuffle_negative_control: bool = False,
        save_dir: str = "checkpoints/pretrain"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_log_psd_baseline = is_log_psd_baseline
        self.shuffle_negative_control = shuffle_negative_control
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.grad_clip_norm = grad_clip_norm
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model.to(self.device)

        if not self.is_log_psd_baseline:
            self.ranking_loss_fn = MultiVariantRankingLoss(
                loss_type=loss_type,
                w_a=variant_a_weight,
                w_b=variant_b_weight,
                w_c=variant_c_weight,
                tie_margin=tie_margin
            ).to(self.device)
        else:
            self.regression_loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, num_epochs - warmup_epochs),
            eta_min=1e-6
        )

        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_spearman_a': [],
            'val_spearman_c': [],
            'epoch_times': []
        }

    def _shuffle_targets(self, targ_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Negative control: randomly permutes ground-truth ranks to verify learning is not shortcut-based.
        """
        out = {}
        for k, v in targ_dict.items():
            if k in ('ranks', 'order'):
                idx = torch.randperm(v.shape[-1])
                out[k] = v[..., idx]
            elif k in ('pairwise_y', 'pairwise_mask'):
                idx = torch.randperm(v.shape[-1])
                out[k] = v[..., idx, :][..., :, idx]
            else:
                out[k] = v
        return out

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        epoch_metrics: Dict[str, float] = {}

        for batch in self.train_loader:
            x_win = batch['window'].to(self.device)
            seq_win = batch['seq_windows'].to(self.device) if 'seq_windows' in batch else None

            self.optimizer.zero_grad()

            if self.is_log_psd_baseline:
                target_log_psd = batch['log_psd'].to(self.device)
                pred_log_psd, _ = self.model(x_win)
                loss = self.regression_loss_fn(pred_log_psd, target_log_psd)
            else:
                targets_a = {k: v.to(self.device) for k, v in batch['targets_a'].items()}
                targets_b = {k: v.to(self.device) for k, v in batch['targets_b'].items()}
                targets_c = {k: v.to(self.device) for k, v in batch['targets_c'].items()}

                if self.shuffle_negative_control:
                    targets_a = self._shuffle_targets(targets_a)
                    targets_b = self._shuffle_targets(targets_b)
                    targets_c = self._shuffle_targets(targets_c)

                out = self.model(x_win, seq_win)
                loss, step_metrics = self.ranking_loss_fn(
                    preds_a=out['scores_a'],
                    targets_a=targets_a,
                    preds_b=out['scores_b'],
                    targets_b=targets_b,
                    preds_c=out['scores_c'],
                    targets_c=targets_c
                )
                for k, v in step_metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        res = {'train_loss': avg_loss}
        for k, v in epoch_metrics.items():
            res[f"avg_{k}"] = v / max(1, n_batches)

        return res

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_metrics: Dict[str, float] = {}
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                x_win = batch['window'].to(self.device)
                seq_win = batch['seq_windows'].to(self.device) if 'seq_windows' in batch else None

                if self.is_log_psd_baseline:
                    pred_log_psd, _ = self.model(x_win)
                    target_log_psd = batch['log_psd'].to(self.device)
                    mse = F.mse_loss(pred_log_psd, target_log_psd).item()
                    all_metrics['val_log_psd_mse'] = all_metrics.get('val_log_psd_mse', 0.0) + mse
                else:
                    out = self.model(x_win, seq_win)
                    if out['scores_a'] is not None and 'targets_a' in batch:
                        m_a = compute_ranking_metrics_dict(out['scores_a'], batch['targets_a'], prefix='val_varA_')
                        for k, v in m_a.items():
                            all_metrics[k] = all_metrics.get(k, 0.0) + v

                    if out['scores_c'] is not None and 'targets_c' in batch:
                        m_c = compute_ranking_metrics_dict(out['scores_c'], batch['targets_c'], prefix='val_varC_')
                        for k, v in m_c.items():
                            all_metrics[k] = all_metrics.get(k, 0.0) + v
                count += 1

        if count > 0:
            for k in list(all_metrics.keys()):
                all_metrics[k] /= count

        return all_metrics

    def fit(self) -> Dict[str, Any]:
        print(f"Starting Pretraining on {self.device} | Epochs: {self.num_epochs} | Baseline: {self.is_log_psd_baseline}")
        best_metric = -1.0 if not self.is_log_psd_baseline else 1e9

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            train_stats = self.train_epoch(epoch)

            # Warmup vs Cosine LR
            if epoch > self.warmup_epochs:
                self.lr_scheduler.step()

            # Validation
            val_stats = {}
            if self.val_loader is not None:
                val_stats = self.evaluate(self.val_loader)

            elapsed = time.time() - t0
            self.history['train_loss'].append(train_stats['train_loss'])
            self.history['epoch_times'].append(elapsed)

            val_rho_a = val_stats.get('val_varA_spearman_rho', 0.0)
            val_rho_c = val_stats.get('val_varC_spearman_rho', 0.0)
            self.history['val_spearman_a'].append(val_rho_a)
            self.history['val_spearman_c'].append(val_rho_c)

            print(
                f"[Epoch {epoch:02d}/{self.num_epochs:02d}] "
                f"Loss: {train_stats['train_loss']:.4f} | "
                f"Val Rho A: {val_rho_a:.3f} | "
                f"Val Rho C: {val_rho_c:.3f} | "
                f"Time: {elapsed:.2f}s"
            )

        # Save final checkpoint
        ckpt_path = os.path.join(self.save_dir, "pretrained_encoder.pt")
        backbone = self.model.backbone if hasattr(self.model, 'backbone') else self.model
        torch.save(backbone.state_dict(), ckpt_path)
        print(f"Saved pretrained encoder checkpoint to: {ckpt_path}")

        return self.history
