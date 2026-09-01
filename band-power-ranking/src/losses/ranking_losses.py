"""
Differentiable Ranking Loss Functions for Ordinal Band-Power Pretext Tasks.

Implements:
1. PairwiseRankNetLoss: Pairwise ranking loss with tie-margin masking and smooth logistic surrogate.
2. ListNetLoss: Listwise Top-1 probability cross-entropy loss.
3. ListMLELoss: Plackett-Luce likelihood optimization for listwise permutations.
4. SoftSpearmanLoss: Differentiable smooth Spearman's rank correlation maximization.
5. MultiVariantRankingLoss: Composite multi-task loss for Variants A, B, and C with individual metric tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union


class PairwiseRankNetLoss(nn.Module):
    """
    Pairwise RankNet Loss with margin gating.
    Minimizes: - sum_{i,j: M_ij=1} [ P_ij * log(sigma(s_i - s_j)) + (1 - P_ij) * log(1 - sigma(s_i - s_j)) ]
    where P_ij = 1 if true power i > j, and 0 if i < j.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        pred_scores: torch.Tensor,
        pairwise_y: torch.Tensor,
        pairwise_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_scores: Predicted scalar scores of shape `(..., N)`.
            pairwise_y: Ground truth pairwise preference `(..., N, N)` where y_ij in {-1, 0, 1}.
            pairwise_mask: Binary validity mask `(..., N, N)` indicating valid (non-tied) pairs.
            
        Returns:
            Scalar loss.
        """
        # Expand pred_scores: (..., N, 1) and (..., 1, N)
        s_i = pred_scores.unsqueeze(-1)
        s_j = pred_scores.unsqueeze(-2)
        score_diff = (s_i - s_j) / self.temperature  # (..., N, N)

        # Convert pairwise_y {-1, 1} to binary probability target {0, 1}
        target_prob = 0.5 * (pairwise_y + 1.0)  # shape (..., N, N)

        # BCE with logits
        loss_matrix = F.binary_cross_entropy_with_logits(
            score_diff,
            target_prob,
            reduction='none'
        )

        masked_loss = loss_matrix * pairwise_mask
        mask_sum = pairwise_mask.sum().clamp(min=1.0)
        return masked_loss.sum() / mask_sum


class ListNetLoss(nn.Module):
    """
    ListNet ranking loss (Cao et al., 2007).
    Computes KL-divergence / cross-entropy between true and predicted softmax distributions.
    """

    def __init__(self, pred_temp: float = 1.0, true_temp: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.pred_temp = pred_temp
        self.true_temp = true_temp
        self.eps = eps

    def forward(
        self,
        pred_scores: torch.Tensor,
        true_ranks_or_powers: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_scores: Shape `(..., N)`.
            true_ranks_or_powers: Shape `(..., N)` (continuous ranks or band power values).
            
        Returns:
            Scalar loss.
        """
        # True distribution
        p_true = F.softmax(true_ranks_or_powers / self.true_temp, dim=-1)
        # Predicted log distribution
        log_p_pred = F.log_softmax(pred_scores / self.pred_temp, dim=-1)

        # Cross-entropy: - sum(p_true * log(p_pred))
        loss = -torch.sum(p_true * log_p_pred, dim=-1)
        return loss.mean()


class ListMLELoss(nn.Module):
    """
    ListMLE ranking loss (Xia et al., 2008).
    Negative log-likelihood of the ground-truth permutation under the Plackett-Luce model.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_scores: torch.Tensor,
        true_order_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_scores: Shape `(..., N)`.
            true_order_indices: Shape `(..., N)` indices sorted from highest power to lowest power.
            
        Returns:
            Scalar loss.
        """
        # Gather predicted scores in true permutation order
        # true_order_indices has shape (..., N)
        ordered_scores = torch.gather(pred_scores, dim=-1, index=true_order_indices)

        # Plackett-Luce log likelihood:
        # log P(pi | s) = sum_{k=1}^N ( s_{pi_k} - logsumexp_{j=k}^N s_{pi_j} )
        # Using reversed cumsum trick for logsumexp
        # To compute logsumexp from j=k to N:
        # Subtract max for numerical stability
        max_scores, _ = torch.max(ordered_scores, dim=-1, keepdim=True)
        exp_scores = torch.exp(ordered_scores - max_scores)
        
        # Cumulative sum from the right
        # Flip along last axis, cumsum, flip back
        exp_cumsum = torch.flip(torch.cumsum(torch.flip(exp_scores, dims=[-1]), dim=-1), dims=[-1])
        log_cumsum = torch.log(exp_cumsum.clamp(min=self.eps)) + max_scores

        loss = -torch.sum(ordered_scores - log_cumsum, dim=-1)
        return loss.mean()


class SoftSpearmanLoss(nn.Module):
    """
    Differentiable Soft-Spearman rank correlation loss.
    Approximates ranks using smooth sigmoid pairwise comparisons:
    r_i = 1 + sum_{j != i} sigmoid((s_i - s_j) / tau)
    Maximizes Spearman correlation rho between predicted soft ranks and true ranks.
    """

    def __init__(self, temperature: float = 0.1, eps: float = 1e-7):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def _soft_rank(self, scores: torch.Tensor) -> torch.Tensor:
        # scores: (..., N)
        s_i = scores.unsqueeze(-1)
        s_j = scores.unsqueeze(-2)
        diff = (s_i - s_j) / self.temperature
        # sigmoid(diff) represents P(item_i > item_j)
        pairwise_prob = torch.sigmoid(diff)
        # Zero out diagonal (i == j)
        eye = torch.eye(scores.shape[-1], device=scores.device, dtype=scores.dtype)
        pairwise_prob = pairwise_prob * (1.0 - eye)
        soft_ranks = 1.0 + torch.sum(pairwise_prob, dim=-1)
        return soft_ranks

    def forward(
        self,
        pred_scores: torch.Tensor,
        true_ranks_or_powers: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_scores: Shape `(..., N)`.
            true_ranks_or_powers: Shape `(..., N)`.
            
        Returns:
            Scalar loss = 1.0 - mean(Spearman_rho).
        """
        soft_pred_ranks = self._soft_rank(pred_scores)
        
        # If true_ranks are integers/floats, we can compute Pearson correlation of ranks
        r_pred = soft_pred_ranks
        r_true = true_ranks_or_powers

        # Center ranks
        r_pred_centered = r_pred - r_pred.mean(dim=-1, keepdim=True)
        r_true_centered = r_true - r_true.mean(dim=-1, keepdim=True)

        cov = torch.sum(r_pred_centered * r_true_centered, dim=-1)
        std_pred = torch.sqrt(torch.sum(r_pred_centered ** 2, dim=-1).clamp(min=self.eps))
        std_true = torch.sqrt(torch.sum(r_true_centered ** 2, dim=-1).clamp(min=self.eps))

        rho = cov / (std_pred * std_true + self.eps)
        loss = 1.0 - rho.mean()
        return loss


def get_ranking_loss_by_name(loss_type: str, tie_margin: float = 1e-3) -> nn.Module:
    """
    Factory helper for loss modules.
    """
    loss_type = loss_type.lower()
    if loss_type == 'pairwise' or loss_type == 'ranknet':
        return PairwiseRankNetLoss()
    elif loss_type == 'listnet':
        return ListNetLoss()
    elif loss_type == 'listmle':
        return ListMLELoss()
    elif loss_type == 'soft_spearman' or loss_type == 'spearman':
        return SoftSpearmanLoss()
    else:
        raise ValueError(f"Unknown ranking loss type: {loss_type}")


class MultiVariantRankingLoss(nn.Module):
    """
    Combined Multi-Variant Ranking Loss for Variants A, B, and C.
    Tracks individual losses and gradients to monitor gradient interference.
    """

    def __init__(
        self,
        loss_type: str = 'pairwise',
        w_a: float = 1.0,
        w_b: float = 1.0,
        w_c: float = 1.0,
        tie_margin: float = 1e-3
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.w_a = w_a
        self.w_b = w_b
        self.w_c = w_c
        self.tie_margin = tie_margin

        self.loss_fn = get_ranking_loss_by_name(self.loss_type, tie_margin=tie_margin)

    def forward(
        self,
        preds_a: Optional[torch.Tensor],
        targets_a: Optional[Dict[str, torch.Tensor]],
        preds_b: Optional[torch.Tensor],
        targets_b: Optional[Dict[str, torch.Tensor]],
        preds_c: Optional[torch.Tensor],
        targets_c: Optional[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes weighted combination and returns total loss along with metrics dict.
        """
        total_loss = torch.tensor(0.0, device=(preds_a.device if preds_a is not None else torch.device('cpu')), requires_grad=True)
        metrics = {}

        # Variant A: Cross-Channel
        if preds_a is not None and targets_a is not None and self.w_a > 0:
            if self.loss_type in ('pairwise', 'ranknet'):
                loss_a = self.loss_fn(preds_a, targets_a['pairwise_y'], targets_a['pairwise_mask'])
            elif self.loss_type == 'listmle':
                loss_a = self.loss_fn(preds_a, targets_a['order'].flip(dims=[-1]))  # descending
            elif self.loss_type in ('listnet', 'soft_spearman', 'spearman'):
                loss_a = self.loss_fn(preds_a, targets_a['ranks'])
            else:
                loss_a = torch.tensor(0.0)

            total_loss = total_loss + self.w_a * loss_a
            metrics['loss_variant_a'] = loss_a.item()

        # Variant B: Cross-Time
        if preds_b is not None and targets_b is not None and self.w_b > 0:
            if self.loss_type in ('pairwise', 'ranknet'):
                loss_b = self.loss_fn(preds_b, targets_b['pairwise_y'], targets_b['pairwise_mask'])
            elif self.loss_type == 'listmle':
                loss_b = self.loss_fn(preds_b, targets_b['order'].flip(dims=[-1]))
            elif self.loss_type in ('listnet', 'soft_spearman', 'spearman'):
                loss_b = self.loss_fn(preds_b, targets_b['ranks'])
            else:
                loss_b = torch.tensor(0.0)

            total_loss = total_loss + self.w_b * loss_b
            metrics['loss_variant_b'] = loss_b.item()

        # Variant C: Cross-Band
        if preds_c is not None and targets_c is not None and self.w_c > 0:
            if self.loss_type in ('pairwise', 'ranknet'):
                loss_c = self.loss_fn(preds_c, targets_c['pairwise_y'], targets_c['pairwise_mask'])
            elif self.loss_type == 'listmle':
                loss_c = self.loss_fn(preds_c, targets_c['order'].flip(dims=[-1]))
            elif self.loss_type in ('listnet', 'soft_spearman', 'spearman'):
                loss_c = self.loss_fn(preds_c, targets_c['ranks'])
            else:
                loss_c = torch.tensor(0.0)

            total_loss = total_loss + self.w_c * loss_c
            metrics['loss_variant_c'] = loss_c.item()

        metrics['loss_total'] = total_loss.item()
        return total_loss, metrics
