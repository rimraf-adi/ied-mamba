"""
Ranking Quality and Correlation Evaluation Metrics.

Computes:
- Spearman rank correlation (rho)
- Kendall's tau rank correlation
- Pairwise classification accuracy on non-tied pairs
- Normalized Discounted Cumulative Gain (NDCG@k)
"""

import numpy as np
from scipy import stats
from typing import Dict, Union, Optional
import torch


def compute_spearman_correlation(
    pred_scores: Union[np.ndarray, torch.Tensor],
    true_ranks: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes mean Spearman rank correlation across batch items.
    """
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.detach().cpu().numpy()
    if isinstance(true_ranks, torch.Tensor):
        true_ranks = true_ranks.detach().cpu().numpy()

    # Flatten to list of 1D vectors if multidimensional
    pred_flat = pred_scores.reshape(-1, pred_scores.shape[-1])
    true_flat = true_ranks.reshape(-1, true_ranks.shape[-1])

    rhos = []
    for p, t in zip(pred_flat, true_flat):
        if np.all(p == p[0]) or np.all(t == t[0]):
            rhos.append(0.0)
            continue
        rho, _ = stats.spearmanr(p, t)
        if not np.isnan(rho):
            rhos.append(rho)
        else:
            rhos.append(0.0)

    return float(np.mean(rhos)) if len(rhos) > 0 else 0.0


def compute_kendall_tau(
    pred_scores: Union[np.ndarray, torch.Tensor],
    true_ranks: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes mean Kendall's tau correlation across batch items.
    """
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.detach().cpu().numpy()
    if isinstance(true_ranks, torch.Tensor):
        true_ranks = true_ranks.detach().cpu().numpy()

    pred_flat = pred_scores.reshape(-1, pred_scores.shape[-1])
    true_flat = true_ranks.reshape(-1, true_ranks.shape[-1])

    taus = []
    for p, t in zip(pred_flat, true_flat):
        if np.all(p == p[0]) or np.all(t == t[0]):
            taus.append(0.0)
            continue
        tau, _ = stats.kendalltau(p, t)
        if not np.isnan(tau):
            taus.append(tau)
        else:
            taus.append(0.0)

    return float(np.mean(taus)) if len(taus) > 0 else 0.0


def compute_pairwise_accuracy(
    pred_scores: Union[np.ndarray, torch.Tensor],
    pairwise_y: Union[np.ndarray, torch.Tensor],
    pairwise_mask: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes pairwise accuracy: fraction of non-tied pairs correctly ordered.
    """
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.detach().cpu().numpy()
    if isinstance(pairwise_y, torch.Tensor):
        pairwise_y = pairwise_y.detach().cpu().numpy()
    if isinstance(pairwise_mask, torch.Tensor):
        pairwise_mask = pairwise_mask.detach().cpu().numpy()

    # pred_scores: (..., N)
    s_i = np.expand_dims(pred_scores, -1)
    s_j = np.expand_dims(pred_scores, -2)
    pred_diff = s_i - s_j
    pred_y = np.sign(pred_diff)

    # Correct matches where sign agrees with true pairwise preference
    correct = (pred_y == pairwise_y) & (pairwise_mask > 0)
    total_valid = np.sum(pairwise_mask > 0)

    if total_valid == 0:
        return 1.0
    return float(np.sum(correct) / total_valid)


def compute_ndcg(
    pred_scores: Union[np.ndarray, torch.Tensor],
    true_relevance: Union[np.ndarray, torch.Tensor],
    k: Optional[int] = None
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG@k).
    """
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.detach().cpu().numpy()
    if isinstance(true_relevance, torch.Tensor):
        true_relevance = true_relevance.detach().cpu().numpy()

    pred_flat = pred_scores.reshape(-1, pred_scores.shape[-1])
    true_flat = true_relevance.reshape(-1, true_relevance.shape[-1])

    n_items = pred_flat.shape[-1]
    if k is None or k > n_items:
        k = n_items

    ndcg_scores = []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    for p, rel in zip(pred_flat, true_flat):
        # Predicted ranking order (descending score)
        pred_order = np.argsort(-p)[:k]
        dcg = np.sum((2.0 ** rel[pred_order] - 1.0) * discounts)

        # Ideal ranking order
        ideal_order = np.argsort(-rel)[:k]
        idcg = np.sum((2.0 ** rel[ideal_order] - 1.0) * discounts)

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(1.0)

    return float(np.mean(ndcg_scores)) if len(ndcg_scores) > 0 else 1.0


def compute_ranking_metrics_dict(
    pred_scores: Union[np.ndarray, torch.Tensor],
    target_dict: Dict[str, Union[np.ndarray, torch.Tensor]],
    prefix: str = ""
) -> Dict[str, float]:
    """
    Computes all standard ranking metrics for a given variant.
    """
    ranks = target_dict['ranks']
    p_y = target_dict.get('pairwise_y')
    p_m = target_dict.get('pairwise_mask')

    spearman = compute_spearman_correlation(pred_scores, ranks)
    kendall = compute_kendall_tau(pred_scores, ranks)
    ndcg = compute_ndcg(pred_scores, ranks)

    metrics = {
        f"{prefix}spearman_rho": spearman,
        f"{prefix}kendall_tau": kendall,
        f"{prefix}ndcg": ndcg,
    }

    if p_y is not None and p_m is not None:
        p_acc = compute_pairwise_accuracy(pred_scores, p_y, p_m)
        metrics[f"{prefix}pairwise_acc"] = p_acc

    return metrics
