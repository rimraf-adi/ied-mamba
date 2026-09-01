"""
Unit Tests for Ranking Losses: Pairwise RankNet, ListNet, ListMLE, and Soft-Spearman.
"""

import unittest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.losses.ranking_losses import (
    PairwiseRankNetLoss,
    ListNetLoss,
    ListMLELoss,
    SoftSpearmanLoss,
    MultiVariantRankingLoss
)
from src.losses.metrics import (
    compute_spearman_correlation,
    compute_pairwise_accuracy,
    compute_ndcg
)


class TestRankingLosses(unittest.TestCase):

    def test_pairwise_ranknet_loss_gradient(self):
        loss_fn = PairwiseRankNetLoss()
        pred_scores = torch.tensor([[1.0, 3.0, 2.0]], requires_grad=True)
        # Pairwise true relation: item 1 (val 3) > item 2 (val 2) > item 0 (val 1)
        pairwise_y = torch.tensor([[[0.0, -1.0, -1.0],
                                    [1.0,  0.0,  1.0],
                                    [1.0, -1.0,  0.0]]])
        mask = torch.tensor([[[0.0, 1.0, 1.0],
                              [1.0, 0.0, 1.0],
                              [1.0, 1.0, 0.0]]])

        loss = loss_fn(pred_scores, pairwise_y, mask)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(pred_scores.grad)
        self.assertEqual(pred_scores.grad.shape, pred_scores.shape)

    def test_listnet_loss(self):
        loss_fn = ListNetLoss()
        pred_scores = torch.randn(4, 5, requires_grad=True)
        true_ranks = torch.tensor([[1., 2., 3., 4., 5.] for _ in range(4)])
        loss = loss_fn(pred_scores, true_ranks)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(pred_scores.grad)

    def test_listmle_loss(self):
        loss_fn = ListMLELoss()
        pred_scores = torch.randn(4, 5, requires_grad=True)
        true_order = torch.tensor([[4, 3, 2, 1, 0] for _ in range(4)], dtype=torch.long)
        loss = loss_fn(pred_scores, true_order)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(pred_scores.grad)

    def test_soft_spearman_loss(self):
        loss_fn = SoftSpearmanLoss(temperature=0.1)
        # Perfectly aligned scores should yield loss close to 0
        pred_perfect = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True)
        true_ranks = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        loss_perfect = loss_fn(pred_perfect, true_ranks)
        self.assertLess(loss_perfect.item(), 0.1, "Perfect alignment should give low Spearman loss")

        # Inverted scores should yield high loss (~ 2.0)
        pred_inverted = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]], requires_grad=True)
        loss_inverted = loss_fn(pred_inverted, true_ranks)
        self.assertGreater(loss_inverted.item(), 1.5, "Inverted alignment should give high Spearman loss")

    def test_metrics_computation(self):
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trues = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rho = compute_spearman_correlation(preds, trues)
        self.assertAlmostEqual(rho, 1.0, places=4)

        ndcg = compute_ndcg(preds, trues)
        self.assertAlmostEqual(ndcg, 1.0, places=4)


if __name__ == '__main__':
    unittest.main()
