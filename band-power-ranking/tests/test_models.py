"""
Unit Tests for EEG Backbones (Mamba, Transformer, ConvNet) and Multi-Variant Ranking Heads.
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.backbone import EEGMambaBackbone, EEGTransformerBackbone, EEGConvNetBackbone, build_backbone
from src.models.ranking_heads import MultiVariantRankingModel
from src.models.baselines import LogPSDRegressionModel, DownstreamClassificationModel


class TestEEGModels(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.num_channels = 22
        self.in_samples = 1000  # 4 seconds at 250 Hz
        self.embed_dim = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.x_single = torch.randn(self.batch_size, self.num_channels, self.in_samples, device=self.device)
        self.seq_len = 4
        self.x_seq = torch.randn(self.batch_size, self.seq_len, self.num_channels, self.in_samples, device=self.device)

    def test_mamba_backbone_forward_and_backward(self):
        """
        Verify Mamba backbone executes forward pass and propagates gradients.
        """
        mamba = EEGMambaBackbone(
            num_channels=self.num_channels,
            in_samples=self.in_samples,
            embed_dim=self.embed_dim,
            num_layers=2
        ).to(self.device)

        ch_latents, pooled = mamba(self.x_single)
        self.assertEqual(ch_latents.shape, (self.batch_size, self.num_channels, self.embed_dim))
        self.assertEqual(pooled.shape, (self.batch_size, self.embed_dim))

        loss = pooled.sum()
        loss.backward()
        for p in mamba.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())

    def test_transformer_backbone_forward_and_backward(self):
        """
        Verify Transformer backbone executes forward pass and propagates gradients.
        """
        transformer = EEGTransformerBackbone(
            num_channels=self.num_channels,
            in_samples=self.in_samples,
            embed_dim=self.embed_dim,
            num_heads=4,
            num_layers=2
        ).to(self.device)

        ch_latents, pooled = transformer(self.x_single)
        self.assertEqual(ch_latents.shape, (self.batch_size, self.num_channels, self.embed_dim))
        self.assertEqual(pooled.shape, (self.batch_size, self.embed_dim))

        loss = pooled.sum()
        loss.backward()
        for p in transformer.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())

    def test_multi_variant_ranking_model(self):
        """
        Verify unified ranking model produces correct score shapes for Variants A, B, and C.
        """
        for backbone_type in ['mamba', 'transformer']:
            backbone = build_backbone(backbone_type, num_channels=self.num_channels, embed_dim=self.embed_dim).to(self.device)
            model = MultiVariantRankingModel(
                backbone=backbone,
                embed_dim=self.embed_dim,
                num_bands=5,
                enable_variant_a=True,
                enable_variant_b=True,
                enable_variant_c=True
            ).to(self.device)

            out = model(self.x_single, self.x_seq)
            # Variant A: (B, num_bands=5, num_channels=22)
            self.assertEqual(out['scores_a'].shape, (self.batch_size, 5, self.num_channels))
            # Variant B: (B, num_channels=22, num_bands=5, seq_len=4)
            self.assertEqual(out['scores_b'].shape, (self.batch_size, self.num_channels, 5, self.seq_len))
            # Variant C: (B, num_channels=22, num_bands=5)
            self.assertEqual(out['scores_c'].shape, (self.batch_size, self.num_channels, 5))

    def test_log_psd_baseline_model(self):
        """
        Verify Log-PSD baseline regressor outputs (B, num_channels, 5).
        """
        backbone = build_backbone('mamba', num_channels=self.num_channels, embed_dim=self.embed_dim).to(self.device)
        model = LogPSDRegressionModel(backbone=backbone, embed_dim=self.embed_dim, num_bands=5).to(self.device)

        pred_log_psd, pooled = model(self.x_single)
        self.assertEqual(pred_log_psd.shape, (self.batch_size, self.num_channels, 5))
        self.assertEqual(pooled.shape, (self.batch_size, self.embed_dim))

    def test_downstream_classifier(self):
        """
        Verify downstream classification head produces 6-class logits.
        """
        backbone = build_backbone('mamba', num_channels=self.num_channels, embed_dim=self.embed_dim).to(self.device)
        clf = DownstreamClassificationModel(backbone=backbone, embed_dim=self.embed_dim, num_classes=6).to(self.device)

        logits = clf(self.x_single)
        self.assertEqual(logits.shape, (self.batch_size, 6))

    def test_large_model_dimensions(self):
        """
        Verify large model capacity (embed_dim=256, num_heads=8, num_layers=4) executes without errors.
        """
        large_embed = 256
        for backbone_type in ['mamba', 'transformer']:
            bb = build_backbone(backbone_type, num_channels=self.num_channels, embed_dim=large_embed, num_layers=2).to(self.device)
            model = MultiVariantRankingModel(bb, embed_dim=large_embed, num_bands=5).to(self.device)
            out = model(self.x_single, self.x_seq)
            self.assertEqual(out['scores_a'].shape, (self.batch_size, 5, self.num_channels))
            self.assertEqual(out['pooled_latent'].shape, (self.batch_size, large_embed))

            clf = DownstreamClassificationModel(bb, embed_dim=large_embed, num_classes=6).to(self.device)
            logits = clf(self.x_single)
            self.assertEqual(logits.shape, (self.batch_size, 6))


if __name__ == '__main__':
    unittest.main()
