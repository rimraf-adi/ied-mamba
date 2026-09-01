"""
Ranking Heads for Self-Supervised Ordinal Pretext Tasks.

Implements scoring heads:
- Variant A: Cross-Channel Ranking Head (conditioned on learned frequency band embeddings)
- Variant B: Cross-Time Ranking Head (scoring temporal sequences per channel and band)
- Variant C: Cross-Band Ranking Head (scoring 5 canonical bands per channel)
- MultiVariantRankingModel: Unified model combining backbone and active ranking heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union


class VariantARankingHead(nn.Module):
    """
    Variant A: Cross-Channel Spatial Ranking Head.
    Given channel embeddings `(B, C, embed_dim)` and 5 canonical frequency bands,
    conditions each channel representation on a learned band embedding and predicts a scalar score.
    Output: `(B, num_bands, num_channels)`
    """

    def __init__(self, embed_dim: int = 128, band_embed_dim: int = 32, num_bands: int = 5):
        super().__init__()
        self.num_bands = num_bands
        self.band_embeddings = nn.Embedding(num_bands, band_embed_dim)

        hidden_dim = max(64, embed_dim // 2)
        mid_dim = max(32, hidden_dim // 2)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim + band_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 1)  # scalar score
        )

    def forward(self, channel_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel_latents: Shape `(B, C, embed_dim)`.
            
        Returns:
            scores: Shape `(B, num_bands, num_channels)`
        """
        B, C, D = channel_latents.shape
        band_ids = torch.arange(self.num_bands, device=channel_latents.device)
        b_embeds = self.band_embeddings(band_ids)  # (num_bands, band_embed_dim)

        # Expand channel latents: (B, num_bands, C, D)
        ch_exp = channel_latents.unsqueeze(1).expand(B, self.num_bands, C, D)
        # Expand band embeds: (B, num_bands, C, band_embed_dim)
        b_exp = b_embeds.unsqueeze(0).unsqueeze(2).expand(B, self.num_bands, C, -1)

        # Concat along feature dim: (B, num_bands, C, D + band_embed_dim)
        combined = torch.cat([ch_exp, b_exp], dim=-1)
        scores = self.scorer(combined).squeeze(-1)  # (B, num_bands, C)
        return scores


class VariantBRankingHead(nn.Module):
    """
    Variant B: Cross-Time Temporal Ranking Head.
    Given a sequence of K window embeddings per channel `(B, K, C, embed_dim)`,
    predicts temporal ordering scores for each channel and band.
    Output: `(B, C, num_bands, K)`
    """

    def __init__(self, embed_dim: int = 128, band_embed_dim: int = 32, num_bands: int = 5):
        super().__init__()
        self.num_bands = num_bands
        self.band_embeddings = nn.Embedding(num_bands, band_embed_dim)

        hidden_dim = max(64, embed_dim // 2)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim + band_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq_channel_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_channel_latents: Shape `(B, K, C, embed_dim)`.
            
        Returns:
            scores: Shape `(B, C, num_bands, K)`
        """
        B, K, C, D = seq_channel_latents.shape
        band_ids = torch.arange(self.num_bands, device=seq_channel_latents.device)
        b_embeds = self.band_embeddings(band_ids)  # (num_bands, band_embed_dim)

        # Transpose sequence latents to (B, C, K, D)
        seq_trans = seq_channel_latents.permute(0, 2, 1, 3)

        # Expand dimensions for bands: (B, C, num_bands, K, D)
        ch_exp = seq_trans.unsqueeze(2).expand(B, C, self.num_bands, K, D)
        b_exp = b_embeds.view(1, 1, self.num_bands, 1, -1).expand(B, C, self.num_bands, K, -1)

        combined = torch.cat([ch_exp, b_exp], dim=-1)
        scores = self.scorer(combined).squeeze(-1)  # (B, C, num_bands, K)
        return scores


class VariantCRankingHead(nn.Module):
    """
    Variant C: Cross-Band Spectral Profile Head.
    Given channel latents `(B, C, embed_dim)`, scores the 5 canonical bands relative to each other.
    Output: `(B, C, num_bands)`
    """

    def __init__(self, embed_dim: int = 128, num_bands: int = 5):
        super().__init__()
        self.num_bands = num_bands
        hidden_dim = max(64, embed_dim // 2)
        mid_dim = max(32, hidden_dim // 2)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, num_bands)
        )

    def forward(self, channel_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            channel_latents: Shape `(B, C, embed_dim)`
            
        Returns:
            scores: Shape `(B, C, num_bands)`
        """
        scores = self.scorer(channel_latents)  # (B, C, num_bands)
        return scores


class MultiVariantRankingModel(nn.Module):
    """
    Unified Self-Supervised Model wrapping an EEG Backbone (Mamba or Transformer)
    and modular ranking heads (Variant A, Variant B, Variant C).
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 128,
        num_bands: int = 5,
        enable_variant_a: bool = True,
        enable_variant_b: bool = True,
        enable_variant_c: bool = True
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.num_bands = num_bands

        self.head_a = VariantARankingHead(embed_dim=embed_dim, num_bands=num_bands) if enable_variant_a else None
        self.head_b = VariantBRankingHead(embed_dim=embed_dim, num_bands=num_bands) if enable_variant_b else None
        self.head_c = VariantCRankingHead(embed_dim=embed_dim, num_bands=num_bands) if enable_variant_c else None

    def forward(
        self,
        x_window: torch.Tensor,
        seq_windows: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            x_window: Shape `(B, C, in_samples)`
            seq_windows: Optional shape `(B, K, C, in_samples)` for Variant B
            
        Returns:
            Dictionary of predicted scores:
            - 'scores_a': `(B, num_bands, C)`
            - 'scores_b': `(B, C, num_bands, K)` if seq_windows is provided
            - 'scores_c': `(B, C, num_bands)`
            - 'pooled_latent': `(B, embed_dim)`
        """
        ch_latents, pooled = self.backbone(x_window)

        scores_a = self.head_a(ch_latents) if self.head_a is not None else None
        scores_c = self.head_c(ch_latents) if self.head_c is not None else None

        scores_b = None
        if self.head_b is not None and seq_windows is not None:
            B, K, C, T = seq_windows.shape
            seq_flat = seq_windows.view(B * K, C, T)
            seq_ch_latents, _ = self.backbone(seq_flat)
            seq_ch_latents = seq_ch_latents.view(B, K, C, self.embed_dim)
            scores_b = self.head_b(seq_ch_latents)

        return {
            'scores_a': scores_a,
            'scores_b': scores_b,
            'scores_c': scores_c,
            'pooled_latent': pooled,
            'channel_latents': ch_latents
        }
