"""
Baseline Models and Downstream Classification Heads.

1. LogPSDRegressionModel: Baseline SSL model directly regressing log-PSD values.
2. DownstreamClassificationModel: Classifier head on top of backbone for 6-class TUEV classification (Linear Probe & Fine-Tuning).
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class LogPSDRegressionModel(nn.Module):
    """
    Baseline SSL Model: Directly regresses log(band_power) values `(B, C, 5)` using MSE loss.
    This serves as the primary baseline to demonstrate the value of the ordinal ranking formulation.
    """

    def __init__(self, backbone: nn.Module, embed_dim: int = 128, num_bands: int = 5):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.num_bands = num_bands

        hidden_dim = max(64, embed_dim // 2)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bands)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Shape `(B, C, in_samples)`.
            
        Returns:
            pred_log_psd: Shape `(B, C, num_bands)`
            pooled_latent: Shape `(B, embed_dim)`
        """
        ch_latents, pooled = self.backbone(x)
        pred_log_psd = self.regressor(ch_latents)  # (B, C, num_bands)
        return pred_log_psd, pooled


class DownstreamClassificationModel(nn.Module):
    """
    Downstream 6-Class Event Classifier for TUEV.
    Supports both Linear Probe (frozen backbone) and Full Fine-Tuning.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 128,
        num_classes: int = 6,
        freeze_backbone: bool = True,
        use_mlp_head: bool = False
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if use_mlp_head:
            hidden_dim = embed_dim
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # Linear probe head
            self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape `(B, C, in_samples)`.
            
        Returns:
            logits: Shape `(B, num_classes)`
        """
        if self.freeze_backbone:
            with torch.no_grad():
                _, pooled = self.backbone(x)
        else:
            _, pooled = self.backbone(x)

        logits = self.classifier(pooled)
        return logits

    def unfreeze(self):
        """
        Enables gradient updates on the backbone for full fine-tuning.
        """
        self.freeze_backbone = False
        for param in self.backbone.parameters():
            param.requires_grad = True
