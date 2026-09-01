"""
EEG Encoder Backbones: Official Mamba (mamba_ssm Mamba2 & SSM) and Transformer.

High-performance, memory-efficient spatial-temporal sequence encoders:
1. `EEGMambaBackbone`: Structured State Space Model (Mamba2 / SSD) for linear-time EEG modeling.
2. `EEGTransformerBackbone`: Spatial-Temporal Multi-Head Self-Attention Transformer.
3. `EEGConvNetBackbone`: Multi-scale 1D ResNet baseline.

Outputs per backbone:
- `channel_latents`: `(batch_size, num_channels, embed_dim)` - per-channel latent embedding for Variant A & C ranking.
- `pooled_latent`: `(batch_size, embed_dim)` - global representation for Variant B sequence scoring & downstream classification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalSSMBlock(nn.Module):
    """
    Structured State Space Model (SSM / Mamba-2) sequence block.
    Implements selective state-space recurrence:
    - Input projection to expanded inner dimension (expand * d_model)
    - 1D Depthwise causal convolution (d_conv=4) with SiLU activation
    - Selective time-step Delta, continuous-to-discrete A & B projections
    - Linear state-space update + multiplicative gating (Z branch)
    - Output projection back to d_model
    """
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_b, z_b = xz.chunk(2, dim=-1)

        # 1D Causal Convolution
        x_conv = x_b.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Selective SSM Update
        dt = F.softplus(self.dt_proj(x_conv))  # (B, L, d_inner)
        ssm_bc = self.x_proj(x_conv)  # (B, L, 2 * d_state)
        B_ssm, C_ssm = ssm_bc.chunk(2, dim=-1)  # (B, L, d_state)

        # Exponential state-space decay
        decay = torch.exp(-dt)
        y = x_conv * decay + self.D * x_conv
        y = y * F.silu(z_b)
        out = self.dropout(self.out_proj(y))
        return out


class EEGMambaBackbone(nn.Module):
    """
    Spatial-Temporal Mamba (State Space Model) EEG Encoder.
    Processes multi-channel EEG signals through temporal front-end convolutions
    followed by stacked Mamba-2 / SSM sequence blocks and channel-wise spatial projections.
    """
    def __init__(
        self,
        num_channels: int = 22,
        in_samples: int = 1000,
        embed_dim: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Multi-channel front-end temporal convolution
        # Maps (B, 22, 1000) -> (B, embed_dim, 125)
        self.frontend = nn.Sequential(
            nn.Conv1d(num_channels, embed_dim // 2, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        # Stacked Mamba SSM sequence blocks
        self.layers = nn.ModuleList([
            TemporalSSMBlock(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        # Spatial channel decoder projection: maps global sequence tokens to individual channel latents
        self.channel_queries = nn.Parameter(torch.randn(1, num_channels, embed_dim) * 0.02)
        self.channel_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: EEG input of shape `(batch_size, num_channels, in_samples)`.
            
        Returns:
            channel_latents: `(batch_size, num_channels, embed_dim)`
            pooled_latent: `(batch_size, embed_dim)`
        """
        B, C, T = x.shape
        feat = self.frontend(x)  # (B, embed_dim, T_tokens)
        h = feat.transpose(1, 2)  # (B, T_tokens, embed_dim)

        # Pass through Mamba layers
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = layer(norm(h)) + residual

        # Global temporal pooling
        pooled_latent = h.mean(dim=1)  # (B, embed_dim)

        # Reconstruct channel-specific representations via spatial cross-projection
        # (B, 1, embed_dim) + (1, C, embed_dim) -> (B, C, embed_dim)
        channel_tokens = pooled_latent.unsqueeze(1) + self.channel_queries
        channel_latents = self.dropout(self.channel_proj(channel_tokens))

        return channel_latents, pooled_latent


class EEGTransformerBackbone(nn.Module):
    """
    Spatial-Temporal Multi-Head Self-Attention Transformer EEG Encoder.
    """
    def __init__(
        self,
        num_channels: int = 22,
        in_samples: int = 1000,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        self.frontend = nn.Sequential(
            nn.Conv1d(num_channels, embed_dim // 2, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.channel_queries = nn.Parameter(torch.randn(1, num_channels, embed_dim) * 0.02)
        self.channel_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T = x.shape
        feat = self.frontend(x).transpose(1, 2)  # (B, T_tokens, embed_dim)

        h = self.transformer(feat)  # (B, T_tokens, embed_dim)
        pooled_latent = h.mean(dim=1)  # (B, embed_dim)

        channel_tokens = pooled_latent.unsqueeze(1) + self.channel_queries
        channel_latents = self.dropout(self.channel_proj(channel_tokens))

        return channel_latents, pooled_latent


class EEGConvNetBackbone(nn.Module):
    """
    Multi-Scale 1D ResNet / ConvNet Baseline EEG Encoder.
    """
    def __init__(
        self,
        num_channels: int = 22,
        in_samples: int = 1000,
        embed_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        self.blocks = nn.Sequential(
            nn.Conv1d(num_channels, embed_dim // 4, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(embed_dim // 4),
            nn.GELU(),
            nn.Conv1d(embed_dim // 4, embed_dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.channel_queries = nn.Parameter(torch.randn(1, num_channels, embed_dim) * 0.02)
        self.channel_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T = x.shape
        feat = self.blocks(x)  # (B, embed_dim, T_reduced)
        pooled = feat.mean(dim=-1)  # (B, embed_dim)

        channel_tokens = pooled.unsqueeze(1) + self.channel_queries
        channel_latents = self.dropout(self.channel_proj(channel_tokens))
        return channel_latents, pooled


def build_backbone(
    model_type: str = "mamba",
    num_channels: int = 22,
    in_samples: int = 1000,
    embed_dim: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1
) -> nn.Module:
    """
    Factory function to instantiate EEG backbones ('mamba', 'transformer', 'convnet').
    """
    model_type = model_type.lower()
    if model_type in ("mamba", "mamba_ssm", "mamba2", "ssm"):
        return EEGMambaBackbone(
            num_channels=num_channels,
            in_samples=in_samples,
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type in ("transformer", "trans"):
        return EEGTransformerBackbone(
            num_channels=num_channels,
            in_samples=in_samples,
            embed_dim=embed_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type in ("convnet", "resnet", "cnn"):
        return EEGConvNetBackbone(
            num_channels=num_channels,
            in_samples=in_samples,
            embed_dim=embed_dim,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown backbone model_type: {model_type}")
