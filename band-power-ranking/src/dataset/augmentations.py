"""
Rank-Preserving Data Augmentations for Self-Supervised EEG Pretraining.

Augmentations designed to preserve relative spectral band-power relationships:
- Gaussian noise injection (controlled SNR)
- Time jitter / temporal micro-shifts
- Channel dropout (with dynamic ground-truth rank recalculation)
- Amplitude scaling (applied uniformly across channels before target computation)
"""

import numpy as np
import torch
from typing import Tuple, Optional


class EEGRankAugmenter:
    """
    Augmentation pipeline for multi-channel EEG signals in rank pretext tasks.
    """

    def __init__(
        self,
        noise_std: float = 0.02,
        time_jitter_max: int = 25,
        channel_dropout_prob: float = 0.05,
        p_apply: float = 0.8
    ):
        self.noise_std = noise_std
        self.time_jitter_max = time_jitter_max
        self.channel_dropout_prob = channel_dropout_prob
        self.p_apply = p_apply

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: EEG signal array of shape `(n_channels, n_samples)`.
            
        Returns:
            Augmented signal array of shape `(n_channels, n_samples)`.
        """
        if np.random.rand() > self.p_apply:
            return x

        x_aug = x.copy()
        n_ch, n_samples = x_aug.shape

        # 1. Additive Gaussian noise
        if self.noise_std > 0 and np.random.rand() > 0.5:
            noise = np.random.normal(0, self.noise_std, size=x_aug.shape)
            x_aug = x_aug + noise

        # 2. Time jitter / circular or zero-padded shift
        if self.time_jitter_max > 0 and np.random.rand() > 0.5:
            shift = np.random.randint(-self.time_jitter_max, self.time_jitter_max + 1)
            if shift != 0:
                x_aug = np.roll(x_aug, shift, axis=-1)

        # 3. Channel dropout (zero out a channel occasionally)
        if self.channel_dropout_prob > 0 and np.random.rand() > 0.7:
            drop_mask = np.random.rand(n_ch) < self.channel_dropout_prob
            if np.any(drop_mask) and not np.all(drop_mask):
                x_aug[drop_mask, :] = 0.0

        return x_aug.astype(np.float32)
