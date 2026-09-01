"""
Ordinal Ranking Target Construction for EEG Self-Supervised Pretraining.

Implements rank target builders for:
- Variant A: Cross-Channel Ranking (spatial structure per band per window)
- Variant B: Cross-Time Ranking (temporal dynamics per channel per band across window sequences)
- Variant C: Cross-Band Ranking (spectral profile shape per channel per window)

Includes tie-aware handling, power-difference margin gating, and differentiable ranking formats.
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, Union
import torch


class RankTargetBuilder:
    """
    Constructs ground-truth ranking targets, rank scores, and pairwise comparison matrices.
    """

    def __init__(self, tie_margin: float = 1e-3, relative_tie_margin: bool = True):
        """
        Args:
            tie_margin: Minimum power difference threshold below which pairs are considered tied.
            relative_tie_margin: If True, uses relative difference `|p1 - p2| / (0.5 * (p1 + p2) + eps)`.
        """
        self.tie_margin = tie_margin
        self.relative_tie_margin = relative_tie_margin

    def compute_ranks_1d(self, values: np.ndarray) -> np.ndarray:
        """
        Computes fractional average ranks for a 1D array of values (higher value = higher rank).
        Normalized rank scale: 0 to (len - 1), or 1 to len.
        """
        # rankdata returns 1-based ranks with average ties
        ranks = stats.rankdata(values, method='average')
        return ranks.astype(np.float32)

    def build_pairwise_targets(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds pairwise target matrix Y and validity mask M for an array of values.
        
        Args:
            values: 1D array of length N (e.g. power values across channels, time, or bands).
            
        Returns:
            Y: Pairwise preference matrix (N, N) where:
               Y[i, j] = 1 if values[i] > values[j]
               Y[i, j] = 0 if values[i] == values[j]
               Y[i, j] = -1 if values[i] < values[j]
            M: Validity mask matrix (N, N), M[i, j] = 1 if pair is distinct (exceeds tie_margin) and i != j.
        """
        n = len(values)
        val_i = values[:, np.newaxis]
        val_j = values[np.newaxis, :]
        diff = val_i - val_j

        if self.relative_tie_margin:
            denom = 0.5 * (np.abs(val_i) + np.abs(val_j)) + 1e-12
            rel_diff = np.abs(diff) / denom
            valid_mask = (rel_diff >= self.tie_margin) & (np.eye(n) == 0)
        else:
            valid_mask = (np.abs(diff) >= self.tie_margin) & (np.eye(n) == 0)

        pairwise_y = np.sign(diff).astype(np.float32)
        # For tied items within margin, zero them out in mask
        return pairwise_y, valid_mask.astype(np.float32)

    def build_variant_a_targets(
        self,
        band_powers: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Variant A: Cross-Channel Ranking (Spatial Structure).
        For each band b, rank all C channels.
        
        Args:
            band_powers: Shape `(batch_size, num_channels, num_bands)` or `(num_channels, num_bands)`.
            
        Returns:
            Dictionary containing:
            - 'ranks': Shape `(B, num_bands, num_channels)` continuous normalized ranks.
            - 'order': Shape `(B, num_bands, num_channels)` channel indices sorted by ascending/descending power.
            - 'pairwise_y': Shape `(B, num_bands, num_channels, num_channels)`.
            - 'pairwise_mask': Shape `(B, num_bands, num_channels, num_channels)`.
            - 'raw_powers': Shape `(B, num_bands, num_channels)`.
        """
        if band_powers.ndim == 2:
            band_powers = band_powers[np.newaxis, ...]  # (1, C, B)

        B, C, Bnd = band_powers.shape
        # Transpose to (B, Bnd, C) for band-first indexing
        bp_trans = np.transpose(band_powers, (0, 2, 1))

        ranks_out = np.zeros((B, Bnd, C), dtype=np.float32)
        order_out = np.zeros((B, Bnd, C), dtype=np.int64)
        pairwise_y = np.zeros((B, Bnd, C, C), dtype=np.float32)
        pairwise_mask = np.zeros((B, Bnd, C, C), dtype=np.float32)

        for b_idx in range(B):
            for band_idx in range(Bnd):
                powers = bp_trans[b_idx, band_idx]  # (C,)
                ranks_out[b_idx, band_idx] = stats.rankdata(powers, method='average')
                order_out[b_idx, band_idx] = np.argsort(powers)
                y_mat, m_mat = self.build_pairwise_targets(powers)
                pairwise_y[b_idx, band_idx] = y_mat
                pairwise_mask[b_idx, band_idx] = m_mat

        return {
            'ranks': ranks_out,
            'order': order_out,
            'pairwise_y': pairwise_y,
            'pairwise_mask': pairwise_mask,
            'raw_powers': bp_trans,
        }

    def build_variant_b_targets(
        self,
        seq_band_powers: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Variant B: Cross-Time Ranking (Temporal Structure / Drift).
        For each channel c and band b, rank K temporal windows.
        
        Args:
            seq_band_powers: Shape `(batch_size, seq_len, num_channels, num_bands)`.
            
        Returns:
            Dictionary containing:
            - 'ranks': Shape `(B, num_channels, num_bands, seq_len)`.
            - 'order': Shape `(B, num_channels, num_bands, seq_len)`.
            - 'pairwise_y': Shape `(B, num_channels, num_bands, seq_len, seq_len)`.
            - 'pairwise_mask': Shape `(B, num_channels, num_bands, seq_len, seq_len)`.
        """
        if seq_band_powers.ndim == 3:
            seq_band_powers = seq_band_powers[np.newaxis, ...]  # (1, K, C, Bnd)

        B, K, C, Bnd = seq_band_powers.shape
        # Transpose to (B, C, Bnd, K) for temporal ranking
        seq_trans = np.transpose(seq_band_powers, (0, 2, 3, 1))

        ranks_out = np.zeros((B, C, Bnd, K), dtype=np.float32)
        order_out = np.zeros((B, C, Bnd, K), dtype=np.int64)
        pairwise_y = np.zeros((B, C, Bnd, K, K), dtype=np.float32)
        pairwise_mask = np.zeros((B, C, Bnd, K, K), dtype=np.float32)

        for b_idx in range(B):
            for c_idx in range(C):
                for band_idx in range(Bnd):
                    powers = seq_trans[b_idx, c_idx, band_idx]  # (K,)
                    ranks_out[b_idx, c_idx, band_idx] = stats.rankdata(powers, method='average')
                    order_out[b_idx, c_idx, band_idx] = np.argsort(powers)
                    y_mat, m_mat = self.build_pairwise_targets(powers)
                    pairwise_y[b_idx, c_idx, band_idx] = y_mat
                    pairwise_mask[b_idx, c_idx, band_idx] = m_mat

        return {
            'ranks': ranks_out,
            'order': order_out,
            'pairwise_y': pairwise_y,
            'pairwise_mask': pairwise_mask,
            'raw_powers': seq_trans,
        }

    def build_variant_c_targets(
        self,
        band_powers: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Variant C: Cross-Band Ranking (Spectral Profile Shape).
        For each channel c in window t, rank the 5 canonical bands.
        
        Args:
            band_powers: Shape `(batch_size, num_channels, num_bands)`.
            
        Returns:
            Dictionary containing:
            - 'ranks': Shape `(B, num_channels, num_bands)`.
            - 'order': Shape `(B, num_channels, num_bands)`.
            - 'pairwise_y': Shape `(B, num_channels, num_bands, num_bands)`.
            - 'pairwise_mask': Shape `(B, num_channels, num_bands, num_bands)`.
        """
        if band_powers.ndim == 2:
            band_powers = band_powers[np.newaxis, ...]  # (1, C, Bnd)

        B, C, Bnd = band_powers.shape

        ranks_out = np.zeros((B, C, Bnd), dtype=np.float32)
        order_out = np.zeros((B, C, Bnd), dtype=np.int64)
        pairwise_y = np.zeros((B, C, Bnd, Bnd), dtype=np.float32)
        pairwise_mask = np.zeros((B, C, Bnd, Bnd), dtype=np.float32)

        for b_idx in range(B):
            for c_idx in range(C):
                powers = band_powers[b_idx, c_idx]  # (Bnd,)
                ranks_out[b_idx, c_idx] = stats.rankdata(powers, method='average')
                order_out[b_idx, c_idx] = np.argsort(powers)
                y_mat, m_mat = self.build_pairwise_targets(powers)
                pairwise_y[b_idx, c_idx] = y_mat
                pairwise_mask[b_idx, c_idx] = m_mat

        return {
            'ranks': ranks_out,
            'order': order_out,
            'pairwise_y': pairwise_y,
            'pairwise_mask': pairwise_mask,
            'raw_powers': band_powers,
        }
