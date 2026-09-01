"""
Unit Tests for Spectral Feature Extraction, PSD Welch Integration, and Rank Target Construction.
Tests on synthetic sinusoidal signals of precisely known spectral content.
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add package root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spectral.psd_extractor import compute_welch_psd, extract_canonical_band_powers, PSDBandPowerExtractor
from src.spectral.rank_target_builder import RankTargetBuilder


class TestSpectralAndRanking(unittest.TestCase):

    def setUp(self):
        self.fs = 250.0
        self.duration = 4.0  # 4 seconds
        self.n_samples = int(self.fs * self.duration)
        self.t = np.linspace(0, self.duration, self.n_samples, endpoint=False)

    def test_pure_sine_spectral_peaks(self):
        """
        Verify that pure sine waves at 2 Hz (Delta), 10 Hz (Alpha), and 20 Hz (Beta)
        yield dominant power in their respective canonical frequency bands.
        """
        extractor = PSDBandPowerExtractor(fs=self.fs)
        
        # 1. Pure Alpha Sine Wave (10 Hz)
        sig_alpha = np.sin(2 * np.pi * 10.0 * self.t).reshape(1, -1)
        bp_alpha = extractor.extract(sig_alpha)[0]  # (5,) -> [delta, theta, alpha, beta, gamma]
        # Alpha is index 2
        self.assertEqual(np.argmax(bp_alpha), 2, "10 Hz sine wave must peak in Alpha band (index 2)")

        # 2. Pure Delta Sine Wave (2 Hz)
        sig_delta = np.sin(2 * np.pi * 2.0 * self.t).reshape(1, -1)
        bp_delta = extractor.extract(sig_delta)[0]
        # Delta is index 0
        self.assertEqual(np.argmax(bp_delta), 0, "2 Hz sine wave must peak in Delta band (index 0)")

        # 3. Pure Beta Sine Wave (20 Hz)
        sig_beta = np.sin(2 * np.pi * 20.0 * self.t).reshape(1, -1)
        bp_beta = extractor.extract(sig_beta)[0]
        # Beta is index 3
        self.assertEqual(np.argmax(bp_beta), 3, "20 Hz sine wave must peak in Beta band (index 3)")

    def test_multi_channel_spatial_ranking(self):
        """
        Verify Variant A rank targets correctly rank channels with increasing alpha power.
        """
        num_channels = 5
        signals = np.zeros((num_channels, self.n_samples), dtype=np.float32)
        # Channel i has amplitude (i + 1) * 1.0 of 10 Hz alpha wave
        for ch in range(num_channels):
            signals[ch] = (ch + 1.0) * np.sin(2 * np.pi * 10.0 * self.t)

        extractor = PSDBandPowerExtractor(fs=self.fs)
        bp = extractor.extract(signals)  # (5, 5)

        builder = RankTargetBuilder(tie_margin=1e-3)
        targs_a = builder.build_variant_a_targets(bp)

        alpha_ranks = targs_a['ranks'][0, 2]  # batch 0, alpha band (idx 2), 5 channels
        expected_ranks = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(alpha_ranks, expected_ranks, atol=1e-3,
                                   err_msg="Channels with strictly increasing power must have ranks [1, 2, 3, 4, 5]")

    def test_tie_margin_handling(self):
        """
        Verify that near-identical power values below tie_margin are masked out in pairwise matrix.
        """
        builder = RankTargetBuilder(tie_margin=0.05, relative_tie_margin=True)
        # 3 values: 1.0, 1.001 (near tie with 1.0), 3.0 (distinct)
        values = np.array([1.0, 1.001, 3.0])
        pairwise_y, mask = builder.build_pairwise_targets(values)

        # Pair (0, 1) and (1, 0) should be masked (mask == 0)
        self.assertEqual(mask[0, 1], 0.0, "Near-tied pair (0, 1) must be masked out")
        self.assertEqual(mask[1, 0], 0.0, "Near-tied pair (1, 0) must be masked out")
        # Pair (0, 2) should be valid
        self.assertEqual(mask[0, 2], 1.0, "Distinct pair (0, 2) must be active")
        self.assertEqual(pairwise_y[2, 0], 1.0, "Value 3.0 > 1.0 must have preference +1")


if __name__ == '__main__':
    unittest.main()
