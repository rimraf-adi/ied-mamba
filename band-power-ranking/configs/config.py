"""
Configuration dataclasses and defaults for Ordinal PSD Band-Power Ranking SSL Pretraining & TUEV Evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class SpectralConfig:
    fs: float = 250.0  # Common resampled sampling rate (Hz)
    window_duration_sec: float = 4.0  # Window duration in seconds (4s = 1000 samples at 250 Hz)
    welch_nperseg_sec: float = 1.0  # Welch segment duration (1s = 250 samples, yields 1.0 Hz bin resolution)
    welch_overlap_ratio: float = 0.5  # 50% segment overlap within window
    notch_freq: float = 60.0  # US powerline frequency
    notch_q: float = 30.0
    bandpass_low: float = 0.5  # Filter lower bound
    bandpass_high: float = 45.0  # Filter upper bound (below notch)
    
    # Canonical frequency bands (Hz)
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0),
    })

    @property
    def window_samples(self) -> int:
        return int(self.fs * self.window_duration_sec)

    @property
    def nperseg(self) -> int:
        return int(self.fs * self.welch_nperseg_sec)

    @property
    def num_bands(self) -> int:
        return len(self.bands)


@dataclass
class PretrainConfig:
    # Ranking variants active weights
    variant_a_weight: float = 1.0  # Cross-channel ranking (spatial)
    variant_b_weight: float = 1.0  # Cross-time ranking (temporal/drift)
    variant_c_weight: float = 1.0  # Cross-band ranking (spectral profile)
    
    # Ranking loss selection: 'pairwise', 'listnet', 'listmle', 'soft_spearman'
    loss_type: str = 'pairwise'
    tie_margin: float = 1e-3  # Minimum relative power difference margin before pairwise loss is applied
    
    # Temporal sequence length for Variant B (number of consecutive/sampled windows)
    variant_b_seq_len: int = 4
    
    # Training hyperparams
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    grad_clip_norm: float = 1.0
    
    # Augmentations
    use_augmentation: bool = True
    noise_std: float = 0.05
    time_jitter_max_shift: int = 25  # samples (100ms at 250Hz)
    channel_dropout_prob: float = 0.1
    
    # Negative control
    shuffle_labels_control: bool = False
    
    # Logging & Checkpoints
    log_interval: int = 10
    save_dir: str = os.path.join(BASE_DIR, "checkpoints", "pretrain")


@dataclass
class ModelConfig:
    num_channels: int = 22  # Standard ACNS TCP montage channels
    in_samples: int = 1000  # 4 seconds at 250 Hz
    embed_dim: int = 256  # Sufficiently large dimension for high-capacity representation
    encoder_type: str = "mamba"  # 'mamba', 'transformer', 'convnet'
    num_encoder_layers: int = 4
    num_heads: int = 8
    dim_feedforward: int = 512
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    band_embed_dim: int = 64


@dataclass
class DownstreamConfig:
    num_classes: int = 6  # SPSW, GPED, PLED, EYEM, ARTF, BCKG
    class_names: List[str] = field(default_factory=lambda: [
        'bckg', 'spsw', 'gped', 'pled', 'eyem', 'artf'
    ])
    batch_size: int = 64
    linear_probe_epochs: int = 25
    fine_tune_epochs: int = 15
    linear_probe_lr: float = 1e-3
    fine_tune_lr: float = 1e-4
    weight_decay: float = 1e-2
    save_dir: str = os.path.join(BASE_DIR, "checkpoints", "downstream")
