"""
Command-Line Interface for Self-Supervised EEG Pretraining on TUEV.

Supports:
- Backbones: Mamba (SSM) and Transformer
- Pretext Tasks: Ordinal Ranking (Variants A, B, C, or Combined) and Log-PSD Baseline
- Loss functions: Pairwise RankNet, ListNet, ListMLE, Soft-Spearman
- Negative Control: Shuffled label permutation
"""

import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader, random_split

# Set path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.backbone import build_backbone
from src.models.ranking_heads import MultiVariantRankingModel
from src.models.baselines import LogPSDRegressionModel
from src.dataset.tuev_pretrain_dataset import TUEVPretrainDataset
from src.training.pretrainer import EEGPretrainer


try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Self-Supervised Pretraining for EEG")
    parser.add_argument("--model-type", type=str, default="mamba", choices=["mamba", "transformer", "convnet"],
                        help="Backbone architecture type")
    parser.add_argument("--data-root", type=str, default="TU-v2.0.1", help="Path to TUEV corpus")
    parser.add_argument("--loss-type", type=str, default="pairwise",
                        choices=["pairwise", "listnet", "listmle", "soft_spearman"],
                        help="Ranking loss formulation")
    parser.add_argument("--variants", type=str, default="A+B+C",
                        help="Ranking variants to activate, e.g., 'A', 'B', 'C', 'A+B+C'")
    parser.add_argument("--tie-margin", type=float, default=1e-3, help="Tie margin for pairwise ranking")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding latent dimension")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of pretraining epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--synthetic-samples", type=int, default=256,
                        help="Number of synthetic samples (if TUEV edf not found or fast benchmarking)")
    parser.add_argument("--is-baseline-logpsd", action="store_true", help="Train Log-PSD direct regression baseline")
    parser.add_argument("--shuffle-negative-control", action="store_true", help="Enable shuffled label negative control")
    parser.add_argument("--save-dir", type=str, default="checkpoints/pretrain", help="Checkpoint directory")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(">>> Self-Supervised EEG Pretraining Pipeline")
    print(f"Model: {args.model_type.upper()} | Loss: {args.loss_type} | Variants: {args.variants}")
    print(f"Baseline Log-PSD: {args.is_baseline_logpsd} | Neg Control: {args.shuffle_negative_control}")
    print("=" * 70)

    # 1. Dataset & DataLoaders
    full_dataset = TUEVPretrainDataset(
        data_root=args.data_root,
        split="train",
        window_duration_sec=4.0,
        tie_margin=args.tie_margin,
        use_augmentation=True,
        synthetic_samples=args.synthetic_samples
    )

    val_size = max(1, int(len(full_dataset) * 0.15))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Dataset Loaded: {len(full_dataset)} windows (Train: {train_size}, Val: {val_size})")

    # 2. Build Backbone & Ranking / Baseline Model
    backbone = build_backbone(
        model_type=args.model_type,
        num_channels=22,
        in_samples=1000,
        embed_dim=args.embed_dim
    )

    var_str = args.variants.upper()
    enable_a = 'A' in var_str
    enable_b = 'B' in var_str
    enable_c = 'C' in var_str

    if args.is_baseline_logpsd:
        model = LogPSDRegressionModel(backbone=backbone, embed_dim=args.embed_dim, num_bands=5)
    else:
        model = MultiVariantRankingModel(
            backbone=backbone,
            embed_dim=args.embed_dim,
            num_bands=5,
            enable_variant_a=enable_a,
            enable_variant_b=enable_b,
            enable_variant_c=enable_c
        )

    # 3. Pretraining Engine
    pretrainer = EEGPretrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_type=args.loss_type,
        variant_a_weight=(1.0 if enable_a else 0.0),
        variant_b_weight=(1.0 if enable_b else 0.0),
        variant_c_weight=(1.0 if enable_c else 0.0),
        tie_margin=args.tie_margin,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        is_log_psd_baseline=args.is_baseline_logpsd,
        shuffle_negative_control=args.shuffle_negative_control,
        save_dir=args.save_dir
    )

    history = pretrainer.fit()
    print("✅ Pretraining completed successfully!")


if __name__ == "__main__":
    main()
