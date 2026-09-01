"""
Command-Line Interface for Downstream 6-Class TUEV Evaluation (Linear Probe & Fine-Tuning).
"""

import argparse
import sys
import os
import json
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.backbone import build_backbone
from src.dataset.tuev_downstream_dataset import TUEVDownstreamDataset
from src.training.evaluator import TUEVEvaluator


try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Downstream TUEV Evaluation")
    parser.add_argument("--model-type", type=str, default="mamba", choices=["mamba", "transformer", "convnet"],
                        help="Backbone architecture type")
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="Path to pretrained encoder checkpoint (optional; random init if None)")
    parser.add_argument("--mode", type=str, default="linear_probe", choices=["linear_probe", "fine_tune"],
                        help="Downstream evaluation mode")
    parser.add_argument("--data-root", type=str, default="TU-v2.0.1", help="Path to TUEV corpus")
    parser.add_argument("--embed-dim", type=int, default=128, help="Latent embedding dimension")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Downstream training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--synthetic-samples", type=int, default=300,
                        help="Number of synthetic samples (if TUEV edf not found or fast test)")
    parser.add_argument("--save-dir", type=str, default="checkpoints/downstream", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(">>> Downstream TUEV 6-Class Event Classification Evaluation")
    print(f"Backbone: {args.model_type.upper()} | Mode: {args.mode.upper()}")
    print(f"Checkpoint: {args.ckpt_path or 'Random Initialization (Untrained)'}")
    print("=" * 70)

    # 1. Dataset & Loaders
    full_dataset = TUEVDownstreamDataset(
        data_root=args.data_root,
        split="eval",
        window_duration_sec=4.0,
        synthetic_samples=args.synthetic_samples
    )

    val_size = max(1, int(len(full_dataset) * 0.30))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Backbone
    backbone = build_backbone(
        model_type=args.model_type,
        num_channels=22,
        in_samples=1000,
        embed_dim=args.embed_dim
    )

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        state_dict = torch.load(args.ckpt_path, map_location='cpu', weights_only=True)
        backbone.load_state_dict(state_dict)
        print(f"Loaded pretrained backbone weights from: {args.ckpt_path}")
    else:
        print("Using freshly initialized (untrained) backbone as baseline.")

    # 3. Evaluator
    evaluator = TUEVEvaluator(
        backbone=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        embed_dim=args.embed_dim,
        num_classes=6,
        mode=args.mode,
        num_epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir
    )

    results = evaluator.run()
    print("\n--- Final Downstream Evaluation Results ---")
    print(f"Balanced Accuracy: {results['balanced_accuracy']*100:.2f}%")
    print(f"Macro F1-Score:    {results['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {results['weighted_f1']:.4f}")
    print(f"AUROC:             {results['auroc']:.4f}")
    print(f"AUPRC:             {results['auprc']:.4f}")

    # Save results to JSON
    os.makedirs(args.save_dir, exist_ok=True)
    out_file = os.path.join(args.save_dir, f"eval_{args.model_type}_{args.mode}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {out_file}")


if __name__ == "__main__":
    main()
