"""
Automated Ablation Study and Comprehensive Benchmark Runner across Mamba and Transformer Architectures.

Executes and compares:
1. Model Backbones: Mamba (SSM) vs. Transformer vs. ConvNet
2. Pretext Objectives:
   - Proposed Ordinal Multi-Variant Ranking (A+B+C)
   - Variant A (Spatial Cross-Channel)
   - Variant B (Temporal Cross-Time)
   - Variant C (Spectral Profile Cross-Band)
   - Biased Baseline: Direct Log-PSD Regression
   - Untrained Lower Bound (Random Initialization)
   - Negative Control: Shuffled Rank Permutations
3. Ranking Loss Functions: Pairwise RankNet vs. ListNet vs. ListMLE vs. Soft-Spearman
4. Tie-Handling Margins: delta = 0, 1e-4, 1e-3, 1e-2
5. Downstream Protocol: Linear Probe vs. Full Fine-Tuning on TUEV 6-class classification
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.backbone import build_backbone
from src.models.ranking_heads import MultiVariantRankingModel
from src.models.baselines import LogPSDRegressionModel
from src.dataset.tuev_pretrain_dataset import TUEVPretrainDataset
from src.dataset.tuev_downstream_dataset import TUEVDownstreamDataset
from src.dataset import FastSubset
from src.training.pretrainer import EEGPretrainer
from src.training.evaluator import TUEVEvaluator


try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def run_experiment(
    model_type: str,
    pretext_task: str,
    loss_type: str = 'pairwise',
    tie_margin: float = 1e-3,
    eval_mode: str = 'linear_probe',
    pretrain_epochs: int = 10,
    eval_epochs: int = 15,
    synthetic_samples: int = 256,
    embed_dim: int = 256,
    patience: int = 10,
    device: Optional[torch.device] = None,
    output_dir: str = "ablation_results"
) -> dict:
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"[ABLATION] [{model_type.upper()}] | Task: [{pretext_task}] | Loss: [{loss_type}] | Tie Margin: [{tie_margin}] | Mode: [{eval_mode}]")
    print("=" * 80)

    # 1. Prepare Datasets (Single Split)
    pretrain_ds = TUEVPretrainDataset(
        data_root=r"d:\ied\TU-v2.0.1",
        window_duration_sec=4.0,
        tie_margin=tie_margin,
        use_augmentation=True,
        synthetic_samples=synthetic_samples
    )
    val_sz = max(1, int(len(pretrain_ds) * 0.15))
    train_sz = len(pretrain_ds) - val_sz
    import random
    import numpy as np
    from torch.utils.data import WeightedRandomSampler
    
    indices = list(range(len(pretrain_ds)))
    random.seed(42)
    random.shuffle(indices)
    pt_train = FastSubset(pretrain_ds, indices[:train_sz])
    pt_val = FastSubset(pretrain_ds, indices[train_sz:])
    pt_train_loader = DataLoader(pt_train, batch_size=512, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    pt_val_loader = DataLoader(pt_val, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # Downstream dataset parameters are defined here; instantiated per task later.
    ds_kwargs = dict(
        data_root=r"d:\ied\TU-v2.0.1",
        window_duration_sec=4.0,
        synthetic_samples=synthetic_samples if synthetic_samples is None else int(synthetic_samples * 1.2)
    )

    # 2. Build Backbone
    backbone = build_backbone(model_type, num_channels=22, embed_dim=embed_dim).to(device)

    # 3. Pretraining (if not random_init)
    pretrain_time = 0.0
    val_rho_a = 0.0
    val_rho_c = 0.0

    run_signature = f"{model_type}_{pretext_task}_{loss_type}_m{tie_margin}_dim{embed_dim}"
    save_dir = os.path.join(output_dir, run_signature)
    os.makedirs(save_dir, exist_ok=True)

    if pretext_task != 'random_init':
        is_log_psd = (pretext_task == 'log_psd')
        is_shuffled = (pretext_task == 'shuffled_control')

        enable_a = ('variant_a' in pretext_task or pretext_task in ('ordinal_all', 'shuffled_control'))
        enable_b = ('variant_b' in pretext_task or pretext_task in ('ordinal_all', 'shuffled_control'))
        enable_c = ('variant_c' in pretext_task or pretext_task in ('ordinal_all', 'shuffled_control'))

        if is_log_psd:
            pt_model = LogPSDRegressionModel(backbone, embed_dim=embed_dim, num_bands=5).to(device)
        else:
            pt_model = MultiVariantRankingModel(
                backbone=backbone,
                embed_dim=embed_dim,
                num_bands=5,
                enable_variant_a=enable_a,
                enable_variant_b=enable_b,
                enable_variant_c=enable_c
            ).to(device)

        pretrainer = EEGPretrainer(
            model=pt_model,
            train_loader=pt_train_loader,
            val_loader=pt_val_loader,
            loss_type=loss_type,
            variant_a_weight=(1.0 if enable_a else 0.0),
            variant_b_weight=(1.0 if enable_b else 0.0),
            variant_c_weight=(1.0 if enable_c else 0.0),
            tie_margin=tie_margin,
            lr=5e-4,
            num_epochs=pretrain_epochs,
            patience=patience,
            device=device,
            is_log_psd_baseline=is_log_psd,
            shuffle_negative_control=is_shuffled,
            save_dir=os.path.join(save_dir, "temp_ckpt")
        )
        t0 = time.time()
        pt_history = pretrainer.fit()
        pretrain_time = time.time() - t0
        if pt_history['val_spearman_a']:
            val_rho_a = pt_history['val_spearman_a'][-1]
        if pt_history['val_spearman_c']:
            val_rho_c = pt_history['val_spearman_c'][-1]

    # 4. Downstream Evaluation (Three-Pronged Protocol)
    eval_tasks = [
        ("detection", 2),
        ("typing", 5)
    ]
    
    downstream_metrics = {}
    eval_time = 0.0
    
    for task_name, num_cls in eval_tasks:
        print(f"\n--- Running Downstream Task: {task_name.upper()} ---")
        
        # Bug #4 Fix: Reload pretrained backbone from checkpoint before each task.
        # This prevents Detection fine-tuning from corrupting the backbone for Typing.
        pretrained_ckpt = os.path.join(save_dir, "temp_ckpt", "pretrained_encoder.pt")
        if os.path.exists(pretrained_ckpt) and pretext_task not in ('random_init',):
            print(f"  Reloading pretrained backbone from: {pretrained_ckpt}")
            backbone = build_backbone(model_type, num_channels=22, embed_dim=embed_dim).to(device)
            backbone.load_state_dict(torch.load(pretrained_ckpt, map_location=device))
        
        curr_train = TUEVDownstreamDataset(**ds_kwargs, split="train", task_mode=task_name)
        curr_val = TUEVDownstreamDataset(**ds_kwargs, split="eval", task_mode=task_name)
        
        ds_train_labels = []
        curr_train._ensure_loaded()
        for i in range(len(curr_train)):
            real_idx = curr_train._valid_indices[i] if hasattr(curr_train, '_valid_indices') else i
            lbl = curr_train._labels_mmap[real_idx].item()
            if task_name == "typing" and lbl > 0: lbl -= 1
            if task_name == "detection" and lbl > 0: lbl = 1
            ds_train_labels.append(lbl)
            
        curr_train._labels_mmap = None
        curr_train._windows_mmap = None
        curr_val._ensure_loaded()
        curr_val._labels_mmap = None
        curr_val._windows_mmap = None
        
        # Class-stratified sampling for typing (balances rare classes like SPSW)
        if task_name == "typing":
            ds_labels_arr = np.array(ds_train_labels)
            class_counts = np.bincount(ds_labels_arr, minlength=num_cls)
            class_weights = 1.0 / (class_counts + 1e-8)
            sample_weights = class_weights[ds_labels_arr]
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(ds_train_labels),
                replacement=True
            )
            curr_train_loader = DataLoader(curr_train, batch_size=512, sampler=sampler, drop_last=False, num_workers=4, pin_memory=True)
        else:
            curr_train_loader = DataLoader(curr_train, batch_size=512, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
        curr_val_loader = DataLoader(curr_val, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
        
        evaluator = TUEVEvaluator(
            backbone=backbone,
            train_loader=curr_train_loader,
            val_loader=curr_val_loader,
            embed_dim=embed_dim,
            num_classes=num_cls,
            mode=eval_mode,
            num_epochs=eval_epochs,
            patience=patience,
            device=device,
            save_dir=os.path.join(save_dir, f"temp_eval_{task_name}")
        )
        
        t_eval0 = time.time()
        metrics = evaluator.run()
        eval_time += (time.time() - t_eval0)
        downstream_metrics[task_name] = metrics

    record = {
        'model_type': model_type,
        'pretext_task': pretext_task,
        'loss_type': loss_type,
        'tie_margin': tie_margin,
        'embed_dim': embed_dim,
        'eval_mode': eval_mode,
        'pretrain_epochs': pretrain_epochs,
        'eval_epochs': eval_epochs,
        'pretrain_time_sec': round(pretrain_time, 2),
        'eval_time_sec': round(eval_time, 2),
        'pretrain_val_rho_a': round(val_rho_a, 4),
        'pretrain_val_rho_c': round(val_rho_c, 4),
        # Detection Metrics
        'detect_auroc': round(downstream_metrics['detection']['auroc'], 4),
        'detect_prauc': round(downstream_metrics['detection']['auprc'], 4),
        'detect_macro_f1': round(downstream_metrics['detection']['macro_f1'], 4),
        # Typing Metrics (Background excluded)
        'type_weighted_f1': round(downstream_metrics['typing']['weighted_f1'], 4),
        'type_macro_f1': round(downstream_metrics['typing']['macro_f1'], 4),
    }
    import json
    with open(os.path.join(save_dir, "detailed_metrics.json"), "w") as f:
        json.dump(downstream_metrics, f, indent=2)
    # Incremental streaming save
    csv_path = os.path.join(output_dir, "ablation_benchmark_results.csv")
    df_row = pd.DataFrame([record])
    if not os.path.exists(csv_path):
        df_row.to_csv(csv_path, index=False)
    else:
        df_row.to_csv(csv_path, mode='a', header=False, index=False)
            
    return record

import numpy as np

def run_full_suite(output_dir: str, pretrain_eps: int = 15, eval_eps: int = 50, patience: int = 15, fast_mode: bool = False):
    records = []
    
    # Force real data!
    samples = None

    backbones = ["mamba"]
    pretext_tasks = ["variant_c"]

    for model in backbones:
        for p_task in pretext_tasks:
            rec = run_experiment(
                model_type=model,
                pretext_task=p_task,
                loss_type="pairwise",
                tie_margin=1e-3,
                embed_dim=512,
                eval_mode="fine_tune",
                pretrain_epochs=pretrain_eps,
                eval_epochs=eval_eps,
                synthetic_samples=samples,
                patience=patience,
                output_dir=output_dir
            )
            records.append(rec)
    # Save to DataFrame & JSON / CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "ablation_benchmark_results.csv")
    json_path = os.path.join(output_dir, "ablation_benchmark_results.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print("\n" + "=" * 80)
    print(">>> FULL ABLATION SUITE COMPLETED")
    print(f"Results saved to: {csv_path} and {json_path}")
    print("=" * 80)
    print(df.to_string())

    try:
        from generate_report import generate_plots_and_report
        generate_plots_and_report(results_csv=csv_path, output_dir=output_dir)
        print(">>> Figures and Markdown summary report generated successfully!")
    except Exception as e:
        print(f"Report generation note: {e}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EEG Ablation Study")
    parser.add_argument("--fast", action="store_true", help="Run fast verification mode")
    parser.add_argument("--epochs", type=int, default=15, help="Number of pretrain epochs")
    parser.add_argument("--eval_epochs", type=int, default=15, help="Number of evaluation epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--out_dir", type=str, default="ablation_results", help="Output directory")
    args = parser.parse_args()

    try:
        run_full_suite(
            output_dir=args.out_dir,
            pretrain_eps=args.epochs,
            eval_eps=args.eval_epochs,
            patience=args.patience,
            fast_mode=args.fast
        )
    except Exception as e:
        import traceback
        print("CRITICAL CRASH IN MAIN SCRIPT:", flush=True)
        traceback.print_exc()
