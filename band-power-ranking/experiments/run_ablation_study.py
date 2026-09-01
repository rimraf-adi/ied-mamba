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
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.backbone import build_backbone
from src.models.ranking_heads import MultiVariantRankingModel
from src.models.baselines import LogPSDRegressionModel
from src.dataset.tuev_pretrain_dataset import TUEVPretrainDataset
from src.dataset.tuev_downstream_dataset import TUEVDownstreamDataset
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
    embed_dim: int = 64,
    device: Optional[torch.device] = None,
    output_dir: str = "ablation_results"
) -> dict:
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"[ABLATION] [{model_type.upper()}] | Task: [{pretext_task}] | Loss: [{loss_type}] | Tie Margin: [{tie_margin}] | Mode: [{eval_mode}]")
    print("=" * 80)

    # 1. Prepare Datasets
    pretrain_ds = TUEVPretrainDataset(
        window_duration_sec=4.0,
        tie_margin=tie_margin,
        use_augmentation=True,
        synthetic_samples=synthetic_samples
    )
    val_sz = max(1, int(len(pretrain_ds) * 0.15))
    train_sz = len(pretrain_ds) - val_sz
    pt_train, pt_val = random_split(pretrain_ds, [train_sz, val_sz])
    pt_train_loader = DataLoader(pt_train, batch_size=32, shuffle=True, drop_last=True)
    pt_val_loader = DataLoader(pt_val, batch_size=32, shuffle=False)

    downstream_ds = TUEVDownstreamDataset(
        window_duration_sec=4.0,
        synthetic_samples=int(synthetic_samples * 1.2)
    )
    ds_val_sz = max(1, int(len(downstream_ds) * 0.30))
    ds_train_sz = len(downstream_ds) - ds_val_sz
    ds_train, ds_val = random_split(downstream_ds, [ds_train_sz, ds_val_sz])
    ds_train_loader = DataLoader(ds_train, batch_size=32, shuffle=True, drop_last=True)
    ds_val_loader = DataLoader(ds_val, batch_size=32, shuffle=False)

    # 2. Build Backbone
    backbone = build_backbone(model_type, num_channels=22, embed_dim=embed_dim).to(device)

    # 3. Pretraining (if not random_init)
    pretrain_time = 0.0
    val_rho_a = 0.0
    val_rho_c = 0.0

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
            device=device,
            is_log_psd_baseline=is_log_psd,
            shuffle_negative_control=is_shuffled,
            save_dir=os.path.join(output_dir, "temp_ckpt")
        )
        t0 = time.time()
        pt_history = pretrainer.fit()
        pretrain_time = time.time() - t0
        if pt_history['val_spearman_a']:
            val_rho_a = pt_history['val_spearman_a'][-1]
        if pt_history['val_spearman_c']:
            val_rho_c = pt_history['val_spearman_c'][-1]

    # 4. Downstream Evaluation
    evaluator = TUEVEvaluator(
        backbone=backbone,
        train_loader=ds_train_loader,
        val_loader=ds_val_loader,
        embed_dim=embed_dim,
        num_classes=6,
        mode=eval_mode,
        num_epochs=eval_epochs,
        device=device,
        save_dir=os.path.join(output_dir, "temp_eval")
    )
    t_eval0 = time.time()
    downstream_metrics = evaluator.run()
    eval_time = time.time() - t_eval0

    record = {
        'model_type': model_type,
        'pretext_task': pretext_task,
        'loss_type': loss_type,
        'tie_margin': tie_margin,
        'eval_mode': eval_mode,
        'pretrain_epochs': pretrain_epochs,
        'eval_epochs': eval_epochs,
        'pretrain_time_sec': round(pretrain_time, 2),
        'eval_time_sec': round(eval_time, 2),
        'pretrain_val_rho_a': round(val_rho_a, 4),
        'pretrain_val_rho_c': round(val_rho_c, 4),
        'balanced_accuracy': round(downstream_metrics['balanced_accuracy'], 4),
        'macro_f1': round(downstream_metrics['macro_f1'], 4),
        'weighted_f1': round(downstream_metrics['weighted_f1'], 4),
        'auroc': round(downstream_metrics['auroc'], 4),
        'auprc': round(downstream_metrics['auprc'], 4),
    }
    return record


def run_full_suite(output_dir: str = "ablation_results", fast_mode: bool = True):
    records = []
    pretrain_eps = 4 if fast_mode else 12
    eval_eps = 6 if fast_mode else 15
    samples = 128 if fast_mode else 300

    backbones = ["mamba", "transformer"]

    # 1. Backbone & Pretext Objective Ablation (Ordinal Ranking vs Log-PSD vs Random vs Shuffled vs Variants)
    pretext_tasks = [
        "ordinal_all",
        "variant_a",
        "variant_b",
        "variant_c",
        "log_psd",
        "random_init",
        "shuffled_control"
    ]

    for model in backbones:
        for p_task in pretext_tasks:
            rec = run_experiment(
                model_type=model,
                pretext_task=p_task,
                loss_type="pairwise",
                tie_margin=1e-3,
                eval_mode="linear_probe",
                pretrain_epochs=pretrain_eps,
                eval_epochs=eval_eps,
                synthetic_samples=samples,
                output_dir=output_dir
            )
            records.append(rec)

    # 2. Ranking Loss Ablation on Mamba & Transformer
    loss_types = ["listnet", "listmle", "soft_spearman"]
    for model in backbones:
        for l_type in loss_types:
            rec = run_experiment(
                model_type=model,
                pretext_task="ordinal_all",
                loss_type=l_type,
                tie_margin=1e-3,
                eval_mode="linear_probe",
                pretrain_epochs=pretrain_eps,
                eval_epochs=eval_eps,
                synthetic_samples=samples,
                output_dir=output_dir
            )
            records.append(rec)

    # 3. Tie Margin Ablation on Mamba
    margins = [0.0, 1e-4, 1e-2]
    for m in margins:
        rec = run_experiment(
            model_type="mamba",
            pretext_task="ordinal_all",
            loss_type="pairwise",
            tie_margin=m,
            eval_mode="linear_probe",
            pretrain_epochs=pretrain_eps,
            eval_epochs=eval_eps,
            synthetic_samples=samples,
            output_dir=output_dir
        )
        records.append(rec)

    # 4. Fine-Tuning Protocol Ablation (Mamba & Transformer on Ordinal vs Baseline)
    for model in backbones:
        for p_task in ["ordinal_all", "log_psd", "random_init"]:
            rec = run_experiment(
                model_type=model,
                pretext_task=p_task,
                loss_type="pairwise",
                tie_margin=1e-3,
                eval_mode="fine_tune",
                pretrain_epochs=pretrain_eps,
                eval_epochs=eval_eps,
                synthetic_samples=samples,
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
    print(df[['model_type', 'pretext_task', 'loss_type', 'eval_mode', 'balanced_accuracy', 'macro_f1', 'auroc']])

    try:
        from generate_report import generate_plots_and_report
        generate_plots_and_report(results_csv=csv_path, output_dir=output_dir)
        print(">>> Figures and Markdown summary report generated successfully!")
    except Exception as e:
        print(f"Report generation note: {e}")

    return df


if __name__ == "__main__":
    run_full_suite()
