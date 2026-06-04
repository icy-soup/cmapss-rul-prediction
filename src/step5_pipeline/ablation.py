"""Ablation experiments for Person 2: MSPatch-iTransformer-RUL.

Usage: python -u src/step5_pipeline/ablation.py [--subset FD001]

Experiments:
  A: Branch count — 8 variants testing different branch combinations
  B: Fusion method — 3 variants on 3-branch architecture
  C: Best config on FD004
"""

import argparse, json, csv, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.system.config import BaseConfig, SUBSET_CONFIG
from src.system.metrics import all_metrics
from src.step1_preprocessing.loader import load_data
from src.step3_models.person2_model import MSPatchiTransformerRUL
from src.step3_models.trainer import Trainer


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(cfg, label, save_dir="results/ablation"):
    """Run a single ablation experiment. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"ABLATION: {label}")
    print(f"  branch_indices={cfg.branch_indices}, fusion={cfg.fusion_mode}")
    print(f"  subset={cfg.subset}, lr={cfg.lr}, bs={cfg.batch_size}")
    print(f"{'='*60}")

    set_seed(cfg.seed)

    # Load data
    X_train, y_train, unit_ids, X_test, y_test = load_data(cfg)

    # Split by engine
    unique_engines = np.unique(unit_ids)
    np.random.shuffle(unique_engines)
    n_val = max(1, int(len(unique_engines) * (1 - cfg.train_ratio)))
    val_engines = set(unique_engines[:n_val])
    train_engines = set(unique_engines[n_val:])
    val_mask = np.array([u in val_engines for u in unit_ids])
    X_val, y_val = X_train[val_mask], y_train[val_mask]
    X_tr, y_tr = X_train[~val_mask], y_train[~val_mask]

    # Model
    model = MSPatchiTransformerRUL(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    # Train
    trainer = Trainer(model, cfg, cfg.device)
    history = trainer.fit(X_tr, y_tr, X_val, y_val)

    # Evaluate
    trainer.load_best()
    y_pred = trainer.predict(X_test)
    metrics = all_metrics(y_test, y_pred)

    # Print
    print(f"\n>>> {label}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  pred_range=[{y_pred.min():.1f}, {y_pred.max():.1f}]")

    # Save results
    prefix = f"{save_dir}/ablation_{label.replace('/', '_').replace(' ', '_')}"
    with open(f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(f"{prefix}_y_pred.npy", y_pred)
    np.save(f"{prefix}_y_test.npy", y_test)
    if history:
        np.savez(f"{prefix}_history.npz", **{k: np.array(v) for k, v in history.items()})

    return {**metrics, "params": n_params, "pred_range": f"{y_pred.min():.0f}-{y_pred.max():.0f}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="FD001", choices=["FD001", "FD004"])
    args = parser.parse_args()
    subset = args.subset

    # Determine branch description strings
    BRANCH_NAMES = ["P8S4", "P16S8", "P32S16"]

    # ── Ablation A: Branch count ──────────────────────────
    branch_configs = [
        ((0,), "A0-P8S4"),
        ((1,), "A1-P16S8"),
        ((2,), "A2-P32S16"),
        ((0, 1), "A3-small+medium"),
        ((0, 2), "A4-small+large"),
        ((1, 2), "A5-medium+large"),
        ((0, 1, 2), "A6-all3"),
    ]

    # ── Ablation B: Fusion method ─────────────────────────
    fusion_configs = [
        ("concat", "B0-concat"),
        ("weighted_sum", "B1-weighted_sum"),
        ("gated", "B2-gated"),
    ]

    results = []

    # Run Ablation A
    print(f"\n{'#'*60}")
    print(f"# ABLATION A: Branch Count (subset={subset})")
    print(f"{'#'*60}")
    for indices, label in branch_configs:
        cfg = BaseConfig(**SUBSET_CONFIG[subset])
        cfg.branch_indices = indices
        cfg.fusion_mode = "concat"
        cfg.epochs = 200
        m = run_experiment(cfg, f"{label}-{subset}")
        results.append({
            "experiment": f"{label}-{subset}",
            "branches": "+".join(BRANCH_NAMES[i] for i in indices),
            "n_branches": len(indices),
            "fusion": "concat",
            "subset": subset,
            **m,
        })

    # Run Ablation B (only on 3-branch)
    print(f"\n{'#'*60}")
    print(f"# ABLATION B: Fusion Method (subset={subset})")
    print(f"{'#'*60}")
    for fusion, label in fusion_configs:
        cfg = BaseConfig(**SUBSET_CONFIG[subset])
        cfg.branch_indices = (0, 1, 2)
        cfg.fusion_mode = fusion
        cfg.epochs = 200
        m = run_experiment(cfg, f"{label}-{subset}")
        results.append({
            "experiment": f"{label}-{subset}",
            "branches": "P8S4+P16S8+P32S16",
            "n_branches": 3,
            "fusion": fusion,
            "subset": subset,
            **m,
        })

    # Save ablation summary
    csv_path = "results/ablation/ablation_summary.csv"
    fieldnames = ["experiment", "branches", "n_branches", "fusion", "subset",
                   "RMSE", "R2", "Score", "params", "pred_range"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"ABLATION SUMMARY ({subset})")
    print(f"{'='*70}")
    print(f"{'Experiment':25s} {'Branches':25s} {'RMSE':>8s} {'R²':>7s} {'Score':>10s} {'Params':>10s}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["RMSE"]):
        print(f"{r['experiment']:25s} {r['branches']:25s} {r['RMSE']:>8.2f} {r['R2']:>7.3f} {r['Score']:>10.1f} {r['params']:>10,}")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
