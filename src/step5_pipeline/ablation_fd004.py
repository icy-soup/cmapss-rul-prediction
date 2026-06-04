"""Run key ablation configs on FD004 for validation."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from src.system.config import BaseConfig, SUBSET_CONFIG
from src.system.metrics import all_metrics
from src.step1_preprocessing.loader import load_data
from src.step3_models.person2_model import MSPatchiTransformerRUL
from src.step3_models.trainer import Trainer

def run_one(branch_indices, fusion_mode, label):
    cfg = BaseConfig(**SUBSET_CONFIG['FD004'])
    cfg.branch_indices = branch_indices
    cfg.fusion_mode = fusion_mode
    cfg.epochs = 200
    cfg.num_workers = 0  # Windows compatibility

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    X_train, y_train, unit_ids, X_test, y_test = load_data(cfg)
    unique_engines = np.unique(unit_ids)
    np.random.shuffle(unique_engines)
    n_val = max(1, int(len(unique_engines) * (1 - cfg.train_ratio)))
    val_engines = set(unique_engines[:n_val])
    train_engines = set(unique_engines[n_val:])
    val_mask = np.array([u in val_engines for u in unit_ids])
    X_val, y_val = X_train[val_mask], y_train[val_mask]
    X_tr, y_tr = X_train[~val_mask], y_train[~val_mask]

    model = MSPatchiTransformerRUL(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(f"FD004: {label}")
    print(f"  branches={branch_indices}, fusion={fusion_mode}, params={n_params:,}")
    trainer = Trainer(model, cfg, cfg.device)
    history = trainer.fit(X_tr, y_tr, X_val, y_val)
    trainer.load_best()
    y_pred = trainer.predict(X_test)
    metrics = all_metrics(y_test, y_pred)
    print(f">>> {label}: RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.3f}, Score={metrics['Score']:.1f}")
    print(f"  pred_range=[{y_pred.min():.1f}, {y_pred.max():.1f}]")

    prefix = f"results/ablation/ablation_FD004_{label}"
    with open(f"{prefix}_metrics.json", "w") as f: json.dump(metrics, f)
    np.save(f"{prefix}_y_pred.npy", y_pred)
    np.save(f"{prefix}_y_test.npy", y_test)
    if history:
        np.savez(f"{prefix}_history.npz", **{k: np.array(v) for k, v in history.items()})
    return metrics, n_params, y_pred

results = []
# Best from FD001: small+medium concat
results.append(run_one((0, 1), "concat", "small+medium"))
# Single scales for comparison
results.append(run_one((0,), "concat", "single-small"))
results.append(run_one((1,), "concat", "single-medium"))

print(f"\n{'='*50}")
print("FD004 消融验证结果")
print(f"{'='*50}")
print(f"{'配置':20s} {'RMSE':>8s} {'R²':>7s} {'Score':>10s} {'参数量':>10s} {'预测范围':>15s}")
print("-"*70)
labels = ["small+medium", "single-small", "single-medium"]
for label, (m, p, yp) in zip(labels, results):
    r = yp.max() - yp.min()
    print(f"{label:20s} {m['RMSE']:>8.2f} {m['R2']:>7.3f} {m['Score']:>10.1f} {p:>10,} [{yp.min():.0f},{yp.max():.0f}]")
