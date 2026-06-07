"""Person 2 的 Patch 设计消融实验
测试不同的 patch_len / stride 组合在单分支下的表现（FD001）
"""
import sys, os, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from src.system.config import BaseConfig, SUBSET_CONFIG
from src.system.metrics import all_metrics
from src.step1_preprocessing.loader import load_data
from src.step3_models.person2_model import MSPatchiTransformerRUL
from src.step3_models.trainer import Trainer


PATCH_CONFIGS = [
    # (patch_len, stride, label)
    (30, 30, "P0-whole-window"),   # 整窗单 patch，≈无 patch
    (8,  4,  "P1-small-50pct"),    # 小尺度 50% overlap
    (16, 8,  "P2-medium-50pct"),   # 中尺度 50% overlap（默认）
    (32, 16, "P3-large-50pct"),    # 大尺度 50% overlap
    (16, 4,  "P4-medium-75pct"),   # 中尺度 75% overlap（密集）
    (16, 12, "P5-medium-25pct"),   # 中尺度 25% overlap（稀疏）
]


def run_one(patch_len, stride, label):
    cfg = BaseConfig(**SUBSET_CONFIG["FD001"])
    # 单分支 + 指定 patch 参数
    cfg.branch_indices = (0,)
    cfg.fusion_mode = "concat"
    cfg.epochs = 200

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.num_workers = 0

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

    # 用自定义 patch 配置覆盖模型默认
    MSPatchiTransformerRUL.PATCH_CONFIGS = [(patch_len, stride)]
    model = MSPatchiTransformerRUL(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(f"PATCH ABLATION: {label} (patch={patch_len}, stride={stride})")
    print(f"  params={n_params:,}")
    trainer = Trainer(model, cfg, cfg.device)
    history = trainer.fit(X_tr, y_tr, X_val, y_val)
    trainer.load_best()
    y_pred = trainer.predict(X_test)
    metrics = all_metrics(y_test, y_pred)
    print(f">>> {label}: RMSE={metrics['RMSE']:.2f}, R²={metrics['R2']:.3f}, Score={metrics['Score']:.1f}")

    prefix = f"results/ablation/ablation_patch_{label}"
    with open(f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f)
    np.save(f"{prefix}_y_pred.npy", y_pred)
    np.save(f"{prefix}_y_test.npy", y_test)
    if history:
        np.savez(f"{prefix}_history.npz",
                 **{k: np.array(v) for k, v in history.items()})
    return metrics, n_params, y_pred, cfg


results = []
for patch_len, stride, label in PATCH_CONFIGS:
    m, p, yp, cfg = run_one(patch_len, stride, label)
    results.append({
        "experiment": label,
        "patch_len": patch_len,
        "stride": stride,
        "overlap": f"{int((1 - stride/patch_len)*100)}%",
        "window": cfg.window_size,
        "RMSE": round(m["RMSE"], 2),
        "R2": round(m["R2"], 4),
        "Score": round(m["Score"], 1),
        "params": p,
    })

# 保存 CSV
csv_path = "results/ablation/ablation_patch_summary.csv"
fieldnames = ["experiment", "patch_len", "stride", "overlap", "window",
              "RMSE", "R2", "Score", "params"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in results:
        w.writerow(r)

print(f"\n{'='*60}")
print("PATCH DESIGN ABLATION SUMMARY (FD001)")
print(f"{'='*60}")
print(f"{'Experiment':20s} {'P':>3s} {'S':>3s} {'Overlap':>8s} {'RMSE':>8s} {'R²':>7s} {'Score':>10s} {'Params':>10s}")
print("-"*70)
for r in sorted(results, key=lambda x: x["RMSE"]):
    print(f"{r['experiment']:20s} {r['patch_len']:3d} {r['stride']:3d} {r['overlap']:>8s} "
          f"{r['RMSE']:>8.2f} {r['R2']:>7.3f} {r['Score']:>10.1f} {r['params']:>10,}")
print(f"\nResults saved to {csv_path}")
