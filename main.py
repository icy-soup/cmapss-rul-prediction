import argparse
import json, os
import numpy as np
import torch

from src.system.config import BaseConfig, SUBSET_CONFIG
from src.system.metrics import all_metrics
from src.step1_preprocessing.loader import load_data
from src.step3_models.base_model import PatchiTransformerRUL
from src.step3_models.person1_model import ConvPatchiTransformerRUL
from src.step3_models.person2_model import MSPatchiTransformerRUL
from src.step3_models.trainer import Trainer

MODEL_MAP = {
    "base": PatchiTransformerRUL,
    "person1": ConvPatchiTransformerRUL,
    "person2": MSPatchiTransformerRUL,
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_results(model_name, subset, cfg, history, y_test, y_pred, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"{save_dir}/{model_name}_{subset}"

    # Metrics
    metrics = all_metrics(y_test, y_pred)
    with open(f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions
    np.save(f"{prefix}_y_test.npy", y_test)
    np.save(f"{prefix}_y_pred.npy", y_pred)

    # Training history
    if history:
        np.savez(f"{prefix}_history.npz", **{k: np.array(v) for k, v in history.items()})

    # Config
    with open(f"{prefix}_config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, (list, tuple)) and all(isinstance(x, int) for x in v) else v
                    for k, v in cfg.__dict__.items() if not k.startswith("_")},
                  f, indent=2, default=str)

    print(f"\nResults saved to {prefix}_*")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base", choices=list(MODEL_MAP))
    parser.add_argument("--subset", type=str, default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()

    cfg = BaseConfig(**SUBSET_CONFIG[args.subset])
    cfg.device = args.device
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr
    if args.workers is not None: cfg.num_workers = args.workers
    set_seed(cfg.seed)

    model_cls = MODEL_MAP[args.model]
    model_name = model_cls.__name__
    print(f"=== {model_name} on {cfg.subset} ===")
    print(f"Window={cfg.window_size}, d_model={cfg.d_model}, layers={cfg.e_layers}, heads={cfg.n_heads}")

    # Data
    print("\nLoading data...")
    X_train, y_train, unit_ids, X_test, y_test = load_data(cfg)
    print(f"Train: {len(X_train)}, Test engines: {len(X_test)}, Shape: {X_train.shape[1:]}")

    # Split train/val by engine to avoid data leakage
    unique_engines = np.unique(unit_ids)
    np.random.seed(cfg.seed)
    np.random.shuffle(unique_engines)
    n_val_engines = max(1, int(len(unique_engines) * (1 - cfg.train_ratio)))
    val_engines = set(unique_engines[:n_val_engines])
    train_engines = set(unique_engines[n_val_engines:])

    val_mask = np.array([uid in val_engines for uid in unit_ids])
    X_val, y_val = X_train[val_mask], y_train[val_mask]
    X_tr, y_tr = X_train[~val_mask], y_train[~val_mask]
    print(f"Train/Val: {len(X_tr)}/{len(X_val)} (engines: {len(train_engines)}/{len(val_engines)})")

    # Model
    model = model_cls(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Train
    trainer = Trainer(model, cfg, cfg.device)
    history = trainer.fit(X_tr, y_tr, X_val, y_val)

    # Evaluate
    trainer.load_best()
    y_pred = trainer.predict(X_test)

    # Save & print
    metrics = save_results(model_name, cfg.subset, cfg, history, y_test, y_pred, args.save_dir)
    print(f"\n=== {cfg.subset} {model_name} Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    for i in range(min(10, len(y_test))):
        print(f"  Engine {i+1:3d}: true={y_test[i]:5.1f}, pred={y_pred[i]:5.1f}")

    return metrics


if __name__ == "__main__":
    main()
