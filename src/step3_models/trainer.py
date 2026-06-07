import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.system.metrics import rmse, phm08_score, r2


class WarmupCosineScheduler:
    """Warmup + 余弦退火学习率调度器"""

    # 每个 epoch 前调用 step()，从近零开始线性上升到 peak_lr，再余弦衰减
    def __init__(self, optimizer, peak_lr: float, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.epoch = 0
        for group in self.optimizer.param_groups:
            group["lr"] = 1e-8  # 从近乎零的学习率开始热身

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            lr = self.peak_lr * self.epoch / self.warmup_epochs
        elif self.epoch <= self.total_epochs:
            progress = (self.epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.peak_lr * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            lr = 0.0
        for group in self.optimizer.param_groups:
            group["lr"] = lr


class Trainer:
    def __init__(self, model, cfg, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, peak_lr=cfg.lr, warmup_epochs=10, total_epochs=cfg.epochs,
        )
        self.criterion = torch.nn.MSELoss()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        cfg = self.cfg
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=cfg.num_workers > 0,
        )

        if X_val is not None:
            X_v = torch.tensor(X_val, dtype=torch.float32)
            y_v = torch.tensor(y_val, dtype=torch.float32)
            val_loader = DataLoader(
                TensorDataset(X_v, y_v),
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                persistent_workers=cfg.num_workers > 0,
            )

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_rmse": [], "val_score": [], "val_r2": []}

        for epoch in range(1, cfg.epochs + 1):
            self.scheduler.step()  # 每个 epoch 先更新学习率
            self.model.train()
            epoch_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(Xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            train_loss = epoch_loss / len(X_t)
            history["train_loss"].append(train_loss)

            if X_val is not None:
                val_loss, val_rmse, val_score, val_r2 = self._evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_rmse"].append(val_rmse)
                history["val_score"].append(val_score)
                history["val_r2"].append(val_r2)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if (epoch % 10 == 0) or (epoch == 1):
                    lr_now = self.optimizer.param_groups[0]["lr"]
                    print(f"Epoch {epoch:3d} | lr={lr_now:.2e} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_rmse={val_rmse:.2f} | val_score={val_score:.1f}")

                if patience_counter >= cfg.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f}")

        return history

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        losses, preds, targets = [], [], []
        for Xb, yb in loader:
            Xb, yb = Xb.to(self.device), yb.to(self.device)
            pred = self.model(Xb)
            losses.append(self.criterion(pred, yb).item() * len(Xb))
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
        avg_loss = sum(losses) / sum(len(t) for t in targets)
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        return avg_loss, rmse(targets, preds), phm08_score(targets, preds), r2(targets, preds)

    @torch.no_grad()
    def predict(self, X_test):
        self.model.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        preds = []
        batch_size = self.cfg.batch_size
        for i in range(0, len(X_t), batch_size):
            preds.append(self.model(X_t[i:i+batch_size]).cpu().numpy())
        return np.concatenate(preds)

    def load_best(self):
        if hasattr(self, "best_state"):
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_state.items()}
            )
