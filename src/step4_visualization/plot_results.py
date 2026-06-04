"""CMAPSS 预测结果可视化，生成两套图：全模型对比 + Base vs Person 2 精简版"""
import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 中文字体支持
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path("results")
FIGS    = Path("results/figures")
FIGS.mkdir(exist_ok=True)

def _find(pattern: str) -> Path:
    p = RESULTS / pattern
    if p.exists(): return p
    for sub in ["main_experiments", "ablation"]:
        p = RESULTS / sub / pattern
        if p.exists(): return p
    return RESULTS / pattern

# 两套模型配置
ALL_MODELS = {
    "PatchiTransformerRUL":    ("Base",      "#4C72B0"),
    "ConvPatchiTransformerRUL": ("Person 1", "#DD8452"),
    "MSPatchiTransformerRUL":  ("Person 2", "#55A868"),
}
BASE_P2_MODELS = {k: v for k, v in ALL_MODELS.items() if "Person 1" not in v[0]}

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

def load_history(model_name, subset):
    path = _find(f"{model_name}_{subset}_history.npz")
    if not path.exists(): return None
    data = np.load(path)
    return {k: data[k].tolist() for k in data.files}

def load_metrics(model_name, subset):
    path = _find(f"{model_name}_{subset}_metrics.json")
    if not path.exists(): return None
    with open(path) as f: return json.load(f)

def load_preds(model_name, subset):
    y_test = _find(f"{model_name}_{subset}_y_test.npy")
    y_pred = _find(f"{model_name}_{subset}_y_pred.npy")
    if not y_test.exists() or not y_pred.exists(): return None, None
    return np.load(y_test), np.load(y_pred)


def plot_training_curves(models, save_path, ncols, figsize):
    """训练曲线：rows=4子集，cols=ncols模型"""
    fig, axes = plt.subplots(4, ncols, figsize=figsize)
    fig.suptitle("Training & Validation Curves", fontsize=14, y=0.98)
    model_items = list(models.items())
    for col_idx, (model_name, (model_label, color)) in enumerate(model_items):
        for row_idx, subset in enumerate(SUBSETS):
            ax = axes[row_idx][col_idx]
            hist = load_history(model_name, subset)
            if hist is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{model_label}\n{subset}", fontsize=9)
                continue
            epochs = range(1, len(hist["train_loss"]) + 1)
            ax.semilogy(epochs, hist["train_loss"], label="Train Loss", color=color, linewidth=0.8)
            if "val_loss" in hist and hist["val_loss"]:
                ax.semilogy(epochs, hist["val_loss"], label="Val Loss", color=color, linestyle="--", linewidth=0.8)
            if "val_rmse" in hist and hist["val_rmse"]:
                ax2 = ax.twinx()
                ax2.plot(epochs, hist["val_rmse"], label="Val RMSE", color="gray", linewidth=0.7, linestyle=":")
                ax2.set_ylabel("RMSE", fontsize=7, color="gray")
                ax2.tick_params(labelsize=6, colors="gray")
                final_rmse = hist["val_rmse"][-1]
                ax2.text(0.95, 0.95, f"RMSE={final_rmse:.1f}", transform=ax2.transAxes,
                         fontsize=7, color="gray", ha="right", va="top",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
            ax.set_xlabel("Epoch", fontsize=7)
            ax.set_ylabel("Loss (log)", fontsize=7)
            ax.set_title(f"{model_label}\n{subset}", fontsize=9)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5, loc="lower left", framealpha=0.7)
            ax.grid(True, alpha=0.15)
            ax.set_ylim(bottom=max(1, min(hist["train_loss"]) * 0.5))
    plt.subplots_adjust(left=0.06, right=0.92, top=0.95, bottom=0.04, hspace=0.25, wspace=0.25)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}")


def plot_scatter(models, save_path, ncols, figsize):
    """散点图 + 残差直方图"""
    fig, axes = plt.subplots(4, ncols, figsize=figsize)
    fig.suptitle("Predicted vs True RUL", fontsize=14, y=0.98)
    model_items = list(models.items())
    for col_idx, (model_name, (model_label, color)) in enumerate(model_items):
        for row_idx, subset in enumerate(SUBSETS):
            ax = axes[row_idx][col_idx]
            y_test, y_pred = load_preds(model_name, subset)
            if y_test is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{model_label} — {subset}", fontsize=9)
                continue
            metrics = load_metrics(model_name, subset)
            rmse_val = metrics["RMSE"] if metrics else 0
            r2_val   = metrics["R2"]   if metrics else 0
            score    = metrics["Score"] if metrics else 0
            errors = y_pred - y_test
            bias = errors.mean()
            e_std = errors.std()
            mae = np.abs(errors).mean()
            ax.scatter(y_test, y_pred, s=6, alpha=0.5, color=color, edgecolors="none", rasterized=True)
            lims = [min(y_test.min(), y_pred.min()) - 5, min(max(y_test.max(), y_pred.max()) + 5, 135)]
            ax.plot(lims, lims, "k--", linewidth=0.6, alpha=0.4)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_xlabel("True RUL", fontsize=7); ax.set_ylabel("Predicted RUL", fontsize=7)
            stats_text = (
                f"RMSE={rmse_val:.1f}  R²={r2_val:.3f}\n"
                f"Bias={bias:+.1f}  σ={e_std:.1f}\n"
                f"MAE={mae:.1f}  Score={score:.0f}"
            )
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=6.5,
                    ha="left", va="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            ax.set_title(f"{model_label} — {subset}", fontsize=9)
            ax.set_aspect("equal"); ax.tick_params(labelsize=6); ax.grid(alpha=0.15)
            ax_hist = ax.inset_axes([0.55, 0.05, 0.4, 0.25])
            ax_hist.hist(errors, bins=30, color=color, alpha=0.6, edgecolor="white", linewidth=0.3)
            ax_hist.axvline(0, color="black", linestyle="--", linewidth=0.5)
            ax_hist.set_xlabel("Residual", fontsize=5); ax_hist.set_ylabel("Count", fontsize=5)
            ax_hist.tick_params(labelsize=5)
    plt.subplots_adjust(left=0.06, right=0.95, top=0.96, bottom=0.04, hspace=0.25, wspace=0.25)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}")


def plot_comparison(models, save_path):
    """RMSE + Score 对比柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison Across CMAPSS Subsets", fontsize=14, y=1.08)
    x = np.arange(len(SUBSETS))
    n_models = len(models)
    width = 0.25 if n_models >= 3 else 0.3
    model_items = list(models.items())
    offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)
    for i, (model_name, (model_label, color)) in enumerate(model_items):
        rmses = []; scores = []
        for subset in SUBSETS:
            m = load_metrics(model_name, subset)
            rmses.append(m["RMSE"] if m else 0)
            scores.append(m["Score"] if m else 0)
        offset = offsets[i]
        bars = ax1.bar(x + offset, rmses, width, label=model_label, color=color, alpha=0.85)
        for bar, val in zip(bars, rmses):
            if val > 0:
                ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=6.5)
        bars = ax2.bar(x + offset, scores, width, label=model_label, color=color, alpha=0.85)
        for bar, val in zip(bars, scores):
            if val > 0:
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(100, val*0.05),
                         f"{val:.0f}", ha="center", va="bottom", fontsize=5, rotation=45)
    ax1.set_ylim(0, max(ax1.get_ylim()[1]*1.25, 45))
    ax2.set_ylim(0, max(ax2.get_ylim()[1]*1.25, 200000))
    ax1.set_xticks(x); ax1.set_xticklabels(SUBSETS, fontsize=10)
    ax1.set_ylabel("RMSE ↓ (lower is better)", fontsize=10)
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)
    ax2.set_xticks(x); ax2.set_xticklabels(SUBSETS, fontsize=10)
    ax2.set_ylabel("PHM Score ↓ (lower is better)", fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    fig.text(0.5, 0.005, "Note: PHM Score is an error metric — LOWER is better (like RMSE)",
             ha="center", fontsize=8, fontstyle="italic", color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}")


def plot_per_engine(models, save_path, ncols, figsize):
    """全引擎排序预测"""
    fig, axes = plt.subplots(4, ncols, figsize=figsize)
    fig.suptitle("All Test Engines — Sorted by True RUL", fontsize=14, y=0.98)
    model_items = list(models.items())
    for col_idx, (model_name, (model_label, color)) in enumerate(model_items):
        for row_idx, subset in enumerate(SUBSETS):
            ax = axes[row_idx][col_idx]
            y_test, y_pred = load_preds(model_name, subset)
            if y_test is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=12)
                ax.set_title(f"{model_label} — {subset}", fontsize=9)
                continue
            idx = np.argsort(y_test)
            x_vals = np.arange(len(y_test))
            n_eng = len(y_test)
            metrics = load_metrics(model_name, subset)
            rmse_val = metrics["RMSE"] if metrics else 0
            ax.fill_between(x_vals, y_test[idx], alpha=0.1, color="black")
            ax.plot(x_vals, y_test[idx], "o", color="black", markersize=1.2, alpha=0.4, label="True")
            ax.plot(x_vals, y_pred[idx], "x", color=color, markersize=1.2, alpha=0.4, label="Pred")
            pred_range = y_pred.max() - y_pred.min()
            bias = (y_pred - y_test).mean()
            stats = f"RMSE={rmse_val:.1f}  bias={bias:+.1f}\npred_range=[{y_pred.min():.0f},{y_pred.max():.0f}]"
            ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=6.5,
                    ha="left", va="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
            if subset == "FD004" and pred_range < 10:
                ax.text(0.5, 0.5, "⚠️ Prediction collapsed\n(model predicts near-constant)",
                        transform=ax.transAxes, fontsize=8, ha="center", va="center",
                        color="red", fontweight="bold",
                        bbox=dict(boxstyle="round", fc="yellow", alpha=0.7))
            ax.set_xlabel(f"Engine (sorted, n={n_eng})", fontsize=7)
            ax.set_ylabel("RUL", fontsize=7)
            ax.set_title(f"{model_label} — {subset}", fontsize=9)
            ax.legend(fontsize=5, loc="lower right", markerscale=0.8)
            ax.tick_params(labelsize=6); ax.grid(alpha=0.15)
        y_min = min(ax.get_ylim()[0] for ax in axes[row_idx])
        y_max = max(ax.get_ylim()[1] for ax in axes[row_idx])
        for ax in axes[row_idx]:
            ax.set_ylim(y_min, y_max)
    plt.subplots_adjust(left=0.06, right=0.95, top=0.96, bottom=0.04, hspace=0.25, wspace=0.25)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}")


# ===== 主流程 =====
OUT_DIRS = {
    "full": FIGS / "full",
    "base_p2": FIGS,
}

for mode, models in [("full", ALL_MODELS), ("base_p2", BASE_P2_MODELS)]:
    out = OUT_DIRS[mode]
    out.mkdir(exist_ok=True)
    n = len(models)
    print(f"\n===== 生成 {mode} 版（{n} 个模型）=====")

    if n == 3:
        plot_training_curves(models, out / "fig_training.png", 3, (16, 16))
        plot_scatter(models, out / "fig_scatter.png", 3, (16, 18))
        plot_per_engine(models, out / "fig_per_engine.png", 3, (16, 20))
    else:
        plot_training_curves(models, out / "fig_training.png", 2, (12, 16))
        plot_scatter(models, out / "fig_scatter.png", 2, (12, 18))
        plot_per_engine(models, out / "fig_per_engine.png", 2, (12, 20))

    # 对比图：full 版放 full/，base_p2 版放根目录
    if mode == "full":
        plot_comparison(models, out / "fig_comparison.png")
    else:
        plot_comparison(models, FIGS / "fig_comparison.png")

print("\n全部图片生成完毕")
print(f"  figures/full/  — 全模型版（3 模型 × 4 子集 = 12 子图）")
print(f"  figures/       — Base+P2 精简版（2 模型 × 4 子集 = 8 子图）")

# ===== 额外生成每子集独立图，中文文件夹分类 =====
print("\n===== 生成独立子集图（中文分类）=====")

SUB_NAMES = {"FD001": "FD001", "FD002": "FD002", "FD003": "FD003", "FD004": "FD004"}
FOLDERS = {
    "训练曲线": plot_training_curves,
    "散点图": plot_scatter,
    "引擎预测": plot_per_engine,
}
FIGS_CH = FIGS / "按子集"

for mode, model_dict in [("全模型", ALL_MODELS), ("精简(Base+P2)", BASE_P2_MODELS)]:
    n = len(model_dict)
    fs = (8, 4) if n == 2 else (12, 4)  # figsize per subset
    nc = n  # number of columns = number of models

    for ch_name, plot_fn in FOLDERS.items():
        for subset in SUBSETS:
            # 临时替换 SUBSETS 为当前子集
            import copy
            old_models = globals().get("_current_models", None)

            # 画单子集图：把模型放列，只有一行
            fig, axes = plt.subplots(1, nc, figsize=fs)
            fig.suptitle(f"{ch_name} — {subset}", fontsize=14, y=1.02)
            model_items = list(model_dict.items())

            for col_idx, (model_name, (model_label, color)) in enumerate(model_items):
                ax = axes[col_idx] if nc > 1 else axes

                if ch_name == "训练曲线":
                    hist = load_history(model_name, subset)
                    if hist is None:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(model_label, fontsize=10)
                        continue
                    epochs = range(1, len(hist["train_loss"]) + 1)
                    ax.semilogy(epochs, hist["train_loss"], label="Train Loss", color=color, linewidth=0.8)
                    if "val_loss" in hist and hist["val_loss"]:
                        ax.semilogy(epochs, hist["val_loss"], label="Val Loss", color=color, linestyle="--", linewidth=0.8)
                    if "val_rmse" in hist and hist["val_rmse"]:
                        ax2 = ax.twinx()
                        ax2.plot(epochs, hist["val_rmse"], color="gray", linewidth=0.7, linestyle=":")
                        ax2.set_ylabel("RMSE", fontsize=8, color="gray")
                        ax2.tick_params(labelsize=7, colors="gray")
                        ax2.text(0.95, 0.95, f"RMSE={hist['val_rmse'][-1]:.1f}", transform=ax2.transAxes,
                                 fontsize=8, color="gray", ha="right", va="top",
                                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
                    ax.set_xlabel("Epoch", fontsize=8)
                    ax.set_ylabel("Loss (log)", fontsize=8)
                    ax.set_title(model_label, fontsize=10)
                    ax.legend(fontsize=6, loc="lower left")
                    ax.grid(True, alpha=0.15)

                elif ch_name == "散点图":
                    y_test, y_pred = load_preds(model_name, subset)
                    if y_test is None:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(model_label, fontsize=10)
                        continue
                    metrics = load_metrics(model_name, subset)
                    rmse_val = metrics["RMSE"] if metrics else 0
                    r2_val = metrics["R2"] if metrics else 0
                    score = metrics["Score"] if metrics else 0
                    errors = y_pred - y_test
                    ax.scatter(y_test, y_pred, s=8, alpha=0.5, color=color, edgecolors="none", rasterized=True)
                    lims = [min(y_test.min(), y_pred.min())-5, min(max(y_test.max(), y_pred.max())+5, 135)]
                    ax.plot(lims, lims, "k--", linewidth=0.6, alpha=0.4)
                    ax.set_xlim(lims); ax.set_ylim(lims)
                    ax.set_xlabel("True RUL", fontsize=8); ax.set_ylabel("Predicted RUL", fontsize=8)
                    stats = f"RMSE={rmse_val:.1f}  R²={r2_val:.3f}\nBias={errors.mean():+.1f}  MAE={np.abs(errors).mean():.1f}\nScore={score:.0f}"
                    ax.text(0.03, 0.95, stats, transform=ax.transAxes, fontsize=7,
                            ha="left", va="top", fontfamily="monospace",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                    ax.set_title(model_label, fontsize=10)
                    ax.set_aspect("equal"); ax.grid(alpha=0.15)

                elif ch_name == "引擎预测":
                    y_test, y_pred = load_preds(model_name, subset)
                    if y_test is None:
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(model_label, fontsize=10)
                        continue
                    idx = np.argsort(y_test)
                    x_vals = np.arange(len(y_test))
                    metrics = load_metrics(model_name, subset)
                    rmse_val = metrics["RMSE"] if metrics else 0
                    ax.fill_between(x_vals, y_test[idx], alpha=0.1, color="black")
                    ax.plot(x_vals, y_test[idx], "o", color="black", markersize=2, alpha=0.4, label="True")
                    ax.plot(x_vals, y_pred[idx], "x", color=color, markersize=2, alpha=0.4, label="Pred")
                    bias = (y_pred - y_test).mean()
                    stats = f"RMSE={rmse_val:.1f}  bias={bias:+.1f}"
                    ax.text(0.03, 0.95, stats, transform=ax.transAxes, fontsize=7,
                            ha="left", va="top", fontfamily="monospace",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
                    ax.set_xlabel(f"Engine (n={len(y_test)})", fontsize=8)
                    ax.set_ylabel("RUL", fontsize=8)
                    ax.set_title(model_label, fontsize=10)
                    ax.legend(fontsize=6); ax.grid(alpha=0.15)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_dir = FIGS_CH / mode / ch_name
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / f"{subset}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  -> {save_dir / subset}")

print("\n所有独立子图生成完毕")
print(f"  figures/按子集/全模型/   — 3 模型版，每子集独立图")
print(f"  figures/按子集/精简(Base+P2)/  — 2 模型版，每子集独立图")
