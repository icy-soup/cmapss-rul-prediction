"""Ablation comparison figures for Person 2 — English labels only."""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIGS = Path("results/figures"); FIGS.mkdir(exist_ok=True)
FIGS_CH = FIGS / "消融实验"; FIGS_CH.mkdir(exist_ok=True)
BEST = "#c44e52"

# ============ Data ============
b_labels = ["P8S4\nsmall", "P16S8\nmed", "P32S16\nlarge", "s+m", "s+l", "m+l", "3-branch"]
b_rmse  = [15.90, 16.06, 15.68, 13.87, 14.16, 14.98, 14.58]
b_r2    = [0.854, 0.851, 0.858, 0.889, 0.884, 0.870, 0.877]
b_pm    = ["2.21M","1.95M","1.82M","2.60M","2.74M","2.22M","3.14M"]
b_colors= ["#7fbf7f","#7fbf7f","#7fbf7f","#4C72B0","#4C72B0","#4C72B0","#55A868"]

f_labels = ["Concat", "Weighted\nSum", "Gated\nFusion"]
f_rmse  = [14.82, 15.52, 15.36]
f_r2    = [0.873, 0.861, 0.863]

d_labels = ["P8S4", "P16S8", "P8S4+P16S8", "3-branch", "Base"]
d_rmse  = [28.90, 26.97, 25.30, 28.61, 26.24]
d_r2    = [0.719, 0.755, 0.785, 0.725, 0.768]
d_pm    = ["2.53M","2.14M","3.13M","3.79M","1.95M"]
d_colors= ["#7fbf7f","#7fbf7f","#c44e52","#55A868","#4C72B0"]

# ============ Fig 1: Branch count ============
print("fig_ablation_branches.png...")
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(b_labels, b_rmse, color=b_colors, edgecolor="white", alpha=0.85)
bars[3].set_color(BEST); bars[3].set_alpha(1.0)
for i,(b,v,r2) in enumerate(zip(bars,b_rmse,b_r2)):
    is_best = (i == 3)
    best_tag = " ★ BEST" if is_best else ""
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.9, f"RMSE={v:.2f}",
            ha="center", fontsize=6, fontweight="bold" if is_best else "normal",
            color=BEST if is_best else "black")
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.35, f"R²={r2:.3f}{best_tag}",
            ha="center", fontsize=6, fontweight="bold" if is_best else "normal",
            color=BEST if is_best else "black")
    ax.annotate(b_pm[i], (b.get_x()+b.get_width()/2, b.get_height()/2),
                ha="center", fontsize=6.5, color="white", alpha=0.85)
ax.set_ylabel("RMSE (lower is better)", fontsize=11)
ax.set_title("Ablation A: Branch Configuration — FD001", fontsize=13)
ax.set_ylim(0, max(b_rmse)*1.7)
ax.axhline(y=17.93, color="#4C72B0", linestyle=":", linewidth=1, alpha=0.5, label="Base (single-scale)")
ax.legend(fontsize=8, loc="upper right", framealpha=0.8); ax.grid(axis="y", alpha=0.2)
plt.tight_layout(); fig.savefig(FIGS_CH/"分支数量消融.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ============ Fig 2: Fusion method ============
print("fig_ablation_fusion.png...")
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(f_labels, f_rmse, color=["#55A868","#DD8452","#4C72B0"], edgecolor="white", alpha=0.85)
bars[0].set_color(BEST); bars[0].set_alpha(1.0)
for i,(b,v,r2) in enumerate(zip(bars,f_rmse,f_r2)):
    is_best = (i == 0)
    best_tag = " ★ BEST" if is_best else ""
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.7, f"RMSE={v:.2f}",
            ha="center", fontsize=6.5, fontweight="bold" if is_best else "normal",
            color=BEST if is_best else "black")
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"R²={r2:.3f}{best_tag}",
            ha="center", fontsize=6.5, fontweight="bold" if is_best else "normal",
            color=BEST if is_best else "black")
ax.set_ylabel("RMSE (lower is better)", fontsize=11)
ax.set_title("Ablation B: Fusion Method — FD001", fontsize=13)
ax.set_ylim(0, max(f_rmse)*1.5); ax.grid(axis="y", alpha=0.2)
plt.tight_layout(); fig.savefig(FIGS_CH/"融合方式消融.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ============ Fig 3: FD004 validation ============
print("fig_ablation_fd004.png...")
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(d_labels, d_rmse, color=d_colors, edgecolor="white", alpha=0.85, width=0.6)
bars[2].set_color(BEST); bars[2].set_alpha(1.0)
for i,(b,v,r2) in enumerate(zip(bars,d_rmse,d_r2)):
    is_best = (i == 2)
    best_tag = " ★ BEST" if is_best else ""
    x_pos = b.get_x()+b.get_width()/2
    ax.text(x_pos, b.get_height()+1.8, f"RMSE={v:.2f}",
            ha="center", fontsize=6, fontweight="bold" if is_best else "normal", color=BEST if is_best else "black")
    ax.text(x_pos, b.get_height()+0.8, f"R²={r2:.3f}{best_tag}",
            ha="center", fontsize=6, fontweight="bold" if is_best else "normal", color=BEST if is_best else "black")
    ax.annotate(d_pm[i], (b.get_x()+b.get_width()/2, b.get_height()/2),
                ha="center", fontsize=6.5, color="white", alpha=0.85)
    if v > 50:
        ax.text(b.get_x()+b.get_width()/2, v*0.5, "COLLAPSED",
                ha="center", fontsize=7, color="red", fontweight="bold")
ax.set_ylabel("RMSE (lower is better)", fontsize=11)
ax.set_title("FD004 Ablation Validation", fontsize=13)
ax.set_ylim(0, max(d_rmse)*1.45); ax.grid(axis="y", alpha=0.2)
plt.tight_layout(); fig.savefig(FIGS_CH/"FD004验证.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ============ Fig 4: FD001 vs FD004 对比 ============
print("fig_ablation_summary.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# FD001: 全部分支配置
fd001_configs = ["small", "med", "large", "s+m", "s+l", "m+l", "3-br"]
fd001_rmse = [15.90, 16.06, 15.68, 13.87, 14.16, 14.98, 14.58]

# FD004: 只显示实际测试的配置
fd004_configs = ["small", "med", "s+m", "3-br"]
fd004_rmse = [28.90, 26.97, 25.30, 28.61]

ax1.bar(range(7), fd001_rmse, color=["#7fbf7f","#7fbf7f","#7fbf7f","#4C72B0","#4C72B0","#4C72B0","#55A868"],
        edgecolor="white", alpha=0.85)
ax1.set_xticks(range(7)); ax1.set_xticklabels(fd001_configs, fontsize=8)
ax1.set_ylabel("RMSE", fontsize=10); ax1.set_title("FD001（全部配置）", fontsize=12)
ax1.axhline(y=19.83, color="gray", linestyle=":", alpha=0.5, label="Base")
ax1.legend(fontsize=7, framealpha=0.8); ax1.grid(axis="y", alpha=0.2)
ax1.set_ylim(0, max(fd001_rmse)*1.4)

ax2.bar(range(4), fd004_rmse, color=["#7fbf7f","#7fbf7f","#c44e52","#55A868"],
        edgecolor="white", alpha=0.85, width=0.5)
ax2.set_xticks(range(4)); ax2.set_xticklabels(fd004_configs, fontsize=8)
ax2.set_ylabel("RMSE", fontsize=10); ax2.set_title("FD004（已测试配置）", fontsize=12)
ax2.axhline(y=26.24, color="#4C72B0", linestyle=":", alpha=0.5, label="Base")
ax2.axhline(y=28.61, color="#55A868", linestyle="--", alpha=0.5, label="3-branch")
ax2.legend(fontsize=7, framealpha=0.8); ax2.grid(axis="y", alpha=0.2)
ax2.set_ylim(0, max(fd004_rmse)*1.4)
ax2.text(2, fd004_rmse[2]+0.5, "× BEST", ha="center", fontsize=7, fontweight="bold", color=BEST)

fig.suptitle("FD001 与 FD004 最优分支配置对比", fontsize=14, y=1.05)
plt.subplots_adjust(top=0.82, wspace=0.3)
fig.savefig(FIGS_CH/"消融总结.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("Done!")
