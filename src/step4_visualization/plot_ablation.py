"""Person 2 消融实验对比图（含 FD002/FD003 验证）"""
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

# ── 分支数量消融（FD001） ──
b_labels = ["P8S4\nsmall", "P16S8\nmed", "P32S16\nlarge", "s+m", "s+l", "m+l", "3-branch"]
b_rmse  = [15.90, 16.06, 15.68, 13.87, 14.16, 14.98, 14.58]
b_r2    = [0.854, 0.851, 0.858, 0.889, 0.884, 0.870, 0.877]
b_pm    = ["2.21M","1.95M","1.82M","2.60M","2.74M","2.22M","3.14M"]
b_colors= ["#7fbf7f","#7fbf7f","#7fbf7f","#4C72B0","#4C72B0","#4C72B0","#55A868"]

print("fig_ablation_branches.png...")
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(b_labels, b_rmse, color=b_colors, edgecolor="white", alpha=0.85)
bars[3].set_color(BEST); bars[3].set_alpha(1.0)
for i,(b,v,r2) in enumerate(zip(bars,b_rmse,b_r2)):
    is_best = (i == 3)
    best_tag = " BEST" if is_best else ""
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.9, f"RMSE={v:.2f}",
            ha="center", fontsize=6, fontweight="bold" if is_best else "normal",
            color=BEST if is_best else "black")
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.35, f"R²={r2:.3f}{best_tag}",
            ha="center", fontsize=6, fontweight="bold" if is_best else "normal",
            color=BEST if is_best else "black")
    ax.annotate(b_pm[i], (b.get_x()+b.get_width()/2, b.get_height()/2),
                ha="center", fontsize=6.5, color="white", alpha=0.85)
ax.set_ylabel("RMSE (lower is better)", fontsize=11)
ax.set_title("Ablation A: Branch Configuration - FD001", fontsize=13)
ax.set_ylim(0, max(b_rmse)*1.7)
ax.axhline(y=17.93, color="#4C72B0", linestyle=":", linewidth=1, alpha=0.5, label="Base (single-scale)")
ax.legend(fontsize=8, loc="upper right", framealpha=0.8); ax.grid(axis="y", alpha=0.2)
plt.tight_layout(); fig.savefig(FIGS_CH/"分支数量消融.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ── 融合方式消融（FD001，三分支） ──
f_labels = ["Concat", "Weighted\nSum", "Gated\nFusion"]
f_rmse  = [14.82, 15.52, 15.36]
f_r2    = [0.873, 0.861, 0.863]

print("fig_ablation_fusion.png...")
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(f_labels, f_rmse, color=["#55A868","#DD8452","#4C72B0"], edgecolor="white", alpha=0.85)
bars[0].set_color(BEST); bars[0].set_alpha(1.0)
for i,(b,v,r2) in enumerate(zip(bars,f_rmse,f_r2)):
    is_best = (i == 0)
    best_tag = " BEST" if is_best else ""
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.7, f"RMSE={v:.2f}",
            ha="center", fontsize=6.5, fontweight="bold" if is_best else "normal",
            color=BEST if is_best else "black")
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"R²={r2:.3f}{best_tag}",
            ha="center", fontsize=6.5, fontweight="bold" if is_best else "normal",
            color=BEST if is_best else "black")
ax.set_ylabel("RMSE (lower is better)", fontsize=11)
ax.set_title("Ablation B: Fusion Method - FD001", fontsize=13)
ax.set_ylim(0, max(f_rmse)*1.5); ax.grid(axis="y", alpha=0.2)
plt.tight_layout(); fig.savefig(FIGS_CH/"融合方式消融.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ── FD004 消融验证 ──
d_labels = ["P8S4", "P16S8", "P8S4+P16S8", "3-branch", "Base"]
d_rmse  = [25.11, 26.17, 24.97, 24.95, 28.81]
d_r2    = [0.780, 0.761, 0.783, 0.783, 0.711]
d_pm    = ["2.53M","2.14M","3.13M","3.79M","1.95M"]
d_colors= ["#7fbf7f","#7fbf7f","#4C72B0","#c44e52","#DD8452"]

print("fig_ablation_fd004.png...")
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(d_labels, d_rmse, color=d_colors, edgecolor="white", alpha=0.85, width=0.6)
bars[3].set_color(BEST); bars[3].set_alpha(1.0)
for i,(b,v,r2) in enumerate(zip(bars,d_rmse,d_r2)):
    is_best = (i == 3)
    best_tag = " BEST" if is_best else ""
    x_pos = b.get_x()+b.get_width()/2
    ax.text(x_pos, b.get_height()+1.8, f"RMSE={v:.2f}",
            ha="center", fontsize=6, fontweight="bold" if is_best else "normal", color=BEST if is_best else "black")
    ax.text(x_pos, b.get_height()+0.8, f"R²={r2:.3f}{best_tag}",
            ha="center", fontsize=6, fontweight="bold" if is_best else "normal", color=BEST if is_best else "black")
    ax.annotate(d_pm[i], (b.get_x()+b.get_width()/2, b.get_height()/2),
                ha="center", fontsize=6.5, color="white", alpha=0.85)
ax.set_ylabel("RMSE (lower is better)", fontsize=11)
ax.set_title("FD004 Ablation: Complex Multi-Condition", fontsize=13)
ax.set_ylim(0, max(d_rmse)*1.45); ax.grid(axis="y", alpha=0.2)
plt.tight_layout(); fig.savefig(FIGS_CH/"FD004验证.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ── 四子集消融总结对比 ──
print("fig_ablation_summary.png...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

subsets_data = {
    "FD001\n(1 cond, 1 fault)": {
        "configs": ["small", "med", "s+m", "3-br"],
        "rmse": [15.90, 16.06, 13.87, 15.27],
        "colors": ["#7fbf7f","#7fbf7f","#c44e52","#55A868"],
        "best_idx": 2,
        "base_line": 19.83,
    },
    "FD002\n(6 cond, 1 fault)": {
        "configs": ["small", "med", "s+m", "3-br"],
        "rmse": [26.52, 25.25, 25.81, 26.78],
        "colors": ["#7fbf7f","#c44e52","#4C72B0","#55A868"],
        "best_idx": 1,
        "base_line": 29.71,
    },
    "FD003\n(1 cond, 2 faults)": {
        "configs": ["small", "med", "s+m", "3-br"],
        "rmse": [12.98, 14.84, 13.60, 16.41],
        "colors": ["#c44e52","#7fbf7f","#4C72B0","#55A868"],
        "best_idx": 0,
        "base_line": 17.36,
    },
    "FD004\n(6 cond, 2 faults)": {
        "configs": ["small", "med", "s+m", "3-br"],
        "rmse": [25.11, 26.17, 24.97, 24.95],
        "colors": ["#7fbf7f","#7fbf7f","#4C72B0","#c44e52"],
        "best_idx": 3,
        "base_line": 28.81,
    },
}

for ax, (title, data) in zip(axes.flat, subsets_data.items()):
    bars = ax.bar(data["configs"], data["rmse"], color=data["colors"], edgecolor="white", alpha=0.85, width=0.5)
    bars[data["best_idx"]].set_color(BEST); bars[data["best_idx"]].set_alpha(1.0)
    for i, v in enumerate(data["rmse"]):
        is_best = (i == data["best_idx"])
        ax.text(i, v + max(0.3, v*0.03), f"{v:.2f}",
                ha="center", fontsize=10, fontweight="bold" if is_best else "normal",
                color=BEST if is_best else "black")
    ax.axhline(y=data["base_line"], color="gray", linestyle=":", alpha=0.5, linewidth=1, label=f"Base ({data['base_line']})")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("RMSE", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, max(data["rmse"])*1.4)

fig.suptitle("Ablation Across All Subsets: Optimal Configuration Depends on Data Complexity",
             fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(FIGS_CH/"消融总结.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("Done!")
