import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.cluster import KMeans
from src.system.config import BaseConfig


COL_NAMES = [
    "unit", "cycle",
    "altitude", "mach", "tra",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",
]

OP_COND_COLS = ["altitude", "mach", "tra"]


def load_raw(cfg: BaseConfig):
    data_dir = Path(cfg.data_dir)
    train = pd.read_csv(data_dir / f"train_{cfg.subset}.txt", sep=r"\s+", header=None, names=COL_NAMES)
    test = pd.read_csv(data_dir / f"test_{cfg.subset}.txt", sep=r"\s+", header=None, names=COL_NAMES)
    rul_true = pd.read_csv(data_dir / f"RUL_{cfg.subset}.txt", sep=r"\s+", header=None).values.flatten()
    return train, test, rul_true


# ── 全局归一化 ──

def compute_normalization_stats(df: pd.DataFrame):
    """算出每个传感器在全数据集上的均值和标准差"""
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    return {c: (df[c].mean(), df[c].std()) for c in sensor_cols}


def apply_normalization(df: pd.DataFrame, stats: dict):
    """用事先算好的统计量做 z-score 标准化"""
    df = df.copy()
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    df[sensor_cols] = df[sensor_cols].astype("float64")
    for c in sensor_cols:
        mean, std = stats[c]
        df[c] = (df[c] - mean) / (std + 1e-8)
    return df


# ── 按工况聚类归一化 ──

def _add_condition_id(df: pd.DataFrame, condition_map: dict = None) -> Tuple[pd.DataFrame, dict]:
    """用 KMeans 给每行数据打上工况簇标签"""
    # FD002 和 FD004 在 altitude/mach/tra 上有六个离散工况点
    # FD001 和 FD003 只有单工况，所以 KMeans 聚类数设为 1
    op_data = df[OP_COND_COLS].values.astype(float)
    if condition_map is None:
        std_per_col = df[OP_COND_COLS].std()
        n_clusters = 1 if std_per_col.max() < 0.01 else 6
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(op_data)
        condition_map = {"kmeans": kmeans, "n_clusters": n_clusters}
    else:
        kmeans = condition_map["kmeans"]
        cluster_ids = kmeans.predict(op_data)
    df = df.copy()
    df["condition_id"] = cluster_ids.astype(int)
    return df, condition_map


def compute_normalization_stats_per_condition(df: pd.DataFrame) -> dict:
    """按工况簇分别算出各传感器的均值和标准差"""
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    stats = {}
    for cid, group in df.groupby("condition_id"):
        cid = int(cid)
        stats[cid] = {c: (group[c].mean(), group[c].std()) for c in sensor_cols}
    return stats


def apply_normalization_per_condition(df: pd.DataFrame, stats: dict):
    """用各工况自己的统计量做 z-score 标准化"""
    df = df.copy()
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    df[sensor_cols] = df[sensor_cols].astype("float64")
    for cid, group in df.groupby("condition_id"):
        cid = int(cid)
        for c in sensor_cols:
            mean, std = stats[cid][c]
            df.loc[group.index, c] = (df.loc[group.index, c] - mean) / (std + 1e-8)
    return df


# ── 工具函数 ──

def add_rul_labels(df: pd.DataFrame, rul_max: int = 125) -> pd.DataFrame:
    """给每个引擎加上从 0 到 rul_max 的分段线性 RUL 标签"""
    df = df.copy()
    rul = np.concatenate([
        g["cycle"].max() - g["cycle"].values
        for _, g in df.groupby("unit")
    ]).astype(float)
    df["rul"] = np.clip(rul, 0, rul_max)
    return df


def sliding_windows(df: pd.DataFrame, window_size: int, stride: int = 1, feature_cols: list = None):
    """对每个引擎做滑窗，返回样本、标签和对应的引擎编号"""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("s")]
    X, y, uids = [], [], []
    for _, g in df.groupby("unit"):
        vals = g[feature_cols].values
        ruls = g["rul"].values
        uid = g["unit"].iloc[0]
        for i in range(0, len(vals) - window_size + 1, stride):
            X.append(vals[i : i + window_size])
            y.append(ruls[i + window_size - 1])
            uids.append(uid)
    return np.array(X), np.array(y), np.array(uids)


# ── 主入口 ──

def load_data(cfg: BaseConfig):
    """加载 CMAPSS 数据并完成预处理

    global 模式对所有数据做全局 z-score 标准化。
    per_condition 模式先用 KMeans 识别工况簇，再按簇分别标准化。
    FD002 和 FD004 会拼接 altitude/mach/tra 作为额外输入通道。

    返回值包括滑窗后的训练样本和标签、训练样本的引擎编号、
    以及每个测试引擎最后一个窗口的样本和真实 RUL。
    """
    train_raw, test_raw, rul_true = load_raw(cfg)

    sensor_cols = [f"s{i}" for i in cfg.sensors]
    cols = ["unit", "cycle"] + OP_COND_COLS + sensor_cols

    train_df = train_raw[cols].copy()
    test_df = test_raw[cols].copy()

    # ── 归一化 ──
    if cfg.norm_mode == "per_condition":
        train_df, cond_map = _add_condition_id(train_df)
        norm_stats = compute_normalization_stats_per_condition(train_df)
        train_df = apply_normalization_per_condition(train_df, norm_stats)

        test_df, _ = _add_condition_id(test_df, cond_map)
        test_df = apply_normalization_per_condition(test_df, norm_stats)
    else:
        norm_stats = compute_normalization_stats(train_df)
        train_df = apply_normalization(train_df, norm_stats)
        test_df = apply_normalization(test_df, norm_stats)

    # ── 加 RUL 标签 ──
    train_df = add_rul_labels(train_df, cfg.rul_max)

    # ── 确定特征列 ──
    use_cond = cfg.use_op_cond and cfg.subset in ("FD002", "FD004")
    feature_cols = sensor_cols + (OP_COND_COLS if use_cond else [])

    # ── 对训练集做滑窗 ──
    X_train, y_train, unit_ids = sliding_windows(train_df, cfg.window_size, cfg.stride, feature_cols)

    # ── 对测试集取每个引擎最后一个窗口 ──
    X_test_list = []
    y_test_list = []
    for i, (_, g) in enumerate(test_df.groupby("unit")):
        vals = g[feature_cols].values
        n = len(vals)
        if n >= cfg.window_size:
            X_test_list.append(vals[-cfg.window_size:])
        else:
            pad = cfg.window_size - n
            X_test_list.append(np.pad(vals, ((pad, 0), (0, 0)), mode="edge"))
        y_test_list.append(min(rul_true[i], cfg.rul_max))

    return X_train, y_train, unit_ids, np.array(X_test_list), np.array(y_test_list)
