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
    """计算每个传感器的全局均值和标准差（从训练集）"""
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    return {c: (df[c].mean(), df[c].std()) for c in sensor_cols}


def apply_normalization(df: pd.DataFrame, stats: dict):
    """用预计算的统计量做 z-score 标准化"""
    df = df.copy()
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    df[sensor_cols] = df[sensor_cols].astype("float64")
    for c in sensor_cols:
        mean, std = stats[c]
        df[c] = (df[c] - mean) / (std + 1e-8)
    return df


# ── 按工况聚类归一化 ──

def _add_condition_id(df: pd.DataFrame, condition_map: dict = None) -> Tuple[pd.DataFrame, dict]:
    """用 KMeans 给每行分配工况簇 ID"""

    # FD002/FD004 在 (altitude, mach, tra) 上有 6 个分离的工况
    # FD001/FD003 单工况，n_clusters=1
    op_data = df[OP_COND_COLS].values.astype(float)
    if condition_map is None:
        # Detect if data is effectively constant (single condition)
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
    """Compute per-condition normalization stats from training set.

    Returns:
        stats[condition_id][sensor_name] = (mean, std)
    """
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    stats = {}
    for cid, group in df.groupby("condition_id"):
        cid = int(cid)
        stats[cid] = {c: (group[c].mean(), group[c].std()) for c in sensor_cols}
    return stats


def apply_normalization_per_condition(df: pd.DataFrame, stats: dict):
    """Apply per-condition normalization using pre-computed stats."""
    df = df.copy()
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    df[sensor_cols] = df[sensor_cols].astype("float64")
    for cid, group in df.groupby("condition_id"):
        cid = int(cid)
        for c in sensor_cols:
            mean, std = stats[cid][c]
            df.loc[group.index, c] = (df.loc[group.index, c] - mean) / (std + 1e-8)
    return df


# ── Common utilities ──

def add_rul_labels(df: pd.DataFrame, rul_max: int = 125) -> pd.DataFrame:
    """Add piecewise linear RUL labels capped at rul_max."""
    df = df.copy()
    rul = np.concatenate([
        g["cycle"].max() - g["cycle"].values
        for _, g in df.groupby("unit")
    ]).astype(float)
    df["rul"] = np.clip(rul, 0, rul_max)
    return df


def sliding_windows(df: pd.DataFrame, window_size: int, stride: int = 1, feature_cols: list = None):
    """Sliding window over each unit. Returns (X, y, unit_ids)."""
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


# ── Main entry point ──

def load_data(cfg: BaseConfig):
    """Load and preprocess CMAPSS data for a given subset.

    Normalization:
      - "global": original global z-score over all training data
      - "per_condition": z-score per operating condition group

    Features:
      - Always includes cfg.sensors (14 trend sensors)
      - When use_op_cond=True and FD002/FD004: includes altitude/mach/tra

    Returns:
        X_train, y_train: sliding window training samples
        unit_ids: engine IDs for training samples (for engine-based val split)
        X_test,  y_test:  last-window-per-engine test samples
    """
    train_raw, test_raw, rul_true = load_raw(cfg)

    # Select config-specified sensors (cfg.sensors is 1-based)
    sensor_cols = [f"s{i}" for i in cfg.sensors]
    cols = ["unit", "cycle"] + OP_COND_COLS + sensor_cols

    train_df = train_raw[cols].copy()
    test_df = test_raw[cols].copy()

    # ── Normalization ──
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

    # ── Add RUL labels ──
    train_df = add_rul_labels(train_df, cfg.rul_max)

    # ── Determine feature columns ──
    use_cond = cfg.use_op_cond and cfg.subset in ("FD002", "FD004")
    feature_cols = sensor_cols + (OP_COND_COLS if use_cond else [])

    # ── Training sliding windows ──
    X_train, y_train, unit_ids = sliding_windows(train_df, cfg.window_size, cfg.stride, feature_cols)

    # ── Test: last window per engine ──
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
        y_test_list.append(rul_true[i])

    return X_train, y_train, unit_ids, np.array(X_test_list), np.array(y_test_list)
