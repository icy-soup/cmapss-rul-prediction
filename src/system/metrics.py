import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(s))


def all_metrics(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "Score": phm08_score(y_true, y_pred),
    }
