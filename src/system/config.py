from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BaseConfig:
    # ---- 数据 ----
    data_dir: str = "archive"
    subset: Literal["FD001", "FD002", "FD003", "FD004"] = "FD001"

    sensors: tuple = (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)

    window_size: int = 30  # FD001/FD003 用 30，FD002/FD004 用 50
    stride: int = 1

    rul_max: int = 125  # 训练时 RUL 上限截断值

    patch_len: int = 16
    stride_patch: int = 8

    d_model: int = 256
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.3
    activation: str = "gelu"

    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 500
    patience: int = 40
    train_ratio: float = 0.8

    seed: int = 42

    num_workers: int = 4

    branch_indices: tuple = (0, 1, 2)  # 0=P8S4, 1=P16S8, 2=P32S16
    fusion_mode: str = "concat"  # concat / weighted_sum / gated

    norm_mode: str = "per_condition"  # global 或 per_condition

    use_op_cond: bool = True  # FD002/FD004 是否加 altitude/mach/tra

    device: str = "cuda"
    _n_vars: int = 0  # 实际输入通道数，初始化时算出来

    @property
    def n_vars(self) -> int:
        return self._n_vars

    def __post_init__(self):
        if self.subset in ("FD002", "FD004"):
            self.window_size = 50
        base_n_vars = len(self.sensors)  # 14
        self._n_vars = base_n_vars + (3 if self.use_op_cond and self.subset in ("FD002", "FD004") else 0)


# Presets per subset
SUBSET_CONFIG = {
    "FD001": dict(subset="FD001", window_size=30, batch_size=64, rul_max=125),
    "FD002": dict(subset="FD002", window_size=50, batch_size=128, rul_max=175),
    "FD003": dict(subset="FD003", window_size=30, batch_size=64, rul_max=125),
    "FD004": dict(subset="FD004", window_size=50, batch_size=128, rul_max=175),
}
