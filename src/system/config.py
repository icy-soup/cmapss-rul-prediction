from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BaseConfig:
    # Data
    data_dir: str = "archive"
    subset: Literal["FD001", "FD002", "FD003", "FD004"] = "FD001"

    # Sensor selection: 14 sensors with trend
    sensors: tuple = (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)

    # Sliding window
    window_size: int = 30  # FD001/FD003=30, FD002/FD004=50
    stride: int = 1

    # RUL
    rul_max: int = 125

    # Patch
    patch_len: int = 16
    stride_patch: int = 8

    # Model
    d_model: int = 256
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.3
    activation: str = "gelu"

    # Train
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 500
    patience: int = 40
    train_ratio: float = 0.8

    # Reproducibility
    seed: int = 42

    # DataLoader
    num_workers: int = 4

    # Ablation: branch selection and fusion (Person 2)
    branch_indices: tuple = (0, 1, 2)  # 0=small(P8/S4), 1=medium(P16/S8), 2=large(P32/S16)
    fusion_mode: str = "concat"  # "concat", "weighted_sum", "gated"

    # Normalization
    norm_mode: str = "per_condition"  # "global" or "per_condition"

    # Input features: include altitude/mach/tra as additional channels
    use_op_cond: bool = True

    # Device
    device: str = "cuda"
    _n_vars: int = 0  # computed in __post_init__, actual input channel count

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
    "FD001": dict(subset="FD001", window_size=30, batch_size=64),
    "FD002": dict(subset="FD002", window_size=50, batch_size=128),
    "FD003": dict(subset="FD003", window_size=30, batch_size=64),
    "FD004": dict(subset="FD004", window_size=50, batch_size=128),
}
