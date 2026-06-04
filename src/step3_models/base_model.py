import torch
import torch.nn as nn
from src.step2_features.patching import PatchEmbedding


class iTransformerEncoder(nn.Module):
    """倒置 Transformer：把变量当 token、时间当特征"""

    def __init__(self, d_in: int, d_model: int, n_heads: int, e_layers: int,
                 d_ff: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation=activation, batch_first=True,
            norm_first=True,  # Pre-LN: stabilize training (iTransformer default)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = x + self.pos_embed            # 加位置编码
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)                 # 在变量维度上池化
        return x


class MLPRegressor(nn.Module):
    """三层 MLP：256→100→50→1"""
    def __init__(self, d_model: int, hidden: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PatchiTransformerRUL(nn.Module):
    """Base 模型：Patch → iTransformer → MLP"""

    def __init__(self, cfg):
        super().__init__()
        n_vars = cfg.n_vars
        self.patch_embed = PatchEmbedding(
            patch_len=cfg.patch_len,
            stride=cfg.stride_patch,
            d_model=cfg.d_model,
        )
        # Compute N*d_model for the iTransformer input dim
        with torch.no_grad():
            dummy = torch.zeros(1, n_vars, cfg.window_size)
            out = self.patch_embed(dummy)  # (1, C, N, d_model)
            d_in = out.shape[2] * out.shape[3]

        self.itrans_encoder = iTransformerEncoder(
            d_in=d_in,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            e_layers=cfg.e_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.activation,
        )
        self.regressor = MLPRegressor(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)                 # (B, C, L) 传感器→通道维度
        x = self.patch_embed(x)               # (B, C, N, d_model)
        B, C, N, D = x.shape
        x = x.reshape(B, C, N * D)            # (B, C, N*D) 拼合 patch 维度
        x = self.itrans_encoder(x)            # (B, d_model)
        x = self.regressor(x)
        return x
