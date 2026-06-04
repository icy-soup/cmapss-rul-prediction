import torch
import torch.nn as nn
from src.step2_features.patching import ConvPatchEmbedding
from src.step3_models.base_model import iTransformerEncoder, MLPRegressor


class ConvPatchiTransformerRUL(nn.Module):
    """Person 1：用卷积 Patch 替换 Base 的线性投影"""

    def __init__(self, cfg):
        super().__init__()
        n_vars = cfg.n_vars
        self.patch_embed = ConvPatchEmbedding(
            patch_len=cfg.patch_len,
            stride=cfg.stride_patch,
            d_model=cfg.d_model,
            kernel_size=5,
            channel_shared=True,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_vars, cfg.window_size)
            out = self.patch_embed(dummy)
            d_in = out.shape[2] * out.shape[3]

        self.itrans_encoder = iTransformerEncoder(
            d_in=d_in, d_model=cfg.d_model, n_heads=cfg.n_heads,
            e_layers=cfg.e_layers, d_ff=cfg.d_ff,
            dropout=cfg.dropout, activation=cfg.activation,
        )
        self.regressor = MLPRegressor(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.patch_embed(x)
        B, C, N, D = x.shape
        x = x.reshape(B, C, N * D)
        x = self.itrans_encoder(x)
        x = self.regressor(x)
        return x
