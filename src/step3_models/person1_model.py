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
        # 把传感器挪到通道维，方便用卷积处理时间序列
        x = x.transpose(1, 2)
        # 用卷积增强的 Patch Embedding 提取局部退化特征
        x = self.patch_embed(x)
        B, C, N, D = x.shape
        # 把每个传感器的 patch 序列拉平，拼成一个 token 给 iTransformer
        x = x.reshape(B, C, N * D)
        # 跨传感器做自注意力，捕捉传感器之间的关联
        x = self.itrans_encoder(x)
        x = self.regressor(x)
        return x
