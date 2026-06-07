import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """线性 Patch 映射（Base 版本）
    输入：(B, C, L) → 输出：(B, C, N, d_model)
    """
    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 把时间维展开成一个个 patch 然后做线性变换
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B, C, N, P = patches.shape
        # 合并传感器维度统一算，效果等于每个传感器独立做线性映射
        return self.proj(patches.reshape(B * C, N, P)).reshape(B, C, N, -1)


class ConvPatchEmbedding(nn.Module):
    """卷积增强的 Patch 映射（Person 1 版本）
    先用 Depthwise Conv1D 提取 patch 内部局部模式，
    再用 Pointwise Conv（即 Linear）投影到 d_model。
    输入：(B, C, L) → 输出：(B, C, N, d_model)
    """
    def __init__(self, patch_len: int, stride: int, d_model: int,
                 kernel_size: int = 3, channel_shared: bool = True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.channel_shared = channel_shared

        if channel_shared:
            # 所有传感器共用同一个深度可分离卷积
            self.depthwise = nn.Conv1d(1, 1, kernel_size=kernel_size,
                                       padding=kernel_size // 2, bias=False)
            self.pointwise = nn.Linear(patch_len, d_model)
        else:
            # 每个传感器各自独立做卷积，接口跟共享版本保持一致
            self.depthwise = nn.Conv1d(1, 1, kernel_size=kernel_size,
                                       padding=kernel_size // 2, bias=False)
            self.pointwise = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 把时间维切成 patch，然后先用 depthwise conv 提取局部模式再投影
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B, C, N, P = patches.shape

        if self.channel_shared:
            # 合拼所有传感器一起算，省显存
            tokens = patches.reshape(B * C, N, P)
            dw_in = tokens.reshape(-1, 1, P)
            dw_out = self.depthwise(dw_in)
            dw_out = dw_out.reshape(B * C, N, P)
            tokens = self.pointwise(dw_out)
            tokens = tokens.view(B, C, N, -1)
        else:
            # 每个传感器单独过卷积
            tokens_list = []
            for c in range(C):
                t = patches[:, c]
                Bc, Nc, Pc = t.shape
                dw_in = t.reshape(-1, 1, Pc)
                dw_out = self.depthwise(dw_in).reshape(Bc, Nc, Pc)
                tokens_list.append(self.pointwise(dw_out)[:, None])
            tokens = torch.cat(tokens_list, dim=1)

        return tokens
