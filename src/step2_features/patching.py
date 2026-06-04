import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Linear patch embedding (Base version).

    Input:  (B, C, L)  Output: (B, C, N, d_model)
    """
    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B, C, N, P = patches.shape
        return self.proj(patches.reshape(B * C, N, P)).reshape(B, C, N, -1)


class ConvPatchEmbedding(nn.Module):
    """Convolution-enhanced patch embedding (Person 1).

    Depthwise Conv1D extracts local patterns within each patch,
    then Pointwise Conv projects to d_model.

    Input:  (B, C, L)  Output: (B, C, N, d_model)
    """
    def __init__(self, patch_len: int, stride: int, d_model: int,
                 kernel_size: int = 3, channel_shared: bool = True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.channel_shared = channel_shared

        if channel_shared:
            # Shared Conv across all channels
            self.depthwise = nn.Conv1d(1, 1, kernel_size=kernel_size,
                                       padding=kernel_size // 2, bias=False)
            self.pointwise = nn.Linear(patch_len, d_model)
        else:
            # Channel-independent Conv
            self.depthwise = nn.Conv1d(1, 1, kernel_size=kernel_size,
                                       padding=kernel_size // 2, bias=False)
            self.pointwise = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B, C, N, P = patches.shape  # (B, C, N, P)

        if self.channel_shared:
            # Merge B and C for shared processing: (B*C, N, P)
            tokens = patches.reshape(B * C, N, P)

            # Depthwise: apply along patch dimension (last dim)
            # Reshape to (B*C*N, 1, P) for Conv1d
            dw_in = tokens.reshape(-1, 1, P)  # (B*C*N, 1, P)
            dw_out = self.depthwise(dw_in)     # (B*C*N, 1, P)
            dw_out = dw_out.reshape(B * C, N, P)

            # Pointwise: map P to d_model
            tokens = self.pointwise(dw_out)     # (B*C, N, d_model)
            tokens = tokens.view(B, C, N, -1)   # (B, C, N, d_model)
        else:
            tokens_list = []
            for c in range(C):
                t = patches[:, c]  # (B, N, P)
                Bc, Nc, Pc = t.shape
                dw_in = t.reshape(-1, 1, Pc)
                dw_out = self.depthwise(dw_in).reshape(Bc, Nc, Pc)
                tokens_list.append(self.pointwise(dw_out)[:, None])
            tokens = torch.cat(tokens_list, dim=1)

        return tokens  # (B, C, N, d_model)
