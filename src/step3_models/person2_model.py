import torch
import torch.nn as nn
import torch.nn.functional as F
from src.step2_features.patching import ConvPatchEmbedding
from src.step3_models.base_model import iTransformerEncoder, MLPRegressor


class MSPatchiTransformerRUL(nn.Module):
    """Person 2：三分支时间 Patch 流，支持消融实验"""
    PATCH_CONFIGS = [
        (8, 4),    # 小尺度
        (16, 8),   # 中尺度
        (32, 16),  # 大尺度
    ]

    def __init__(self, cfg):
        super().__init__()
        n_vars = cfg.n_vars
        D = cfg.d_model

        # 消融：选择分支和融合方式
        branch_indices = getattr(cfg, "branch_indices", (0, 1, 2))
        fusion_mode = getattr(cfg, "fusion_mode", "concat")
        self.fusion_mode = fusion_mode
        active_configs = [self.PATCH_CONFIGS[i] for i in branch_indices]

        # 时间维补零，让最大 patch 能完整覆盖
        max_patch = max(p for p, _ in active_configs)
        self.pad_len = max_patch

        # 构建分支
        self.branches = nn.ModuleList([
            ConvPatchEmbedding(p, s, D, kernel_size=5, channel_shared=True)
            for p, s in active_configs
        ])
        n_branches = len(active_configs)

        # 融合层
        if fusion_mode == "concat":
            with torch.no_grad():
                dummy = torch.zeros(1, n_vars, cfg.window_size + self.pad_len)
                total_d = 0
                for b in self.branches:
                    out = b(dummy)
                    total_d += out.shape[2] * out.shape[3]
            self.fusion = nn.Linear(total_d, D)
        elif fusion_mode == "weighted_sum":
            self.branch_projs = nn.ModuleList()
            with torch.no_grad():
                dummy = torch.zeros(1, n_vars, cfg.window_size + self.pad_len)
                for b in self.branches:
                    out = b(dummy)
                    n_patches = out.shape[2]
                    self.branch_projs.append(nn.Linear(n_patches * D, D))
            self.branch_logits = nn.Parameter(torch.zeros(n_branches))
        elif fusion_mode == "gated":
            self.branch_projs = nn.ModuleList()
            with torch.no_grad():
                dummy = torch.zeros(1, n_vars, cfg.window_size + self.pad_len)
                for b in self.branches:
                    out = b(dummy)
                    n_patches = out.shape[2]
                    self.branch_projs.append(nn.Linear(n_patches * D, D))
            self.gate_net = nn.Sequential(
                nn.Linear(n_branches * D, D),
                nn.ReLU(),
                nn.Linear(D, n_branches * D),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"未知融合方式: {fusion_mode}")

        self.itrans_encoder = iTransformerEncoder(
            d_in=D, d_model=D, n_heads=cfg.n_heads,
            e_layers=cfg.e_layers, d_ff=cfg.d_ff,
            dropout=cfg.dropout, activation=cfg.activation,
        )
        self.regressor = MLPRegressor(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)                     # (B, L, C) → (B, C, L)

        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len), mode="replicate")

        # 三分支提取特征
        branch_feats = []
        for branch in self.branches:
            t = branch(x)
            B, C, N, D = t.shape
            branch_feats.append(t.reshape(B, C, N * D))

        # 融合
        if self.fusion_mode == "concat":
            sensor_feats = torch.cat(branch_feats, dim=-1)
            sensor_feats = self.fusion(sensor_feats)
        elif self.fusion_mode == "weighted_sum":
            proj = [proj(b) for b, proj in zip(branch_feats, self.branch_projs)]
            weights = F.softmax(self.branch_logits, dim=0)
            sensor_feats = sum(w * p for w, p in zip(weights, proj))
        elif self.fusion_mode == "gated":
            proj = [proj(b) for b, proj in zip(branch_feats, self.branch_projs)]
            concat = torch.cat(proj, dim=-1)
            gate = self.gate_net(concat)
            n_b = len(proj)
            gated = [proj[i] * gate[..., i*D:(i+1)*D] for i in range(n_b)]
            sensor_feats = sum(gated)

        out = self.itrans_encoder(sensor_feats)
        return self.regressor(out)
