import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class PluggablePeriodBlock(nn.Module):
    """
    Branch A: Period-aware Temporal Prior (Channel Independent).
    Identifies top-k periods via FFT over patches and processes them with 2D convolutions.
    """

    def __init__(self, configs, conv_builder):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        # Reduced width for Branch A
        self.d_period = configs.d_period if getattr(configs, 'd_period', None) else max(configs.d_model // 4, 8)

        self.down_proj = nn.Linear(configs.d_model, self.d_period)
        self.up_proj = nn.Linear(self.d_period, configs.d_model)

        # Standard building of k conv blocks in d_period space
        self.conv_blocks = nn.ModuleList([
            conv_builder(self.d_period, self.d_period) for _ in range(self.k)
        ])

    def forward(self, x):
        # x shape: [Batch * Variates, Num_Patches (P), d_model]
        x = self.down_proj(x)  # [B*N, P, d_period]
        BN, P, d_period = x.size()

        # FFT across Patches (P)
        xf = torch.fft.rfft(x.transpose(1, 2), dim=-1)  # [B*N, d_period, P//2 + 1]

        # Amplitude spectrum
        amplitude = torch.mean(torch.abs(xf), dim=1)  # [B*N, P//2 + 1]
        amplitude[:, 0] = 0  # ignore DC component

        # Top-k global frequencies
        mean_amplitude = amplitude.mean(dim=0)
        _, top_list = torch.topk(mean_amplitude, self.k)

        # Per-sample adaptive weights
        per_sample_amplitude = amplitude[:, top_list]  # [B*N, k]
        period_weights = torch.softmax(per_sample_amplitude, dim=-1)  # [B*N, k]

        period_list = [max(P // freq.item(), 1) if freq.item() != 0 else P for freq in top_list]

        res = torch.zeros_like(x)

        for i, period in enumerate(period_list):
            length = ((P + period - 1) // period) * period
            padding = length - P

            if padding > 0:
                pad_tensor = torch.zeros((BN, padding, d_period), dtype=x.dtype, device=x.device)
                x_padded = torch.cat([x, pad_tensor], dim=1)
            else:
                x_padded = x

            # 2D Grid reshape over Patches: [B*N, d_period, H, W]
            x_2d = x_padded.reshape(BN, length // period, period, d_period).permute(0, 3, 2, 1).contiguous()

            # Pass through ConvNeXtBlock2D
            out_2d = self.conv_blocks[i](x_2d)

            out_2d = out_2d.permute(0, 3, 2, 1).contiguous()
            out_1d = out_2d.reshape(BN, length, d_period)

            w = period_weights[:, i].view(BN, 1, 1)
            res = res + (out_1d[:, :P, :] * w)

        return self.up_proj(res)  # [B*N, P, d_model]


class CrossVariateBranch(nn.Module):
    """
    Branch B: Cross-variate attention over channel summary tokens.
    """

    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        self.token_proj = nn.Linear(configs.d_model, configs.d_model)

        self.cross_attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                factor=configs.factor,
                attention_dropout=configs.dropout,
                output_attention=False
            ),
            configs.d_model,
            configs.n_heads
        )

        self.context_gate = nn.Linear(configs.d_model, configs.d_model)
        nn.init.zeros_(self.context_gate.bias)
        nn.init.normal_(self.context_gate.weight, std=0.02)

    def forward(self, x):
        # x shape: [B, N, P, D]
        B, N, P, D = x.size()

        # Mean pool over Patches (P) -> Summary token per variate: [B, N, D]
        tokens = self.token_proj(x.mean(dim=2))

        # Cross-channel self-attention across N channels
        attn_out, _ = self.cross_attention(tokens, tokens, tokens, attn_mask=None)  # [B, N, D]

        # Gate and broadcast back across patches
        context = self.context_gate(attn_out)  # [B, N, D]
        context = context.unsqueeze(2).expand(-1, -1, P, -1)  # [B, N, P, D]

        return context


class HybridEncoderLayer(nn.Module):
    def __init__(self, configs, conv_builder):
        super().__init__()
        self.branch_a = PluggablePeriodBlock(configs, conv_builder)
        self.branch_b = CrossVariateBranch(configs)

        self.gate = nn.Parameter(torch.zeros(1, 1, 1, configs.d_model))
        self.target_branch_dropout = getattr(configs, 'branch_dropout', 0.1)
        self.warmup_epochs = getattr(configs, 'branch_warmup_epochs', 3)
        self.current_epoch = 0

        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        self.ffn = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.d_model)
        )

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _get_current_drop_rate(self) -> float:
        if not self.training or self.warmup_epochs <= 0:
            return self.target_branch_dropout
        if self.current_epoch < self.warmup_epochs:
            return self.target_branch_dropout * (self.current_epoch / self.warmup_epochs)
        return self.target_branch_dropout

    def forward(self, x):
        # x shape: [B, N, P, D]
        B, N, P, D = x.size()

        # Branch A: Periodicity over time patches (vectorized B*N)
        h_a_flat = self.branch_a(x.reshape(B * N, P, D))
        h_a = h_a_flat.reshape(B, N, P, D)

        # Branch B: Cross-variate mixing across channels
        h_b = self.branch_b(x)

        # Dynamic Fusion
        current_p = self._get_current_drop_rate()
        if self.training and current_p > 0.0:
            rand_val = torch.rand(1, device=x.device).item()
            if rand_val < current_p / 2:
                fused = h_a
            elif rand_val < current_p:
                fused = h_b
            else:
                g = torch.sigmoid(self.gate)
                fused = g * h_a + (1 - g) * h_b
        else:
            g = torch.sigmoid(self.gate)
            fused = g * h_a + (1 - g) * h_b

        x = self.norm1(x + self.dropout(fused))
        out = self.norm2(x + self.dropout(self.ffn(x)))

        return out