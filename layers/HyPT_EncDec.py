import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class PluggablePeriodBlock(nn.Module):
    """
    Branch A: Period-aware Temporal Prior.
    Identifies top-k periods via FFT and processes them using 2D convolutions.
    Channel-independent: operates on a flattened [Batch * Variates, Time, .]
    tensor, so the SAME conv weights are applied to every one of the N=321
    ECL channels -- this is what makes it scale to all channels instead of
    just one; only activation memory grows with N, not parameter count.

    Runs in a *reduced* d_period dimension rather than the full d_model.
    ConvNeXtBlock2D's pointwise convs cost O(d_model^2) per spatial location,
    and that cost gets paid B*N times once this runs on the full channel set
    -- it's the dominant cost of the whole model at N=321. down_proj/up_proj
    bottleneck this branch to d_period and back, buying roughly
    (d_model / d_period)^2 savings on that cost for two cheap linear layers.
    TimesNet's own published ECL config uses d_model=16-32 for exactly this
    reason (it pays this same per-channel cost, just for its whole backbone).
    """

    def __init__(self, configs, conv_builder):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.k = configs.top_k

        # d_period: Branch A's internal working width, decoupled from d_model.
        self.d_period = configs.d_period if getattr(configs, 'd_period', None) else max(configs.d_model // 4, 8)

        self.down_proj = nn.Linear(configs.d_model, self.d_period)
        self.up_proj = nn.Linear(self.d_period, configs.d_model)

        # All channels are processed together in the channel dimension
        self.conv_blocks = nn.ModuleList([
            conv_builder(self.d_period, self.d_period) for _ in range(self.k)
        ])
    def forward(self, x):
        # x shape: [Batch * Variates, Time, d_model]
        x = self.down_proj(x)  # [B*N, T, d_period]
        B, T, d_period = x.size()

        # Real FFT across Time to find dominant frequencies.
        xf = torch.fft.rfft(x.transpose(1, 2), dim=-1)  # [B*N, d_period, T//2 + 1]

        # Per-sample amplitude spectrum (averaged over the feature dim only).
        amplitude = torch.mean(torch.abs(xf), dim=1)  # [B*N, T//2 + 1]
        amplitude[:, 0] = 0  # ignore DC component


        mean_amplitude = amplitude.mean(dim=0)
        _, top_list = torch.topk(mean_amplitude, self.k)

        per_sample_amplitude = amplitude[:, top_list]  # [B*N, k]
        period_weights = torch.softmax(per_sample_amplitude, dim=-1)  # [B*N, k]

        period_list = [max(T // freq.item(), 1) if freq.item() != 0 else T for freq in top_list]

        res = torch.zeros_like(x)

        for i, period in enumerate(period_list):
            length = ((T + period - 1) // period) * period
            padding = length - T

            if padding > 0:
                pad_tensor = torch.zeros((B, padding, d_period), dtype=x.dtype, device=x.device)
                x_padded = torch.cat([x, pad_tensor], dim=1)
            else:
                x_padded = x

                # Reshape to separate B and N: [B, N, padded_length, d_period]
                x_2d = x_padded.view(B // self.N, self.N, length, d_period)

                # Reshape for 2D Conv: [B, N * d_period, H, W]
                x_2d = x_2d.view(B // self.N, self.N * d_period, length // period, period)

                # Apply Grouped Conv
                out_2d = self.conv_blocks[i](x_2d)

                # Return to [B*N, length, d_period]
                out_2d = out_2d.view(B // self.N, self.N, d_period, length // period, period)
                out_2d = out_2d.permute(0, 1, 3, 4, 2).contiguous()  # [B, N, H, W, d_period]
                out_1d = out_2d.view(B, length, d_period)  # Back to flat B*N

            w = period_weights[:, i].view(B, 1, 1)  # per-sample scalar weight, not a global constant
            res = res + (out_1d[:, :T, :] * w)

        return self.up_proj(res)  # back to d_model


class CrossVariateBranch(nn.Module):
    """
    Branch B: symmetric cross-variate mixing (iTransformer-style), applied to
    ALL N channels -- every channel is both a query and a key/value.
    """

    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model

        self.token_proj = nn.Linear(configs.d_model, configs.d_model)

        self.cross_attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                # Attention scaling factor inherited from the library's FullAttention.
                factor=configs.factor,
                attention_dropout=configs.dropout,
                output_attention=False
            ),
            configs.d_model,
            configs.n_heads
        )

        # Gate controlling how much cross-variate context gets added back
        # into each timestep. Initialized small so training starts close to
        # mostly ignore cross-variate context and ramps it up as useful.
        self.context_gate = nn.Linear(configs.d_model, configs.d_model)
        nn.init.zeros_(self.context_gate.bias)
        nn.init.normal_(self.context_gate.weight, std=0.02)

    def forward(self, x):
        # x: [B, N, T, D]
        B, N, T, D = x.size()

        tokens = self.token_proj(x.mean(dim=2))  # [B, N, D] -- no T*D matmul

        attn_out, _ = self.cross_attention(tokens, tokens, tokens, attn_mask=None)  # [B, N, D]

        context = self.context_gate(attn_out)  # [B, N, D]
        context = context.unsqueeze(2).expand(-1, -1, T, -1)  # broadcast across T

        return context


class HybridEncoderLayer(nn.Module):
    """
    Combines Branch A (period-aware temporal prior) and Branch B (cross-variate
    context) with a learnable convex gate and a branch-dropout annealing
    schedule. Applied symmetrically to ALL N channels.
    """
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
        # x shape: [Batch, Variates (N), Time (T), d_model (D)]
        B, N, T, D = x.size()

        # Branch A: periodicity, vectorized over B*N -- every channel included.
        h_a_flat = self.branch_a(x.reshape(B * N, T, D))
        h_a = h_a_flat.reshape(B, N, T, D)

        # Branch B: cross-variate context, symmetric across every channel.
        h_b = self.branch_b(x)

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