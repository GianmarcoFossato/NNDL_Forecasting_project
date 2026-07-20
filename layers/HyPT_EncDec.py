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

        # top_k: number of dominant periods to model.
        # Suggested range: [2, 5] (matches HyPT_config.json). ECL has strong
        # daily + weekly cycles, so k=3-5 is a reasonable default; k=2 risks
        # missing the weekly component, k>5 mostly adds compute for
        # diminishing returns once the top 2-3 frequencies are captured.
        self.k = configs.top_k

        # d_period: Branch A's internal working width, decoupled from d_model.
        # Suggested range: [16, 32, 64, 128]. Rule of thumb: d_model // 4 to
        # d_model // 8. Start at 32-64; only go bigger if periodicity
        # modeling is clearly the bottleneck in your ablations (step 1 of the
        # ablation plan -- Branch A alone should roughly match TimesNet's
        # published ECL numbers, which used a much smaller d_model than your
        # main model does).
        self.d_period = configs.d_period if getattr(configs, 'd_period', None) else max(configs.d_model // 4, 8)

        self.down_proj = nn.Linear(configs.d_model, self.d_period)
        self.up_proj = nn.Linear(self.d_period, configs.d_model)

        # k distinct 2D conv blocks, one per period, operating in d_period space.
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

        # Average across Batch*Variates ONLY to choose *which* frequencies are
        # globally dominant. This mirrors TimesNet: which periods to use is
        # shared across the batch (you can't reshape different samples into
        # different 2D grids in one batched conv call), but see below --
        # WEIGHTING each chosen period is still per-sample.
        mean_amplitude = amplitude.mean(dim=0)
        _, top_list = torch.topk(mean_amplitude, self.k)

        # BUG FIX (was previously reusing the batch-averaged amplitude as the
        # weight for every sample, which threw away all per-sample
        # adaptivity). Re-index the ORIGINAL per-sample amplitude at the
        # chosen frequencies and softmax per-sample, so a window that's more
        # daily-cycle-dominated gets a different period mix than one that's
        # more weekly-cycle-dominated.
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

            x_2d = x_padded.reshape(B, length // period, period, d_period).permute(0, 3, 2, 1).contiguous()
            out_2d = self.conv_blocks[i](x_2d)
            out_2d = out_2d.permute(0, 3, 2, 1).contiguous()
            out_1d = out_2d.reshape(B, length, d_period)

            w = period_weights[:, i].view(B, 1, 1)  # per-sample scalar weight, not a global constant
            res = res + (out_1d[:, :T, :] * w)

        return self.up_proj(res)  # back to d_model


class CrossVariateBranch(nn.Module):
    """
    Branch B: symmetric cross-variate mixing (iTransformer-style), applied to
    ALL N channels -- every channel is both a query and a key/value, unlike
    the previous version where only one target channel attended to the rest.

    Design note: a literal port of iTransformer's variate embedding would
    flatten each channel's [T, d_model] into one T*d_model -> d_model Linear.
    That was affordable when only 1 target token needed it; now that every
    one of the N channels needs a token AND a way to get context back, doing
    that flatten in both directions costs O(N * T * d_model^2), which is the
    single most expensive thing in the model at N=321. Instead:
      1. mean-pool over T for a cheap per-variate summary token: O(N*T*d_model)
      2. ordinary self-attention across the N variate tokens: O(N^2 * d_model),
         same cost iTransformer itself pays on ECL
      3. broadcast the refined token back across T as additive context,
         instead of re-projecting D -> T*d_model (which would reintroduce the
         cost this whole redesign exists to avoid)
    Branch A already owns fine-grained temporal modeling, so a pooled token
    for Branch B is a deliberate simplification, not an oversight -- if
    ablations show it's underpowered, the first thing to try is swapping the
    mean-pool for attention-pooling (a single learnable query attending over
    T) before reaching for the full flatten again.
    """

    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model

        self.token_proj = nn.Linear(configs.d_model, configs.d_model)

        self.cross_attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                # factor: attention scaling factor inherited from the library's
                # FullAttention. Not worth tuning for full (non-sparse)
                # attention -- leave at the run.py default (1).
                factor=configs.factor,
                attention_dropout=configs.dropout,
                output_attention=False
            ),
            configs.d_model,
            # n_heads: suggested {4, 8} (matches HyPT_config.json). With
            # d_model in the 128-512 range, 8 heads is a safe default;
            # drop to 4 if d_model is on the small end (e.g. 128) so each
            # head still gets a reasonable width.
            configs.n_heads
        )

        # Gate controlling how much cross-variate context gets added back
        # into each timestep. Initialized small so training starts close to
        # "mostly ignore cross-variate context" and ramps it up as useful.
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
    schedule. Applied symmetrically to ALL N channels -- the previous
    endo/exo split is gone, since it meant only 1 of 321 channels ever got
    refined by either branch, which isn't a fair --features M comparison
    against iTransformer/TimeXer/TimesNet.
    """
    def __init__(self, configs, conv_builder):
        super().__init__()
        self.branch_a = PluggablePeriodBlock(configs, conv_builder)
        self.branch_b = CrossVariateBranch(configs)

        self.gate = nn.Parameter(torch.zeros(1, 1, 1, configs.d_model))

        # branch_dropout: suggested range [0.05, 0.25], step 0.05 (matches
        # HyPT_config.json). Higher = more aggressive regularization against
        # over-relying on a single branch. Start at 0.1.
        self.target_branch_dropout = getattr(configs, 'branch_dropout', 0.1)

        # branch_warmup_epochs: suggested {2, 3, 5} (matches HyPT_config.json).
        # Should scale with train_epochs -- use 2-3 for a 10-epoch run, up to
        # 5 for a 20-epoch run, so the warmup doesn't eat the whole schedule.
        self.warmup_epochs = getattr(configs, 'branch_warmup_epochs', 3)
        self.current_epoch = 0

        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        # dropout: suggested range [0.05, 0.3], step 0.05 (matches
        # HyPT_config.json). ECL is fairly low-noise at the daily/weekly
        # scale; 0.1-0.15 is a reasonable starting point, push higher only if
        # you see overfitting in the loss curves.
        self.dropout = nn.Dropout(configs.dropout)

        self.ffn = nn.Sequential(
            # d_ff: suggested {256, 512, 1024} (matches HyPT_config.json).
            # Keep roughly 2x d_model as a starting ratio.
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