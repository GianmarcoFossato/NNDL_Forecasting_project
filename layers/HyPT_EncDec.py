import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class PluggablePeriodBlock(nn.Module):
    """
    Branch A: Period-aware Temporal Prior.
    Identifies top-k periods via FFT and processes them using 2D convolutions.
    Maintains channel independence by processing flattened batch and variate dimensions.
    """

    def __init__(self, configs, conv_builder):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        # Initialize k distinct 2D convolution blocks for the top-k periods
        self.conv_blocks = nn.ModuleList([
            conv_builder(configs.d_model, configs.d_model) for _ in range(self.k)
        ])

    def forward(self, x):
        # x shape: [Batch * Variates, Time, d_model]
        B, T, d_model = x.size()

        # Apply Real FFT across the Time dimension to find dominant frequencies
        xf = torch.fft.rfft(x.transpose(1, 2), dim=-1) # [Batch * Variates, d_model, Time // 2 + 1]

        # Calculate average amplitude across the d_model dimension
        # amplitude shape: [Batch * Variates, Time // 2 + 1]
        amplitude = torch.mean(torch.abs(xf), dim=1)
        amplitude[:, 0] = 0  # Ignore the DC component (0 Hz)

        # Average across the Batch * Variates dimension to find global top-k periods
        # mean_amplitude shape: [Time // 2 + 1]
        mean_amplitude = amplitude.mean(dim=0)
        top_amplitudes, top_list = torch.topk(mean_amplitude, self.k)

        # L2 normalization to the raw amplitudes before softmax.
        # This scales values to prevent a hard argmax
        normalized_amplitudes = F.normalize(top_amplitudes, p=2, dim=-1)
        period_weights = torch.softmax(normalized_amplitudes, dim=-1)

        # Convert frequency indices to period lengths
        period_list = [max(T // freq.item(), 1) if freq.item() != 0 else T for freq in top_list]

        res = torch.zeros_like(x)

        # Reshape into 2D, apply Conv2D, and aggregate using softmax weights
        for i, period in enumerate(period_list):
            # Calculate required padding so Time is divisible by the period
            length = ((T + period - 1) // period) * period
            padding = length - T

            if padding > 0:
                pad_tensor = torch.zeros((B, padding, d_model), dtype=x.dtype, device=x.device)
                x_padded = torch.cat([x, pad_tensor], dim=1)
            else:
                x_padded = x

            # Reshape 1D time series into 2D representation
            # x_2d shape: [Batch * Variates, d_model, period, length // period]
            x_2d = x_padded.reshape(B, length // period, period, d_model).permute(0, 3, 2, 1).contiguous()

            # Apply the corresponding 2D convolution block
            out_2d = self.conv_blocks[i](x_2d)

            # Revert from [B, d_model, period, length // period] back to [B, length // period, period, d_model]
            out_2d = out_2d.permute(0, 3, 2, 1).contiguous()
            out_1d = out_2d.reshape(B, length, d_model)

            # Truncate padding and aggregate using the softmax weight of the current period
            res = res + (out_1d[:, :T, :] * period_weights[i])

        return res


class ExogenousBranch(nn.Module):
    """
    Branch B: iTransformer-style Inverted Cross-Variate Attention.
    Embeds entire temporal sequence vectors into variate tokens and computes
    Self-Attention across channels (N), learning global inter-variate dependencies.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model

        # Linear projection maps individual variate temporal features (T * d_model) -> d_model token vector
        self.variate_embedding = nn.Linear(configs.seq_len * configs.d_model, configs.d_model)

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

        # Map attention-refined variate tokens back to the full sequence feature space (T * d_model)
        self.variate_projection = nn.Linear(configs.d_model, configs.seq_len * configs.d_model)

    def forward(self, x_exo, x_endo):
        # x_exo:  [Batch, N-1, Time, d_model]
        # x_endo: [Batch, 1, Time, d_model]
        B, N_exo, T, D = x_exo.size()

        # Map temporal sequences to token vectors [B, Variates, d_model]
        exo_flat = x_exo.reshape(B, N_exo, T * D)
        endo_flat = x_endo.reshape(B, 1, T * D)

        exo_tokens = self.variate_embedding(exo_flat)  # [B, N-1, d_model]
        endo_token = self.variate_embedding(endo_flat)  # [B, 1, d_model]

        # Target (Query) attends to Exogenous variables (Key/Value)
        attn_out, _ = self.cross_attention(
            endo_token,  # Query
            exo_tokens,  # Key
            exo_tokens,  # Value
            attn_mask=None
        )  # Output shape: [B, 1, d_model]

        # Project back to full sequence shape [B, 1, T, D]
        out_flat = self.variate_projection(attn_out)
        out_endo = out_flat.reshape(B, 1, T, D)

        return out_endo


class HybridEncoderLayer(nn.Module):
    """
    Combines Target Periodicity (Branch A) and Exogenous Context (Branch B)
    using a learnable convex gate and annealing schedule for the Target channel.
    """
    def __init__(self, configs, conv_builder):
        super().__init__()
        self.branch_a = PluggablePeriodBlock(configs, conv_builder)
        self.branch_b = ExogenousBranch(configs)

        # Gate parameter for fusing target representations
        self.gate = nn.Parameter(torch.zeros(1, 1, 1, configs.d_model))

        # Branch drop hyperparameters
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
        """Sets current training epoch to compute annealing schedules."""
        self.current_epoch = epoch

    def _get_current_drop_rate(self) -> float:
        """Computes linearly annealed branch drop probability based on current epoch."""
        if not self.training or self.warmup_epochs <= 0:
            return self.target_branch_dropout
        if self.current_epoch < self.warmup_epochs:
            return self.target_branch_dropout * (self.current_epoch / self.warmup_epochs)
        return self.target_branch_dropout

    def forward(self, x):
        # x shape: [Batch, Variates (N), Time (T), d_model (D)]
        B, N, T, D = x.size()

        x_exo = x[:, :-1, :, :]   # Exogenous variables [B, N-1, T, D]
        x_endo = x[:, -1:, :, :]  # Endogenous target [B, 1, T, D]

        # Branch A: Process period structure on target channel only (Fast B*1 processing)
        h_a_flat = self.branch_a(x_endo.reshape(B, T, D))
        h_a = h_a_flat.reshape(B, 1, T, D)

        # Branch B: Target attends to Exogenous channels
        if N > 1:
            h_b_target = self.branch_b(x_exo, x_endo)
        else:
            h_b_target = torch.zeros_like(h_a)

        # Apply Gating & Branch Dropout Schedule on the Target Channel
        current_p = self._get_current_drop_rate()
        if self.training and current_p > 0.0:
            rand_val = torch.rand(1, device=x.device).item()
            if rand_val < current_p / 2:
                fused_target = h_a
            elif rand_val < current_p:
                fused_target = h_b_target
            else:
                g = torch.sigmoid(self.gate)
                fused_target = g * h_a + (1 - g) * h_b_target
        else:
            g = torch.sigmoid(self.gate)
            fused_target = g * h_a + (1 - g) * h_b_target

        # Re-assemble exogenous channels and the fused target channel
        fused = torch.cat([x_exo, fused_target], dim=1) if N > 1 else fused_target

        # Residual connections + FFN
        x = self.norm1(x + self.dropout(fused))
        out = self.norm2(x + self.dropout(self.ffn(x)))

        return out