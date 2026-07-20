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

    def forward(self, x):
        # x shape: [Batch, Variates (N), Time (T), d_model (D)]
        B, N, T, D = x.size()

        if T != self.seq_len:
            raise ValueError(f"ExogenousBranch expected time sequence dimension {self.seq_len}, received {T}.")

        # Collapse temporal features: [B, N, T * D]
        x_flat = x.reshape(B, N, T * D)

        # Project each variate to d_model token embedding -> [B, N, d_model]
        variate_tokens = self.variate_embedding(x_flat)

        # Apply Multi-Head Self-Attention across the channel/variate dimension (N)
        # In iTransformer, Query, Key, and Value are all the set of variate tokens
        attn_out, _ = self.cross_attention(
            variate_tokens,
            variate_tokens,
            variate_tokens,
            attn_mask=None
        )  # Output shape: [B, N, d_model]

        # Project back to full temporal feature shape [B, N, T * D]
        out_flat = self.variate_projection(attn_out)

        # Reshape back to original tensor layout [B, N, T, D]
        out = out_flat.reshape(B, N, T, D)

        return out


class HybridEncoderLayer(nn.Module):
    """
    Fuses isolated Target Periodicity (Branch A) and Multi-Variate Attention (Branch B).
    Processes all N channels for the M (Multivariate-to-Multivariate) forecasting objective.
    """

    def __init__(self, configs, conv_builder):
        super().__init__()
        self.branch_a = PluggablePeriodBlock(configs, conv_builder)
        self.branch_b = ExogenousBranch(configs)

        # Gate parameter
        self.gate = nn.Parameter(torch.zeros(1, 1, 1, configs.d_model))

        # Branch drop hyperparameters
        self.target_branch_dropout = getattr(configs, 'branch_dropout', 0.1)
        self.warmup_epochs = getattr(configs, 'branch_warmup_epochs', 3)
        self.current_epoch = 0

        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)

        # Learnable gate for fusion; broadcasts automatically over the Variates (N) dimension
        self.gate = nn.Parameter(torch.zeros(1, 1, 1, configs.d_model))

        self.norm1 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

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
        # x shape: [Batch, Variates, Time, d_model]
        B, N, T, D = x.size()

        # Branch B: iTransformer Cross-Variate Attention across all channels
        h_b = self.branch_b(x)

        # Branch A: Process temporal periodicity independently for all channels
        x_flat = x.reshape(B * N, T, D)
        h_a_flat = self.branch_a(x_flat)
        h_a = h_a_flat.reshape(B, N, T, D)

        # Determine drop behavior based on current schedule
        current_p = self._get_current_drop_rate()

        if self.training and current_p > 0.0:
            rand_val = torch.rand(1, device=x.device).item()
            if rand_val < current_p / 2:
                # Direct route Branch A
                fused = h_a
            elif rand_val < current_p:
                # Direct route Branch B
                fused = h_b
            else:
                # Convex gated fusion
                g = torch.sigmoid(self.gate)
                fused = g * h_a + (1 - g) * h_b
        else:
            g = torch.sigmoid(self.gate)
            fused = g * h_a + (1 - g) * h_b

        # First residual block: Fusion normalization
        x = self.norm1(x + self.dropout(fused))

        # Second residual block: Non-linear feature transformation via FFN
        out = self.norm2(x + self.dropout(self.ffn(x)))

        return out