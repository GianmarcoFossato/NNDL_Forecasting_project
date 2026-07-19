import torch
import torch.nn as nn
import torch.fft
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class PluggablePeriodBlock(nn.Module):
    """
    Branch A: Period-aware Temporal Prior.
    Mimics TimesNet's FFT-based reshaping but accepts a `conv_builder` function
    so the internal 2D spatial extraction can be swapped easily (e.g., Inception, Dynamic Conv).
    """

    def __init__(self, configs, conv_builder):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        # Instantiate k distinct 2D convolution blocks using the injected factory
        self.conv_blocks = nn.ModuleList([
            conv_builder(configs.d_model, configs.d_model) for _ in range(self.k)
        ])

    def forward(self, x):
        # x: [B * N, T, d_model]
        B_N, T, d_model = x.size()

        # 1. FFT to find top-k periods across the time dimension
        # We use rfft since the input is strictly real.
        xf = torch.fft.rfft(x.transpose(1, 2), dim=-1)  # [B*N, d_model, T//2 + 1]

        # Calculate amplitude to find dominant frequencies
        amplitude = torch.mean(torch.abs(xf), dim=1)  # [B*N, T//2 + 1]
        amplitude[:, 0] = 0  # Ignore DC component

        # Average across the batch/variate dimension to find global top-k periods
        mean_amplitude = amplitude.mean(dim=0)
        _, top_list = torch.topk(mean_amplitude, self.k)

        # Convert frequencies to period lengths
        period_list = [max(T // freq.item(), 1) if freq.item() != 0 else T for freq in top_list]

        res = torch.zeros_like(x)

        # 2. Reshape into 2D, apply Conv2D, and aggregate
        for i, period in enumerate(period_list):
            # Calculate padding if T is not perfectly divisible by the period
            length = ((T + period - 1) // period) * period
            padding = length - T

            # Pad sequence
            if padding > 0:
                pad_tensor = torch.zeros((B_N, padding, d_model), device=x.device)
                x_padded = torch.cat([x, pad_tensor], dim=1)
            else:
                x_padded = x

            # Reshape 1D sequence into 2D grid: [B*N, d_model, period, length // period]
            x_2d = x_padded.reshape(B_N, length // period, period, d_model).permute(0, 3, 2, 1).contiguous()

            # Apply injected 2D spatial block
            out_2d = self.conv_blocks[i](x_2d)

            # Reshape back to 1D
            out_1d = out_2d.permute(0, 3, 2, 1).reshape(B_N, length, d_model)

            # Remove padding and accumulate
            res += out_1d[:, :T, :]

        # Normalize by number of periods combined
        return res / self.k


class iTransformerBranch(nn.Module):
    """
    Branch B: Cross-Variate Prior.
    Projects the temporal history of each variate into a single token, applies
    attention across variates, and projects back, preserving the [T, d_model] structure.
    """

    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len

        # Bottleneck projections: Time * d_model -> d_model
        self.project_in = nn.Linear(self.seq_len * self.d_model, self.d_model)
        self.project_out = nn.Linear(self.d_model, self.seq_len * self.d_model)

        # Standard O(N^2) Attention mixing variate tokens.
        # We reuse the library's standard AttentionLayer.
        self.attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                          output_attention=False),
            configs.d_model, configs.n_heads
        )

    def forward(self, x):
        # x: [B, N, T, d_model]
        B, N, T, D = x.size()

        # Flatten Time and Feature dimensions to treat the whole series as a token
        x_flat = x.reshape(B, N, T * D)

        # Project to d_model for efficient attention [B, N, d_model]
        variate_tokens = self.project_in(x_flat)

        # Mix across N variates
        mixed_tokens, _ = self.attention(variate_tokens, variate_tokens, variate_tokens, attn_mask=None)

        # Expand back to original sequence length
        out_flat = self.project_out(mixed_tokens)
        return out_flat.reshape(B, N, T, D)


class HybridEncoderLayer(nn.Module):
    """
    Fuses Branch A and Branch B using a learned gate.
    Residual connections and LayerNorm are applied to ensure stability over L layers.
    """

    def __init__(self, configs, conv_builder):
        super().__init__()
        self.branch_a = PluggablePeriodBlock(configs, conv_builder)
        self.branch_b = iTransformerBranch(configs)

        # Learned gate to dynamically balance temporal vs cross-variate priors per feature.
        # Initialized to 0.5 (equal trust).
        self.gate = nn.Parameter(torch.ones(1, 1, 1, configs.d_model) * 0.5)

        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        # x: [B, N, T, d_model]
        B, N, T, D = x.size()

        # --- Branch B: Cross-Variate (Applies to all channels) ---
        h_b = self.branch_b(x)  # [B, N, T, d_model]

        # --- Branch A: Temporal Prior (only targets the last channel) ---
        x_target = x[:, -1:, :, :]  # Extract last channel: [B, 1, T, d_model]
        x_bn = x_target.reshape(B, T, D)  # [B, T, d_model]

        h_a_target = self.branch_a(x_bn)  # [B, T, d_model]
        h_a_target = h_a_target.unsqueeze(1)  # [B, 1, T, d_model]

        # --- Gated Fusion for Target vs Direct Assignment for Exogenous ---
        g = torch.sigmoid(self.gate)

        # Combine representations
        fused_output = h_b.clone()
        # Apply the gated blend ONLY to the last channel
        fused_output[:, -1:, :, :] = g * h_a_target + (1 - g) * h_b[:, -1:, :, :]

        x = x + self.dropout(fused_output)
        return self.norm1(x)

        # Optional FF Network (Standard Transformer blocks) could be added here

        return x