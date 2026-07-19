import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
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
        # x: [B, T, d_model]
        B, T, d_model = x.size()

        # FFT to find top-k periods across the time dimension
        xf = torch.fft.rfft(x.transpose(1, 2), dim=-1)

        # Calculate amplitude to find dominant frequencies
        amplitude = torch.mean(torch.abs(xf), dim=1)
        amplitude[:, 0] = 0  # Ignore DC component

        # Average across the batch/variate dimension to find global top-k periods
        mean_amplitude = amplitude.mean(dim=0)
        _, top_list = torch.topk(mean_amplitude, self.k)

        # Convert frequencies to period lengths
        period_list = [max(T // freq.item(), 1) if freq.item() != 0 else T for freq in top_list]
        res = torch.zeros_like(x)

        # Reshape into 2D, apply Conv2D, and aggregate
        for i, period in enumerate(period_list):
            # Padding if T is not perfectly divisible by the period
            length = ((T + period - 1) // period) * period
            padding = length - T

            if padding > 0:
                pad_tensor = torch.zeros((B, padding, d_model), device=x.device)
                x_padded = torch.cat([x, pad_tensor], dim=1)
            else:
                x_padded = x

            # Reshape into a 2D grid matrix structure: [B, d_model, period, length // period]
            x_2d = x_padded.reshape(B, length // period, period, d_model).permute(0, 3, 2, 1).contiguous()
            out_2d = self.conv_blocks[i](x_2d)
            out_1d = out_2d.permute(0, 3, 2, 1).reshape(B, length, d_model)

            res += out_1d[:, :T, :]

        return res / self.k


class iTransformerBranch(nn.Module):
    """
    Branch B: Cross-Variate Prior.
    Maps temporal tokens across exogenous and target features global variants maps.
    """

    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len

        # Bottleneck projections: Time * d_model -> d_model
        self.project_in = nn.Linear(self.seq_len * self.d_model, self.d_model)
        self.project_out = nn.Linear(self.d_model, self.seq_len * self.d_model)

        # Standard O(N^2) Attention mixing variate tokens
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
    Fuses isolated Target Periodicity (Branch A) and Multi-Variate Attention (Branch B).
    """

    def __init__(self, configs, conv_builder):
        super().__init__()
        self.branch_a = PluggablePeriodBlock(configs, conv_builder)
        self.branch_b = iTransformerBranch(configs)

        # Learned gate to dynamically balance temporal vs cross-variate priors per feature.
        # Initialized to 0.5 (equal trust).
        self.gate = nn.Parameter(torch.ones(1, 1, 1, configs.d_model) * 0.5)
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        # x: [B, N, T, d_model]
        B, N, T, D = x.size()

        # Branch B: evaluates all cross-variate attributes
        h_b = self.branch_b(x)  # [B, N, T, d_model]

        # Branch A: processing is strict to target channel (assumed index -1)
        x_target = x[:, -1, :, :]  # Shape: [B, T, d_model]
        h_a_target = self.branch_a(x_target).unsqueeze(1)  # Shape: [B, 1, T, d_model]

        # gated fusion for target vs direct assignment for exogenous
        g = torch.sigmoid(self.gate)
        fused_output = h_b.clone()

        # Merge target representation explicitly. Exogeneous bypasses Branch A
        fused_output[:, -1:, :, :] = g * h_a_target + (1 - g) * h_b[:, -1:, :, :]

        x = x + self.dropout(fused_output)
        return self.norm1(x)