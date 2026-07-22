import torch
import torch.nn as nn
import torch.fft
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class PrototypePeriodBlock(nn.Module):
    """
    Branch A (Factorized): Runs FFT and 2D ConvNeXt processing on K learned prototype
    temporal sequences (K << N) rather than all N channels independently.
    Reduces compute complexity from O(B * N) to O(B * K).
    """

    def __init__(self, configs, conv_builder, n_prototypes=16):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.n_proto = n_prototypes

        self.d_period = configs.d_period if getattr(configs, 'd_period', None) else max(configs.d_model // 4, 8)
        self.down_proj = nn.Linear(configs.d_model, self.d_period)
        self.up_proj = nn.Linear(self.d_period, configs.d_model)

        # Soft clustering projections (Channels <-> K Prototypes)
        # Accepts [temporal_summary, spectral_summary] of size 2 * d_period
        self.proto_query = nn.Linear(2 * self.d_period, self.d_period)
        self.proto_keys = nn.Parameter(torch.randn(n_prototypes, self.d_period) * 0.02)

        # 2D ConvNeXt blocks operating solely in Prototype Space
        self.conv_blocks = nn.ModuleList([
            conv_builder(self.d_period, self.d_period) for _ in range(self.k)
        ])

    def _run_period_conv(self, proto_seq):
        # proto_seq shape: [B, K, P, d_period]
        B, K, P, d_period = proto_seq.size()

        # FFT across Patches (P) per batch item and prototype -> shape [B, K, d_period, P_fft]
        xf = torch.fft.rfft(proto_seq.transpose(2, 3), dim=-1)

        # Average amplitude across d_period dimension -> [B, K, P_fft]
        amplitude = torch.mean(torch.abs(xf), dim=2)
        amplitude[:, :, 0] = 0.0  # Ignore DC component

        # Average amplitude across Batch (B) -> [K, P_fft] per prototype
        proto_amplitude = amplitude.mean(dim=0)  # [K, P_fft]

        # Select top-k frequencies independently PER PROTOTYPE: [K, k]
        _, top_list_per_proto = torch.topk(proto_amplitude, self.k, dim=-1)

        res = torch.zeros_like(proto_seq)  # [B, K, P, d_period]

        # Process each prototype k with its specialized set of top periods
        for k_idx in range(K):
            x_k = proto_seq[:, k_idx, :, :]  # [B, P, d_period]
            top_list_k = top_list_per_proto[k_idx]  # [k]

            # Compute period weights for prototype k
            per_sample_amp_k = amplitude[:, k_idx, top_list_k]  # [B, k]
            period_weights_k = torch.softmax(per_sample_amp_k, dim=-1)  # [B, k]

            period_list_k = [
                max(P // freq.item(), 1) if freq.item() != 0 else P
                for freq in top_list_k
            ]

            res_k = torch.zeros_like(x_k)  # [B, P, d_period]

            for i, period in enumerate(period_list_k):
                length = ((P + period - 1) // period) * period
                padding = length - P

                if padding > 0:
                    pad_tensor = torch.zeros(
                        (B, padding, d_period),
                        dtype=x_k.dtype,
                        device=x_k.device,
                    )
                    x_padded = torch.cat([x_k, pad_tensor], dim=1)
                else:
                    x_padded = x_k

                x_2d = (
                    x_padded.reshape(B, length // period, period, d_period)
                    .permute(0, 3, 2, 1)
                    .contiguous()
                )
                out_2d = self.conv_blocks[i](x_2d)
                out_2d = out_2d.permute(0, 3, 2, 1).contiguous()
                out_1d = out_2d.reshape(B, length, d_period)

                w = period_weights_k[:, i].view(B, 1, 1)
                res_k = res_k + (out_1d[:, :P, :] * w)

            res[:, k_idx, :, :] = res_k

        return res

    def forward(self, x):
        # x shape: [B, N, P, D]
        B, N, P, D = x.size()
        x_down = self.down_proj(x)  # [B, N, P, d_period]

        # Frequency-aware channel profiles combining temporal DC mean + spectral energy profile
        temporal_summary = x_down.mean(dim=2)  # [B, N, d_period]
        spectral_summary = torch.fft.rfft(x_down, dim=2).abs().mean(dim=2)  # [B, N, d_period]
        channel_summary = torch.cat([temporal_summary, spectral_summary], dim=-1)  # [B, N, 2 * d_period]

        # Soft-assignment weights between N channels and K prototypes
        q = self.proto_query(channel_summary)  # [B, N, d_period]
        scores = q @ self.proto_keys.T / (self.d_period ** 0.5)  # [B, N, K]

        pool_weights = torch.softmax(scores, dim=1)  # Softmax across channels (N -> K)
        recomb_weights = torch.softmax(scores, dim=-1)  # Softmax across prototypes (K -> N)

        # Compress N channels into K prototype sequences
        proto_seq = torch.einsum('bnk,bnpd->bkpd', pool_weights, x_down)  # [B, K, P, d_period]

        # FFT + 2D ConvNeXt with per-prototype period specialization
        proto_out = self._run_period_conv(proto_seq)  # [B, K, P, d_period]

        # Recombine prototypes back to N individual channels
        out = torch.einsum('bnk,bkpd->bnpd', recomb_weights, proto_out)  # [B, N, P, d_period]

        return self.up_proj(out)


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
        self.ablation_mode = getattr(configs, 'ablation_mode', 'both')

        # Instantiation based on mode
        if self.ablation_mode in ['both', 'branch_a']:
            self.branch_a = PrototypePeriodBlock(configs, conv_builder)

        if self.ablation_mode in ['both', 'branch_b']:
            self.branch_b = CrossVariateBranch(configs)

        if self.ablation_mode == 'both':
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
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_gate_value(self) -> float:
        """Returns the current sigmoid(gate) value for logging."""
        if hasattr(self, 'gate'):
            return torch.sigmoid(self.gate).detach().cpu().numpy().mean().item()
        return None

    def _get_current_drop_rate(self) -> float:
        if not self.training or self.warmup_epochs <= 0:
            return self.target_branch_dropout
        if self.current_epoch < self.warmup_epochs:
            return self.target_branch_dropout * (
                    self.current_epoch / self.warmup_epochs
            )
        return self.target_branch_dropout

    def forward(self, x):
        # Branch Routing
        if self.ablation_mode == 'branch_a':
            fused = self.branch_a(x)

        elif self.ablation_mode == 'branch_b':
            fused = self.branch_b(x)

        elif self.ablation_mode in ['none', 'baseline']:
            # Floor comparison baseline: bypasses branches entirely
            fused = torch.zeros_like(x)

        else:  # 'both'
            B, N, P, D = x.size()
            h_a = self.branch_a(x)
            h_b = self.branch_b(x)

            current_p = self._get_current_drop_rate()
            if self.training and current_p > 0.0:
                rand_val = torch.rand(B, 1, 1, 1, device=x.device)
                g = torch.sigmoid(self.gate)
                fused_gate = g * h_a + (1 - g) * h_b

                a_only_mask = (rand_val < current_p / 2).float()
                b_only_mask = (
                        (rand_val >= current_p / 2) & (rand_val < current_p)
                ).float()
                default_mask = 1.0 - a_only_mask - b_only_mask

                fused = a_only_mask * h_a + b_only_mask * h_b + default_mask * fused_gate
            else:
                g = torch.sigmoid(self.gate)
                fused = g * h_a + (1 - g) * h_b

        x = self.norm1(x + self.dropout(fused))
        out = self.norm2(x + self.dropout(self.ffn(x)))

        return out