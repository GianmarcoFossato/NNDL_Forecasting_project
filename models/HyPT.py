import torch
import torch.nn as nn
from layers.Embed import PatchEmbedding
from layers.HyPT_EncDec import HybridEncoderLayer
from layers.ConvNeXtBlock2D import ConvNeXtBlock2D
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    Hybrid Period-Transformer (HyPT) - Optimized
    Period-Aware Cross-Variate Model with Prototype Period Factorization
    and Linear Trend Residual.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_variates = configs.enc_in

        self.revin = RevIN(self.num_variates)

        # Setup Patching Parameters
        self.patch_len = configs.patch_len
        self.stride = self.patch_len // 2
        self.padding = self.stride

        L_in = self.seq_len + self.padding
        self.num_patches = (L_in - self.patch_len) // self.stride + 1

        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            padding=self.padding,
            dropout=configs.dropout
        )

        def conv_factory(in_channels, out_channels):
            return nn.Sequential(
                ConvNeXtBlock2D(in_channels, out_channels)
            )

        self.layers = nn.ModuleList([
            HybridEncoderLayer(configs, conv_factory)
            for _ in range(configs.e_layers)
        ])

        # Head Dropout for Projection Regularization
        self.head_dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.num_patches * configs.d_model, self.pred_len)

        # Parallel Channel-Independent Trend Residual Path
        self.trend_proj = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, T, N]
        B, T, N = x_enc.size()

        # 1. Instance Normalization
        x_enc = self.revin(x_enc, 'norm')

        # 2. Compute Linear Trend Baseline Path
        trend_out = self.trend_proj(x_enc.transpose(1, 2))  # [B, N, pred_len]

        # 3. Patching & Hybrid Encoder Path
        x_enc_patched = x_enc.transpose(1, 2)  # [B, N, T]
        enc_out, n_vars = self.patch_embedding(x_enc_patched)

        # Reshape to [B, N, num_patches, d_model]
        enc_out = enc_out.reshape(B, n_vars, self.num_patches, self.configs.d_model)

        for layer in self.layers:
            enc_out = layer(enc_out)

        # Flatten & Project with Head Dropout
        enc_out = enc_out.reshape(B, N, self.num_patches * self.configs.d_model)
        dec_out = self.projection(self.head_dropout(enc_out))  # [B, N, pred_len]

        # Add Trend Component
        dec_out = dec_out + trend_out

        # Reshape to library expectation: [B, pred_len, N]
        dec_out = dec_out.transpose(1, 2)

        # Denormalize
        dec_out = self.revin(dec_out, 'denorm')

        return dec_out