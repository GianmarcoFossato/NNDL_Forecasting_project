import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.Hybrid_EncDec import HybridEncoderLayer
from layers.ConvNeXtBlock2D import ConvNeXtBlock2D


# Make sure to import RevIN correctly from your repo structure
# from layers.RevIN import RevIN

class Model(nn.Module):
    """
    Hybrid Period-Aware Cross-Variate Model.
    Designed specifically to handle the ECL dataset constraints by retaining
    a strong periodicity prior while explicitly mixing channel information.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_variates = configs.enc_in

        # RevIN is non-negotiable for distribution shift in ECL
        # self.revin = RevIN(self.num_variates)

        # We embed each channel independently to maintain TimesNet's channel-agnostic temporal prior
        # Input to embedding will be reshaped from [B, T, N] to [B*N, T, 1]
        self.enc_embedding = DataEmbedding(
            1, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Define the pluggable 2D Conv block factory.
        # Using Inception by default to replicate TimesNet, but abstracted.
        from layers.Conv_Blocks import Inception_Block_V1  # Assuming it exists in your repo
        def conv_factory(in_channels, out_channels):
            return nn.Sequential(
                ConvNeXtBlock2D(in_channels, out_channels),
                nn.GELU()
            )

        self.layers = nn.ModuleList([
            HybridEncoderLayer(configs, conv_factory)
            for _ in range(configs.e_layers)
        ])

        # Channel-independent linear head (PatchTST / iTransformer style)
        # Prevents the head from destroying the carefully balanced cross-variate representations
        self.projection = nn.Linear(self.seq_len * configs.d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, T, N]
        B, T, N = x_enc.size()

        # 1. Instance Normalization
        # x_enc = self.revin(x_enc, 'norm')

        # 2. Independent Channel Embedding
        # Reshape to [B*N, T, 1] so each variate is embedded using the same shared weights
        x_enc_flat = x_enc.transpose(1, 2).reshape(B * N, T, 1)
        x_mark_enc_flat = x_mark_enc.repeat(N, 1, 1)  # Repeat time markings for all variates

        # Embed to [B*N, T, d_model]
        enc_out = self.enc_embedding(x_enc_flat, x_mark_enc_flat)

        # Reshape to our isolated dimensions: [B, N, T, d_model]
        enc_out = enc_out.reshape(B, N, T, self.configs.d_model)

        # 3. Apply L Hybrid Layers
        for layer in self.layers:
            enc_out = layer(enc_out)

        # 4. Projection
        # Flatten Time and d_model: [B, N, T * d_model]
        enc_out = enc_out.reshape(B, N, T * self.configs.d_model)

        # Project to prediction length: [B, N, pred_len]
        dec_out = self.projection(enc_out)

        # Reshape back to standard library format: [B, pred_len, N]
        dec_out = dec_out.transpose(1, 2)

        # 5. Denormalize
        # dec_out = self.revin(dec_out, 'denorm')

        return dec_out