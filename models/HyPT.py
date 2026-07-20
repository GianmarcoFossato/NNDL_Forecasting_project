import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.HyPT_EncDec import HybridEncoderLayer
from layers.ConvNeXtBlock2D import ConvNeXtBlock2D
from layers.RevIN import RevIN

class Model(nn.Module):
    """
    Hybrid Period-Transformer (HyPT)
    Period-Aware Cross-Variate Model.
    Designed specifically to handle the ECL dataset constraints by retaining
    a strong periodicity prior while explicitly mixing channel information.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_variates = configs.enc_in

        self.revin = RevIN(self.num_variates) #applied at the start of the whole model


        self.enc_embedding = DataEmbedding(
            1, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Pluggable 2D Conv block factory for Branch A.
        def conv_factory(in_channels, out_channels):
            return nn.Sequential(
                ConvNeXtBlock2D(in_channels, out_channels),
                nn.GELU()
            )

        self.layers = nn.ModuleList([
            HybridEncoderLayer(configs, conv_factory)
            for _ in range(configs.e_layers)
        ])

        # Channel-independent linear head
        self.projection = nn.Linear(self.seq_len * configs.d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, T, N]
        B, T, N = x_enc.size()

        # Instance Normalization
        x_enc = self.revin(x_enc, 'norm')

        # Independent Channel Embedding
        x_enc_flat = x_enc.transpose(1, 2).reshape(B * N, T, 1)
        x_mark_enc_flat = x_mark_enc.repeat_interleave(N, dim=0)

        enc_out = self.enc_embedding(x_enc_flat, x_mark_enc_flat)

        enc_out = enc_out.reshape(B, N, T, self.configs.d_model)

        # Apply L Hybrid Layers
        for layer in self.layers:
            enc_out = layer(enc_out)

        # Projection
        # Flatten Time and d_model: [B, N, T * d_model]
        enc_out = enc_out.reshape(B, N, T * self.configs.d_model)

        # Project to prediction length: [B, N, pred_len]
        dec_out = self.projection(enc_out)

        # Reshape back to standard library format: [B, pred_len, N]
        dec_out = dec_out.transpose(1, 2)

        # Denormalize
        dec_out = self.revin(dec_out, 'denorm')

        # In case of MS, pass target index
        # if dec_out.shape[-1] != self.num_variates:
        #     dec_out = self.revin(dec_out, 'denorm', target_idx=-1)
        # else:
        #     dec_out = self.revin(dec_out, 'denorm')

        return dec_out