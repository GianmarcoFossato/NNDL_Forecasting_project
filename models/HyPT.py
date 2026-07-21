import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
from layers.Embed import PatchEmbedding
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

        # Setup Patching Parameters
        self.patch_len = configs.patch_len
        self.stride = self.patch_len // 2  # 50% overlap is standard
        self.padding = self.stride

        # Calculate the new temporal dimension (number of patches)
        # Formula for unfold: (L_in - patch_len) // stride + 1
        L_in = self.seq_len + self.padding
        self.num_patches = (L_in - self.patch_len) // self.stride + 1

        # 3. Replace DataEmbedding with your PatchEmbedding
        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            padding=self.padding,
            dropout=configs.dropout
        )

        # Pluggable 2D Conv block factory for Branch A.
        def conv_factory(in_channels, out_channels):
            return nn.Sequential(
                ConvNeXtBlock2D(in_channels, out_channels)
            )

        self.layers = nn.ModuleList([
            HybridEncoderLayer(configs, conv_factory)
            for _ in range(configs.e_layers)
        ])

        # Channel-independent linear head
        self.projection = nn.Linear(self.num_patches * configs.d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, T, N]
        B, T, N = x_enc.size()

        # Instance Normalization
        x_enc = self.revin(x_enc, 'norm')

        # Patch Embedding (Channel Independent)
        # Transpose to [B, N, T] for your PatchEmbedding class
        x_enc_patched = x_enc.transpose(1, 2)

        # enc_out shape: [B*N, num_patches, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc_patched)

        # Reshape for the Hybrid Layers: [B, N, num_patches, d_model]
        enc_out = enc_out.reshape(B, n_vars, self.num_patches, self.configs.d_model)

        # Apply L Hybrid Layers
        for layer in self.layers:
            enc_out = layer(enc_out)

        # Projection
        # Flatten Time (now num_patches) and d_model: [B, N, num_patches * d_model]
        enc_out = enc_out.reshape(B, N, self.num_patches * self.configs.d_model)

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