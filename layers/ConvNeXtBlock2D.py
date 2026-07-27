import torch
import torch.nn as nn


class ConvNeXtBlock2D(nn.Module):
    """
    Modern replacement for Inception blocks utilizing large-kernel depthwise
    convolutions and inverted bottlenecks to maximize GPU processing throughput.
    """

    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super().__init__()
        # Large receptive field to capture variations across wide period lengths
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)

        # Inverted bottleneck projection layer
        mid_channels = in_channels * expansion_factor
        self.pwconv1 = nn.Linear(in_channels, mid_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mid_channels, out_channels)

        # Residual projection shortcut if channel dimensions change
        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x shape: [B, C, H, W]
        # (where C=d_period, H=period, W=length // period)
        residual = self.shortcut(x)

        x = self.dwconv(x)
        # Permute to channels-last layout for standard LayerNorm processing
        x = x.permute(0, 2, 3, 1)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # Restore to channels-first layout

        return x + residual