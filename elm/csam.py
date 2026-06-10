"""
CSAM_UNet2p5D: a 2.5D U-Net whose per-slice 2D backbone is augmented with a
faithful, uncertainty-aware Cross-Slice Attention Module (CSAM) inserted on the
skip features at every encoder level.

Interface (matches the rest of the pipeline):
    input : [B, 1, D, H, W]
    output: [B, 1, D, H, W]  (logits, for BCEWithLogitsLoss + Dice)

The CSAM here reproduces the original three-branch design
(semantic / positional / slice) including the low-rank-Gaussian uncertainty
formulation in the slice branch, adapted from the original 4D [D,C,H,W] layout
to a batched 5D [B,D,C,H,W] layout.
"""

import torch
import torch.nn as nn
import torch.distributions as td


# =============================================================================
# 2D building blocks (encoder / decoder)
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, max_pool, return_single=False):
        super().__init__()
        self.max_pool = max_pool
        self.return_single = return_single

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
        )
        if max_pool:
            self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        b = x
        if self.max_pool:
            x = self.pool(x)
        if self.return_single:
            return x
        return x, b


class DeconvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, intermediate_channels=-1):
        super().__init__()
        input_channels = int(input_channels)
        output_channels = int(output_channels)

        if intermediate_channels < 0:
            intermediate_channels = output_channels * 2
        else:
            intermediate_channels = input_channels

        self.upconv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(input_channels, intermediate_channels // 2, 3, 1, 1),
        )
        self.conv = ConvBlock(intermediate_channels, output_channels, max_pool=False)

    def forward(self, x, b):
        x = self.upconv(x)
        x = torch.cat((x, b), dim=1)
        x, _ = self.conv(x)
        return x


class UNetDecoder(nn.Module):
    """Expects skips in decoder order: high-res -> low-res."""
    def __init__(self, num_layers, base_num):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers - 1, 0, -1):
            self.blocks.append(
                DeconvBlock(base_num * (2 ** i), base_num * (2 ** (i - 1)))
            )

    def forward(self, x, skips):
        assert len(skips) == len(self.blocks), \
            f"Expected {len(self.blocks)} skips, got {len(skips)}"
        for blk, skip in zip(self.blocks, skips):
            x = blk(x, skip)
        return x


# =============================================================================
# Faithful CSAM, batched to [B, D, C, H, W]
# =============================================================================
class SemanticAttention5D(nn.Module):
    """Channel attention. Reduces over (D, H, W) per sample -> per-channel gate."""
    def __init__(self, channels: int, reduction_rate: int = 16):
        super().__init__()
        hidden = max(channels // reduction_rate, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        B, D, C, H, W = x.shape
        max_x = x.amax(dim=(1, 3, 4))                  # [B, C]
        avg_x = x.mean(dim=(1, 3, 4))                  # [B, C]
        att = self.mlp(max_x) + self.mlp(avg_x)        # shared MLP, as in original
        att = torch.sigmoid(att).view(B, 1, C, 1, 1)
        return x * att


class PositionalAttention5D(nn.Module):
    """Spatial attention. Reduces over (D, C) per sample -> one [H,W] map -> 7x7 conv."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        B, D, C, H, W = x.shape
        max_x = x.amax(dim=(1, 2))                      # [B, H, W]  (over D and C)
        avg_x = x.mean(dim=(1, 2))                      # [B, H, W]
        att = torch.cat([max_x.unsqueeze(1), avg_x.unsqueeze(1)], dim=1)  # [B, 2, H, W]
        att = torch.sigmoid(self.conv(att)).view(B, 1, 1, H, W)
        return x * att


class SliceAttention5D(nn.Module):
    """
    Slice attention with low-rank-Gaussian uncertainty (faithful to original).
    Reduces over (C, H, W) per sample -> per-slice descriptor of length D.
    Expansion MLP (D -> D*rate -> D), max+avg, optional sampling.
    """
    def __init__(self, num_slices: int, rate: int = 4, uncertainty: bool = True, rank: int = 5):
        super().__init__()
        self.uncertainty = uncertainty
        self.rank = rank
        hidden = int(num_slices * rate)
        self.linear = nn.Sequential(
            nn.Linear(num_slices, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_slices),
        )
        if uncertainty:
            self.non_linear = nn.ReLU()
            self.mean = nn.Linear(num_slices, num_slices)
            self.log_diag = nn.Linear(num_slices, num_slices)
            self.factor = nn.Linear(num_slices, num_slices * rank)

    def forward(self, x):
        B, D, C, H, W = x.shape
        max_x = x.amax(dim=(2, 3, 4))                   # [B, D]
        avg_x = x.mean(dim=(2, 3, 4))                   # [B, D]
        att = self.linear(max_x) + self.linear(avg_x)   # [B, D]

        if self.uncertainty:
            temp = self.non_linear(att)                 # [B, D]
            mean = self.mean(temp)                      # [B, D]
            diag = self.log_diag(temp).exp()            # [B, D]  (positive)
            factor = self.factor(temp).view(B, D, self.rank)   # [B, D, rank]
            dist = td.LowRankMultivariateNormal(loc=mean, cov_factor=factor, cov_diag=diag)
            # reparameterised sample in training (grads flow); mean at eval (deterministic)
            att = dist.rsample() if self.training else mean

        att = torch.sigmoid(att).view(B, D, 1, 1, 1)
        return x * att


class CSAM5D(nn.Module):
    """Faithful CSAM on batched volumes [B, D, C, H, W]. num_slices = depth D at this level."""
    def __init__(self, num_slices: int, num_channels: int,
                 semantic: bool = True, positional: bool = True, slice_att: bool = True,
                 uncertainty: bool = True, rank: int = 5):
        super().__init__()
        self.semantic_on = semantic
        self.positional_on = positional
        self.slice_on = slice_att
        if semantic:
            self.semantic = SemanticAttention5D(num_channels)
        if positional:
            self.positional = PositionalAttention5D(kernel_size=7)
        if slice_att:
            self.slice = SliceAttention5D(num_slices, uncertainty=uncertainty, rank=rank)

    def forward(self, x):
        if self.semantic_on:
            x = self.semantic(x)
        if self.positional_on:
            x = self.positional(x)
        if self.slice_on:
            x = self.slice(x)
        return x


# =============================================================================
# Encoder with CSAM on skip features + bottleneck
# =============================================================================
class EncoderCSAM5D(nn.Module):
    """
    Slice-wise 2D encoder with volume CSAM inserted on skip features and bottleneck.
    input : [B, 1, D, H, W]
    """
    def __init__(self, input_channels, num_layers, base_num, num_slices,
                 semantic=True, positional=True, slice_att=True,
                 uncertainty=True, rank=5):
        super().__init__()
        self.num_layers = num_layers

        self.blocks = nn.ModuleList()
        self.attn = nn.ModuleList()

        for i in range(num_layers):
            in_ch = input_channels if i == 0 else base_num * (2 ** (i - 1))
            out_ch = base_num * (2 ** i)
            use_pool = (i != num_layers - 1)  # no pool at last (bottleneck)
            self.blocks.append(ConvBlock(in_ch, out_ch, max_pool=use_pool))
            self.attn.append(
                CSAM5D(num_slices=num_slices, num_channels=out_ch,
                       semantic=semantic, positional=positional, slice_att=slice_att,
                       uncertainty=uncertainty, rank=rank)
            )

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"Expected input [B,1,D,H,W], got {x.shape}")

        B, C0, D, H, W = x.shape

        # Flatten volume into slice-batch: [B*D, C, H, W]
        xs = x.permute(0, 2, 1, 3, 4).reshape(B * D, C0, H, W)

        skips_2d = []
        cur = xs

        for i in range(self.num_layers):
            cur, skip = self.blocks[i](cur)            # skip: [B*D, Ci, Hi, Wi]
            Ci, Hi, Wi = skip.shape[1], skip.shape[2], skip.shape[3]

            # Apply CSAM in 5D space
            skip5d = skip.view(B, D, Ci, Hi, Wi)
            skip5d = self.attn[i](skip5d)
            skip = skip5d.reshape(B * D, Ci, Hi, Wi)

            if i != self.num_layers - 1:
                skips_2d.append(skip)
            else:
                cur = skip

        # Decoder expects high-res first; collected shallow->deep, so reverse.
        skips_2d = skips_2d[::-1]
        return cur, skips_2d, (B, D)


class CSAM_UNet2p5D(nn.Module):
    """
    End-to-end 2.5D segmentation model with faithful uncertainty-aware CSAM.
      input : [B, 1, D, H, W]
      output: [B, 1, D, H, W]  (logits)
    """
    def __init__(self, in_channels=1, out_channels=1, num_layers=5, base_num=32,
                 num_slices=49, semantic=True, positional=True, slice_att=True,
                 uncertainty=True, rank=5):
        super().__init__()
        self.n_classes = out_channels
        self.encoder = EncoderCSAM5D(
            in_channels, num_layers, base_num, num_slices,
            semantic=semantic, positional=positional, slice_att=slice_att,
            uncertainty=uncertainty, rank=rank,
        )
        self.decoder = UNetDecoder(num_layers, base_num)
        self.head = nn.Conv2d(base_num, out_channels, kernel_size=1)

    def forward(self, x):
        bottleneck, skips, meta = self.encoder(x)        # [B*D, Cb, hb, wb]
        y2d = self.decoder(bottleneck, skips)            # [B*D, base_num, H, W]
        y2d = self.head(y2d)                             # [B*D, out, H, W]

        B, D = meta
        H, W = y2d.shape[-2], y2d.shape[-1]
        y = y2d.view(B, D, self.n_classes, H, W).permute(0, 2, 1, 3, 4)  # [B, out, D, H, W]
        return y