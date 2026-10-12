"""Foundation-model encoder + dense-prediction decoder for 2D ELM segmentation.

The encoders live in :mod:`elm.fm_encoders` (RETFound and its controls, MIRAGE,
MedSAM); this module holds the decoders and the segmentation module that couples
them, plus the layer-wise-LR-decay parameter grouping.

Why the decoder matters here
----------------------------
A plain ViT has no high-resolution stage anywhere in the encoder (unlike Swin,
which gives a stride-4 stage-0 feature map): a ViT-L/16 at 320x512 produces a
20x32 token grid and nothing finer. The ELM is a 1-2 px membrane, so recovering it
is entirely a question of how much spatial detail the decoder can reconstruct.

Three decoders, chosen so each ablation isolates exactly one factor:

- ``dpt`` (default) -- DPT-style (Ranftl et al., ICCV 2021) reassembly of tokens
  from four transformer blocks into a 1/4, 1/8, 1/16, 1/32 pyramid, fused
  coarse-to-fine RefineNet-style. Reassembly strides are derived from the encoder's
  patch size, so a patch-32 encoder (MIRAGE) gets a correctly-scaled pyramid rather
  than one silently an octave too coarse.
- ``use_stem=True`` (default) adds a full-resolution convolutional path over the
  raw image (the UNETR ``encoder0`` idea) supplying 1/2 and 1/1 skips, so the final
  two fusion stages see genuine full-resolution detail rather than an upsampled
  token grid. ``use_stem`` is a separate flag from ``decoder`` precisely so the
  ablation can read ``dpt+stem`` / ``dpt-stem`` / ``convnext`` / ``linear_probe``
  and attribute the difference to one thing at a time.
- ``convnext`` -- a ConvNeXt decoder on the last block's tokens. With
  ``freeze_encoder=True`` this is the MIRAGE benchmark protocol (Morano et al.),
  and it is the only configuration directly comparable to their published RETFound
  row.
- ``linear_probe`` -- a 1x1-conv probe on the last block. This is what MIRAGE used
  in their *pretraining ablations*, not in their benchmark; it is here as the floor
  that shows how much of the result is decoder rather than encoder.

Normalisation is GroupNorm throughout. ViT-L at these resolutions with a full-res
32-channel path forces batches of 2-4, where BatchNorm statistics are noise.

NOTE ON CHECKPOINT COMPATIBILITY: the encoder now lives under ``encoder.vit.*``
(it was ``encoder.*``) and every decoder norm is GroupNorm rather than BatchNorm.
Checkpoints trained before these repairs cannot be loaded and must be retrained.

References: DPT (arXiv:2103.13413); UNETR (arXiv:2103.10504); ConvNeXt
(arXiv:2201.03545); RETFound (Zhou et al., Nature 2023); MIRAGE (arXiv:2506.08900).
"""

import logging
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from elm.fm_encoders import (  # noqa: F401  (re-exported for callers and sbatch scripts)
    ENCODER_SOURCES,
    _resolve_checkpoint,
    build_encoder,
    encoder_arch,
    load_pretrained_encoder,
)

DECODERS = ("dpt", "convnext", "linear_probe")

#: Reassembly targets: output stride of each of the four taps, relative to the input.
_PYRAMID_STRIDES = (4, 8, 16, 32)


def _gn(ch: int) -> nn.GroupNorm:
    """GroupNorm with at most 32 groups -- a drop-in for BatchNorm2d at small batch."""
    return nn.GroupNorm(min(32, ch), ch)


def _as_hw(size):
    """Normalise an int-or-pair size to a (h, w) tuple."""
    if isinstance(size, int):
        return (size, size)
    h, w = size
    return (int(h), int(w))


class ResidualConvUnit(nn.Module):
    """DPT residual conv unit, GroupNorm variant."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn1 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn2 = _gn(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.gn1(self.conv1(self.act(x)))
        out = self.gn2(self.conv2(self.act(out)))
        return out + x


class FeatureFusion(nn.Module):
    """DPT feature-fusion block: refine skip, add, refine, upsample, project.

    Sizes are matched to the skip rather than assumed, so the block works for any
    input resolution (including odd, non-power-of-two token grids, which arise as
    soon as inputs are padded to their own next patch multiple rather than to a
    global canvas).
    """

    def __init__(self, in_ch, skip_ch, out_ch, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.skip_proj = (
            nn.Conv2d(skip_ch, in_ch, 1, bias=False) if skip_ch and skip_ch != in_ch else nn.Identity()
        )
        self.res_skip = ResidualConvUnit(in_ch) if skip_ch else None
        self.res_out = ResidualConvUnit(in_ch)
        self.out_conv = nn.Conv2d(in_ch, out_ch, 1, bias=True)

    def forward(self, x, skip=None):
        if skip is not None:
            skip = self.res_skip(self.skip_proj(skip))
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + skip
        x = self.res_out(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.out_conv(x)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (7x7 depthwise -> LN -> 4x MLP), channels-first wrapper."""

    def __init__(self, ch, layer_scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.norm = nn.LayerNorm(ch, eps=1e-6)
        self.pw1 = nn.Linear(ch, 4 * ch)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * ch, ch)
        self.gamma = nn.Parameter(layer_scale * torch.ones(ch))

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pw2(self.act(self.pw1(self.norm(x)))) * self.gamma
        return residual + x.permute(0, 3, 1, 2)


class RETFoundSeg2D(nn.Module):
    """Foundation-model encoder + dense-prediction decoder.

    Input:  [B, n_channels, H, W]   (H and W each a multiple of the encoder patch size)
    Output: [B, n_classes, H, W]    logits

    Matches the forward contract of every other 2D model in ``elm/model.py``, so it
    plugs into train2D.py, predict_cv2d.py, the native-resolution scoring path and
    the hole-decomposition metrics without changes to any of them.

    ``img_size`` only sizes the encoder's *initial* position-embedding grid; the ViT
    arms are built with ``dynamic_img_size`` so any patch-multiple rectangle can be
    passed at any time, which is what §3.3 of the plan requires (native-scale crops
    for training, per-image padding for inference).
    """

    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        img_size=(320, 512),
        encoder_weights="retfound_oct",
        encoder_name="vit_large_patch16_224",
        decoder="dpt",
        use_stem=True,
        features=256,
        freeze_encoder=False,
        drop_path_rate=0.2,
        taps=None,
        prior_prob=0.01,
        strict_min_match=0.9,
        load_encoder_weights=True,
    ):
        super().__init__()
        if decoder not in DECODERS:
            raise ValueError(f"decoder must be one of {DECODERS}, got '{decoder}'")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = _as_hw(img_size)
        self.decoder_kind = decoder
        self.use_stem = bool(use_stem) and decoder == "dpt"
        self.freeze_encoder = freeze_encoder

        self.encoder = build_encoder(
            encoder_weights=encoder_weights,
            encoder_name=encoder_name,
            img_size=self.img_size,
            in_chans=n_channels,
            drop_path_rate=drop_path_rate,
            taps=taps,
            strict_min_match=strict_min_match,
            load_weights=load_encoder_weights,
        )

        embed_dim = self.encoder.embed_dim
        patch = self.encoder.patch_size
        for axis, n in zip(("height", "width"), self.img_size):
            if n % patch:
                raise ValueError(
                    f"img_size {axis} {n} is not a multiple of the encoder patch size {patch}"
                )
        self.patch_size = patch
        self.taps = self.encoder.taps

        # Per-tap LayerNorm: the residual stream grows in scale with depth, so taps
        # from block 5 and block 23 arrive on very different scales. Normalising each
        # tap before reassembly keeps the fusion stages well conditioned.
        self.tap_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in self.taps])

        if decoder == "dpt":
            self._build_dpt_decoder(embed_dim, features, n_channels, n_classes, patch)
        elif decoder == "convnext":
            self._build_convnext_decoder(embed_dim, features, n_classes, patch)
        else:
            self._build_linear_probe(embed_dim, features, n_classes)

        # Bias the head towards "background" at init. The ELM is well under 1% of
        # pixels in this dataset, and starting the head near that prior avoids the
        # early all-positive collapse BCE+Dice can fall into on such a thin class.
        nn.init.constant_(
            self.head[-1].bias,
            -torch.log(torch.tensor((1 - prior_prob) / prior_prob)).item(),
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    # -- decoders -----------------------------------------------------------

    @staticmethod
    def _reassemble_stage(embed_dim, out_ch, patch, target_stride):
        """Resample a token grid of stride ``patch`` to stride ``target_stride``.

        Patch 16 -> (4, 8, 16, 32) gives DPT's canonical stride-4 / stride-2 /
        identity / stride-2-down ladder. Patch 32 (MIRAGE) needs stride-8 and
        stride-4 transposed convolutions on the first two taps instead; reusing the
        patch-16 strides there would produce a pyramid one octave too coarse, with
        no error to show for it.
        """
        proj = nn.Conv2d(embed_dim, out_ch, 1)
        if patch == target_stride:
            return nn.Sequential(proj)
        if patch > target_stride:
            f = patch // target_stride
            if patch % target_stride:
                raise ValueError(f"patch {patch} does not divide evenly to stride {target_stride}")
            return nn.Sequential(proj, nn.ConvTranspose2d(out_ch, out_ch, kernel_size=f, stride=f))
        f = target_stride // patch
        return nn.Sequential(proj, nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=f, padding=1))

    def _build_dpt_decoder(self, embed_dim, features, n_channels, n_classes, patch):
        widths = (256, 512, 1024, 1024)
        self.reassemble = nn.ModuleList([
            self._reassemble_stage(embed_dim, c, patch, s)
            for c, s in zip(widths, _PYRAMID_STRIDES)
        ])
        self.rn = nn.ModuleList([nn.Conv2d(c, features, 3, padding=1, bias=False) for c in widths])

        #  1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2
        self.fuse4 = FeatureFusion(features, 0, features)
        self.fuse3 = FeatureFusion(features, features, features)
        self.fuse2 = FeatureFusion(features, features, features)
        self.fuse1 = FeatureFusion(features, features, 128)

        if self.use_stem:
            # Full-resolution convolutional path over the raw image. Without this
            # every decoder feature traces back to the token grid and the predicted
            # membrane is a blurred band; this is what lets the last fusions place a
            # 1-2 px line.
            self.stem_full = nn.Sequential(
                nn.Conv2d(n_channels, 32, 3, padding=1, bias=False), _gn(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1, bias=False), _gn(32), nn.ReLU(inplace=True),
            )
            self.stem_half = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), _gn(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1, bias=False), _gn(64), nn.ReLU(inplace=True),
            )
            #  1/2 -> 1/1
            self.fuse_half = FeatureFusion(128, 64, 64)
            self.fuse_full = FeatureFusion(64, 32, 32, upsample=False)
            head_ch = 32
        else:
            head_ch = 128

        self.head = nn.Sequential(
            nn.Conv2d(head_ch, 32, 3, padding=1, bias=False), _gn(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1),
        )

    def _build_convnext_decoder(self, embed_dim, features, n_classes, patch):
        """MIRAGE-protocol decoder: ConvNeXt blocks with 2x upsampling to full res.

        MIRAGE's OCT layer-segmentation benchmark attaches a ConvNeXt decoder to a
        *frozen* encoder. Pair this with ``freeze_encoder=True`` for the run that is
        directly comparable to their published RETFound row.
        """
        self.proj = nn.Sequential(nn.Conv2d(embed_dim, features, 1), _gn(features))
        if patch & (patch - 1):
            raise ValueError(f"the convnext decoder assumes a power-of-two patch size, got {patch}")
        n_up = patch.bit_length() - 1  # log2(patch): 4 stages at p16, 5 at p32
        stages, ch = [], features
        for _ in range(n_up):
            out_ch = max(ch // 2, 32)
            stages.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ch, out_ch, 3, padding=1, bias=False), _gn(out_ch), nn.GELU(),
                ConvNeXtBlock(out_ch),
            ))
            ch = out_ch
        self.up_stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), _gn(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, n_classes, 1),
        )

    def _build_linear_probe(self, embed_dim, features, n_classes):
        """1x1-conv probe on the last block, bilinearly upsampled. The floor."""
        self.proj = nn.Conv2d(embed_dim, features, 1)
        self.head = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1, bias=False), _gn(features), nn.ReLU(inplace=True),
            nn.Conv2d(features, n_classes, 1),
        )

    # -- forward ------------------------------------------------------------

    def train(self, mode=True):
        super().train(mode)
        # A frozen encoder must stay in eval mode, or its dropout/drop-path keep
        # perturbing features that are supposed to be fixed.
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                feats = [f.detach() for f in self.encoder(x)]
        else:
            feats = self.encoder(x)

        # tap_norms sit OUTSIDE the no_grad block: they are trainable decoder
        # parameters, and folding them into the frozen path would leave them stuck at
        # initialisation with no gradient and no error to show for it.
        return [
            ln(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            for f, ln in zip(feats, self.tap_norms)
        ]

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = self._encode(x)

        if self.decoder_kind == "linear_probe":
            y = self.head(self.proj(feats[-1]))
        elif self.decoder_kind == "convnext":
            y = self.head(self.up_stages(self.proj(feats[-1])))
        else:
            f1, f2, f3, f4 = [r(f) for r, f in zip(self.reassemble, feats)]
            f1, f2, f3, f4 = [rn(f) for rn, f in zip(self.rn, (f1, f2, f3, f4))]

            p = self.fuse4(f4)
            p = self.fuse3(p, f3)
            p = self.fuse2(p, f2)
            p = self.fuse1(p, f1)          # 1/2 resolution

            if self.use_stem:
                s_full = self.stem_full(x)
                s_half = self.stem_half(s_full)
                p = self.fuse_half(p, s_half)
                p = self.fuse_full(p, s_full)
            else:
                p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
            y = self.head(p)

        if y.shape[-2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        return y


def param_groups_llrd(model, lr, weight_decay=0.05, layer_decay=0.75, verbose=False):
    """Layer-wise LR decay param groups, following the MAE fine-tuning recipe.

    Block i gets ``lr * layer_decay ** (n_blocks + 1 - (i + 1))``; the decoder and
    head train at full ``lr``, and the deepest encoder block at ``lr * layer_decay``.
    Fine-tuning a 24-block ViT-L on a few thousand slices at a flat LR is the
    standard way to wash out the pretrained features, so this is not optional.
    Norm/bias/token parameters are excluded from weight decay as usual.

    The default 0.75 is MAE's published ViT-L setting, not RETFound's repo default of
    0.65. 0.65 was tuned for the 12-block ViT-B; compounded across 24 blocks it
    reaches ~2e-5 x lr at the patch embedding, which freezes the early layers in all
    but name.

    Every trainable parameter lands in exactly one group; a parameter silently
    dropped from the optimiser is invisible in the loss curve, so the tests assert
    the partition.
    """
    encoder = model.encoder
    n_blocks = encoder.n_blocks
    n_layers = n_blocks + 1  # +1 for the decoder group

    def layer_id(name):
        if name.startswith("encoder."):
            return encoder.layer_id(name[len("encoder."):])
        return n_blocks + 1  # decoder / head / stem / tap norms

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(name)
        no_decay = p.ndim == 1 or name.endswith(".bias") or "token" in name or "pos_embed" in name
        key = f"layer{lid}_{'nodecay' if no_decay else 'decay'}"
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr * (layer_decay ** (n_layers - lid)),
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": key,
            }
        groups[key]["params"].append(p)

    out = list(groups.values())
    if verbose:
        for g in sorted(out, key=lambda g: g["name"]):
            logging.info(f"  {g['name']}: {len(g['params'])} tensors, lr={g['lr']:.3e}, wd={g['weight_decay']}")
    return out


def apply_freeze_strategy(model, strategy: str, partial_from: int = 8):
    """Apply §8.2's three freezing strategies in place, returning a short summary.

    - ``full``   -- everything trainable, with LLRD. Expected primary configuration,
      given the Topcon->Spectralis domain shift and macular-hole pathology rare in
      pretraining.
    - ``partial``-- freeze the patch embed and blocks 0..``partial_from``-1,
      fine-tune the rest plus the decoder. The escape hatch when the per-fold
      train/val Dice gap exceeds ~5 points on 69 training eyes.
    - ``frozen`` -- encoder in eval mode with ``requires_grad=False``, decoder only.
      Also the configuration comparable to MIRAGE's published RETFound row.
    """
    if strategy not in ("full", "partial", "frozen"):
        raise ValueError(f"freeze strategy must be full/partial/frozen, got '{strategy}'")

    if strategy == "full":
        for p in model.encoder.parameters():
            p.requires_grad = True
        model.freeze_encoder = False
    elif strategy == "frozen":
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.freeze_encoder = True
        model.encoder.eval()
    else:
        model.freeze_encoder = False
        for name, p in model.encoder.named_parameters():
            lid = model.encoder.layer_id(name)  # 0 = stem, 1..n = blocks
            p.requires_grad = not (lid == 0 or 1 <= lid <= partial_from)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    summary = f"freeze={strategy}: {n_train / 1e6:.1f}M / {n_total / 1e6:.1f}M parameters trainable"
    logging.info(summary)
    return summary


def _legacy_state_dict_hint(missing, unexpected):
    """Explain the one state-dict mismatch we expect people to hit."""
    if any(k.endswith(("running_mean", "running_var")) for k in unexpected) or any(
        re.match(r"encoder\.blocks\.", k) for k in unexpected
    ):
        return (
            "This checkpoint predates the RETFound module repairs (BatchNorm -> GroupNorm, "
            "encoder.* -> encoder.vit.*). It cannot be loaded; retrain under the current module."
        )
    return None
