"""Native-scale input sizing for the foundation-model arms (plan §3.3).

Why not just resize to a square
-------------------------------
The working dataset has 44 distinct native slice sizes: heights 204-490, widths
321-776, with aspect ratios (H/W) from 1.17 down to 0.295 -- a 4x spread. Half the
eyes share exactly 376x321 (portrait); the rest are landscape and mostly unique.
Squaring everything stretches the 376x321 cohort horizontally while squashing a
250x730 scan by nearly 3x in width, so the ELM's slope, apparent thickness and
continuity all become artefacts of which cohort an eye belongs to. On a 1-2 px
structure that confound can easily dominate any architectural effect.

The policy implemented here
---------------------------
- **Training**: augment at native scale, then take a random 384x384 crop, with
  foreground oversampling so most crops contain the membrane. Images smaller than
  the crop are padded, never upscaled, and the padding is masked out of the loss.
  384x384 is nnU-Net's own fingerprint-derived patch size for this dataset, adopted
  rather than invented -- see ``NNUNET_FINGERPRINT_CROP`` below for why a square
  crop, and why the obvious landscape choice is actively harmful here.
- **Inference**: pad each image to *its own* next multiple of the patch size, run
  with dynamic position embeddings, crop the logits back to true size. A single
  global 490x776 canvas would waste 69% of its area on the dominant 376x321 cohort
  and pay ViT-L attention over 1519 tokens to do it.
- **MedSAM**: canonical SAM preprocessing (resize longest side to 1024, pad to
  square) as the primary configuration, with native-scale-pad available as an
  ablation. Upscaling genuinely thickens the ELM in pixels, which may help rather
  than hurt -- worth measuring rather than assuming.

The ground-truth mask is never resized on the native paths: it is cropped and
padded only. The one exception is the SAM canonical path, where the fixed 1024
canvas makes rescaling unavoidable; there the *training target* is nearest-neighbour
resampled and every reported metric is computed against the on-disk native mask
instead (``predict_cv2d.py --native_res``, which those arms require).

Every transform here is duck-type compatible with the albumentations pipelines in
``elm/dataset.py``: called as ``t(image=..., mask=...)``, returning a dict with
``image`` (CHW float tensor), ``mask`` (HW tensor) and ``valid`` (HW uint8, 1 where
the pixel is real rather than padding).
"""

import cv2
import numpy as np
import torch

import albumentations as A

# ImageNet statistics, matching every other 2D arm in the benchmark.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Training crop, taken from nnU-Net's dataset fingerprint for this dataset.
#
# nnU-Net's experiment planner, run on Dataset001_ELM2D, derived
# ``patch_size=[384, 384]`` from a median image shape of 376x321. Adopting its
# answer rather than inventing one is the same move plan §3.4 already makes for
# spacing: a published, citable planner (Isensee et al., Nature Methods 2021)
# fixes the choice, and every arm then trains on the geometry nnU-Net gets, so
# nnU-Net cannot win on preprocessing.
#
# Why square and why 384, concretely. The plan's original 320x512 assumed a
# landscape B-scan, but the minimum width in this dataset is 321, so a 512-wide
# crop *pads* 54.5% of slices instead of cropping them -- and pads them
# unequally: 62.7% real pixels for the 376x321 cohort against 87.4% for the wide
# one. That is a 24.7-point cohort asymmetry in the input itself, along exactly
# the split §9.3 reports as a result. 384x384 pads more in total (81.2% real) but
# pads both cohorts almost equally -- 81.9% vs 80.5%, a 1.3-point gap -- which is
# the property that actually removes the confound.
#
# 384 is a multiple of both 16 and 32, so the patch-16 arms (RETFound, MedSAM) and
# the patch-32 arm (MIRAGE) share one crop. MIRAGE's token grid is then 12x12=144
# against its pretrained 256; that is a real cost, accepted because MIRAGE is a
# Claim B arm and cohort balance matters more than token count.
NNUNET_FINGERPRINT_CROP = (384, 384)

# Square crops whose *cohort* padding gap stays negligible, for the §3.5
# resolution ablation: 0.0 / 1.8 / 1.3 points at 192 / 256 / 384. The absolute
# padding fraction necessarily rises with crop size (100% / 99.1% / 81.2% real),
# but the asymmetry between cohorts does not, so the ablation measures resolution
# rather than differential padding. 448 and 512 are excluded for that reason:
# their gaps are 12.6 and 18.7 points.
RESOLUTION_LADDER = ((192, 192), (256, 256), (384, 384))


def next_multiple(n: int, divisor: int) -> int:
    return ((int(n) + divisor - 1) // divisor) * divisor


def default_crop(patch_size: int):
    """Training crop for an encoder with the given patch size, as a (h, w) tuple.

    One crop for every encoder. 384 divides by 16 and by 32, so there is no reason
    to vary it per architecture -- and varying it would reintroduce exactly the
    confound the fixed geometry exists to remove, since Claim A holds everything
    but the encoder weights constant.
    """
    h, w = NNUNET_FINGERPRINT_CROP
    if h % patch_size or w % patch_size:
        return (next_multiple(h, patch_size), next_multiple(w, patch_size))
    return (h, w)


def pad_to(img, mask, out_h, out_w, valid=None):
    """Bottom/right zero-pad to (out_h, out_w). Returns (img, mask, valid).

    Bottom/right rather than centred so that a padded prediction can be cropped back
    with a plain ``[:h, :w]`` slice, at every stage of the pipeline.
    """
    h, w = img.shape[:2]
    if valid is None:
        valid = np.ones((h, w), dtype=np.uint8)
    ph, pw = max(out_h - h, 0), max(out_w - w, 0)
    if ph == 0 and pw == 0:
        return img, mask, valid
    pad2 = ((0, ph), (0, pw))
    img = np.pad(img, pad2 + ((0, 0),) * (img.ndim - 2), mode="constant")
    mask = np.pad(mask, pad2, mode="constant")
    valid = np.pad(valid, pad2, mode="constant")
    return img, mask, valid


def pad_to_multiple(img, mask=None, divisor=16):
    """Pad an HWC image (and optional mask) to the next multiple of ``divisor``."""
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), dtype=np.uint8)
    return pad_to(img, mask, next_multiple(h, divisor), next_multiple(w, divisor))


def _to_chw_tensor(img):
    if img.ndim == 2:
        img = img[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def _normalize(img, mode="imagenet"):
    """HWC uint8 -> HWC float32.

    ``imagenet`` matches every other 2D arm in the benchmark. ``unit`` leaves values
    in [0, 1], which is what MIRAGE's B-scan adapter expects -- feeding it
    ImageNet-standardised input is invisible in the shapes and merely degrades the
    features, so the encoder declares its own requirement and the transforms honour it.
    """
    x = img.astype(np.float32) / 255.0
    if mode == "unit":
        return x
    if mode != "imagenet":
        raise ValueError(f"unknown normalisation '{mode}'")
    mean = np.array(IMAGENET_MEAN[: x.shape[2]], dtype=np.float32)
    std = np.array(IMAGENET_STD[: x.shape[2]], dtype=np.float32)
    return (x - mean) / std


def native_augmentations():
    """The benchmark's augmentation set, applied at native scale before cropping.

    Same operations and probabilities as ``elm.dataset.make_2d_transforms`` so the
    native-sizing arms differ from the 256x256 arms in sizing alone.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            scale=(0.90, 1.10),
            rotate=(-20, 20),
            p=0.7,
            interpolation=0,
            mask_interpolation=0,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.03), mean_range=(0.0, 0.0), per_channel=False, p=0.3),
    ], p=1.0)


class NativeCropTransform:
    """Training transform: augment at native scale, then a foreground-biased crop.

    ``crop_size`` must be a multiple of the encoder patch size on both axes -- it is
    the fixed shape every training batch is stacked into, which is what lets the
    dataloader batch a set of images with 44 different native sizes at all.
    """

    def __init__(self, crop_size=NNUNET_FINGERPRINT_CROP, divisor=16, fg_prob=0.75, augment=True,
                 rng=None,
                 normalize="imagenet"):
        self.crop_h, self.crop_w = int(crop_size[0]), int(crop_size[1])
        for axis, n in (("height", self.crop_h), ("width", self.crop_w)):
            if n % divisor:
                raise ValueError(f"crop {axis} {n} is not a multiple of the patch size {divisor}")
        self.divisor = divisor
        self.fg_prob = fg_prob
        self.aug = native_augmentations() if augment else None
        self.rng = rng or np.random
        self.normalize = normalize

    def _crop_origin(self, mask):
        """Top-left corner of the crop window, biased towards foreground."""
        h, w = mask.shape[:2]
        max_y, max_x = max(h - self.crop_h, 0), max(w - self.crop_w, 0)
        ys, xs = np.nonzero(mask)
        if ys.size and self.rng.random() < self.fg_prob:
            # Centre the window on a random positive pixel, then clamp into range.
            i = self.rng.randint(ys.size)
            y = int(np.clip(ys[i] - self.crop_h // 2, 0, max_y))
            x = int(np.clip(xs[i] - self.crop_w // 2, 0, max_x))
            return y, x
        return (self.rng.randint(max_y + 1), self.rng.randint(max_x + 1))

    def __call__(self, image, mask):
        if self.aug is not None:
            out = self.aug(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        # Pad first (never upscale: many heights are 204-300, below the crop height),
        # so the crop window always fits.
        image, mask, valid = pad_to(
            image, mask, max(image.shape[0], self.crop_h), max(image.shape[1], self.crop_w)
        )
        y, x = self._crop_origin(mask)
        sl = (slice(y, y + self.crop_h), slice(x, x + self.crop_w))
        image, mask, valid = image[sl], mask[sl], valid[sl]

        return {
            "image": _to_chw_tensor(_normalize(image, self.normalize)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
        }


class NativePadTransform:
    """Inference transform: pad each image to its own next patch multiple.

    Batches are ragged by construction, so loaders using this must run at batch
    size 1 (``predict_cv2d.py`` enforces that). ``orig_size`` travels with the
    sample so the caller can crop logits back with ``[:h, :w]``.
    """

    def __init__(self, divisor=16, normalize="imagenet"):
        self.divisor = divisor
        self.normalize = normalize

    def __call__(self, image, mask):
        h, w = image.shape[:2]
        image, mask, valid = pad_to_multiple(image, mask, self.divisor)
        return {
            "image": _to_chw_tensor(_normalize(image, self.normalize)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "orig_size": (h, w),
        }


class SamResizePadTransform:
    """Canonical SAM preprocessing: resize longest side to ``size``, pad to square.

    MedSAM's encoder was pretrained under exactly this transform, so it is the
    primary configuration for that arm; ``NativePadTransform`` with divisor 16 is
    the ablation. Upscaling thickens the ELM in pixels, which may help on a 1-2 px
    structure -- that is the point of running both.

    The *mask* is nearest-neighbour resampled here, the one place in the pipeline
    where that happens, because a fixed 1024 canvas leaves no alternative for the
    training target. Score these arms with ``predict_cv2d.py --native_res``, which
    reloads the on-disk mask and never sees the resampled one.
    """

    def __init__(self, size=1024, augment=False, normalize="imagenet"):
        self.size = int(size)
        self.aug = native_augmentations() if augment else None
        self.normalize = normalize

    def __call__(self, image, mask):
        if self.aug is not None:
            out = self.aug(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        h, w = image.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        image, mask, valid = pad_to(image, mask, self.size, self.size)

        return {
            "image": _to_chw_tensor(_normalize(image, self.normalize)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "orig_size": (h, w),
            "resized_size": (nh, nw),
        }


def make_sizing_transform(policy, train, patch_size=16, crop_size=None, sam_size=1024,
                          normalize="imagenet"):
    """Build the transform for a sizing policy.

    ``policy``:
      - ``square``      -- the benchmark's fixed 256x256 resize (see
        ``elm.dataset.make_2d_transforms``); not built here, callers keep using that.
      - ``native``      -- §3.3: native-scale crops for training, per-image padding
        for inference.
      - ``sam``         -- canonical SAM resize-and-pad, both phases.
    """
    if policy == "native":
        if train:
            return NativeCropTransform(
                crop_size=crop_size or default_crop(patch_size), divisor=patch_size,
                normalize=normalize,
            )
        return NativePadTransform(divisor=patch_size, normalize=normalize)
    if policy == "sam":
        return SamResizePadTransform(size=sam_size, augment=train, normalize=normalize)
    raise ValueError(f"unknown sizing policy '{policy}' (expected 'native' or 'sam')")


def crop_logits_to(logits, orig_size):
    """Undo bottom/right padding on a [B, C, H, W] logit tensor."""
    h, w = int(orig_size[0]), int(orig_size[1])
    return logits[..., :h, :w]
