"""
Qualitative comparison figures for the 6 baseline architectures: for several
representative eyes per category (worst/median/best nnU-Net Dice, N each via
--n_per_category), saves the native-resolution raw image, an
image+ground-truth overlay, and an image+prediction overlay per model, all at
native resolution.

The displayed slice is always the central slice (index N_SLICES // 2) of each
eye's volume, not the slice with the most annotated area -- picking by area
would systematically avoid macular-hole cases, where the ELM line is
disrupted (and often has *less* annotated area) near the volume's center,
which is exactly the hard case worth showing.

Each baseline model is evaluated with the exact fold checkpoint that treated
the eye as held-out test data during 5-fold CV, per data_no_anomalies/metadata.csv's
split_fold{k} columns (shared across all baselines, since they train on the
same splits).

Writes qualitative/images/manifest.json (eye_id, slice_idx, fold per selected
eye) so qualitative_nnunet.py can add the two nnU-Net panels without needing
to re-derive the fold/slice choice, then make_qualitative_grid.py stitches
everything into one comparison figure per eye.

Usage (run in the elm-gpu conda env, from the repo root):
  python qualitative/qualitative_baselines.py --n_per_category 2
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from elm.dataset import D3Dataset, make_2d_transforms
from predict_cv2d import build_model as build_model_2d, find_fold_checkpoint as find_fold_checkpoint_2d
from predict_cv3d import (
    build_model as build_model_3d,
    find_fold_checkpoint as find_fold_checkpoint_3d,
    match_depth,
    upsample_pred_volume,
)

BASE = REPO_ROOT
DATA_ROOT = BASE / "data_no_anomalies"
IMAGE_DIR = DATA_ROOT / "all" / "image"
MASK_DIR = DATA_ROOT / "all" / "mask"
METADATA_PATH = DATA_ROOT / "metadata.csv"

N_SLICES = 49

# Same model_root assumptions as native_eval.sbatch -- double check these are
# the checkpoints you consider canonical before trusting the qualitative
# panels (see native_eval.sbatch's comments for the ambiguous ones: SegNet
# and UNet2p5D_SlidingWindow).
MODELS_2D = {
    "SegNet": "elm-results/SegNet_Jun-04-2026_1243_model",
    "R2U_Net": "elm-results/R2U_Net_Jun-04-2026_0441_model",
}
MODELS_3D = {
    "UNet2DEnc3DDec": "elm-results/UNet2DEnc3DDec_Jun-07-2026_1946_model",
    "UNet3DFrawley": "elm-results/UNet3DFrawley_Jun-09-2026_1509_model",
    "CSAM_UNet2p5D": "elm-results/CSAM_UNet2p5D_Jun-08-2026_0152_model",
    "UNet2p5D_SlidingWindow": "elm-results/UNet2p5D_SlidingWindow_Jun-11-2026_0120_model",
}

NNUNET_2D_EYE_DICE_CSV = (
    BASE / "nnUNet_results/Dataset002_ELM3D/nnUNetTrainer__nnUNetResEncUNetMPlans__2d"
    "/cv_eval/cv_per_patient_all_folds.csv"
)


def pick_representative_eyes(n_per_category=2):
    """n worst/median/best eyes by nnU-Net (2d config) per-eye Dice -- already
    computed and native-resolution end to end, so it's a stable reference
    for picking a spread of easy/typical/hard cases shared across models.

    Returns a list of (tag, eye_id, dice) tuples, tags like "worst1",
    "worst2", "median1", ..., "best1", ... (ordered worst -> best within
    each category).
    """
    rows = list(csv.DictReader(open(NNUNET_2D_EYE_DICE_CSV)))
    rows.sort(key=lambda r: float(r["vol_dice_pooled"]))
    n = len(rows)
    k = min(n_per_category, n // 3 if n >= 3 else n)

    worst_rows = rows[:k]
    best_rows = rows[-k:] if k else []
    center = n // 2
    start = max(0, center - k // 2)
    median_rows = rows[start:start + k]

    picked = []
    for category, cat_rows in [("worst", worst_rows), ("median", median_rows), ("best", best_rows)]:
        for i, r in enumerate(cat_rows, start=1):
            tag = f"{category}{i}" if len(cat_rows) > 1 else category
            picked.append((tag, r["eye_id"], float(r["vol_dice_pooled"])))
    return picked


def fold_for_eye(metadata: pd.DataFrame, eye_id: str) -> int:
    row = metadata[metadata["patient_id"] == eye_id]
    if row.empty:
        raise ValueError(f"eye_id {eye_id} not found in {METADATA_PATH}")
    row = row.iloc[0]
    for fold in range(5):
        if row.get(f"split_fold{fold}") == "test":
            return fold
    raise ValueError(f"No split_fold column marks eye_id {eye_id} as 'test'")


def central_slice_for_eye(eye_id: str) -> int:
    """The central slice (index N_SLICES // 2), not the slice with the most
    annotated area -- the central slices are where macular holes / ELM
    disruptions actually show up, so picking by max area would systematically
    avoid the hardest cases."""
    center = N_SLICES // 2
    if (MASK_DIR / f"{eye_id}-{center}.png").exists():
        return center
    for offset in range(1, N_SLICES // 2 + 1):
        for z in (center - offset, center + offset):
            if 0 <= z < N_SLICES and (MASK_DIR / f"{eye_id}-{z}.png").exists():
                return z
    raise FileNotFoundError(f"No mask slices found for eye_id {eye_id} near the central slice")


def load_native_gray(eye_id: str, slice_idx: int) -> np.ndarray:
    return np.array(Image.open(IMAGE_DIR / f"{eye_id}-{slice_idx}.png").convert("L"))


def load_native_mask01(eye_id: str, slice_idx: int) -> np.ndarray:
    return (np.array(Image.open(MASK_DIR / f"{eye_id}-{slice_idx}.png")) > 0).astype(np.uint8)


def overlay_contour(gray_uint8: np.ndarray, mask01: np.ndarray, bgr, thickness=2) -> np.ndarray:
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    mask255 = (mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(base, contours, -1, bgr, thickness)
    return base


def predict_2d(model_name: str, model_root: str, eye_id: str, slice_idx: int, fold: int, device) -> np.ndarray:
    ckpt = find_fold_checkpoint_2d(Path(model_root), fold)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint for {model_name} fold {fold} under {model_root}")

    net = build_model_2d(model_name)
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    net.to(device)
    net.eval()

    img_pil = Image.open(IMAGE_DIR / f"{eye_id}-{slice_idx}.png").convert("RGB")
    img_np = np.array(img_pil)
    mask_np = load_native_mask01(eye_id, slice_idx)

    transform = make_2d_transforms(train=False, out_size=(256, 256))
    out = transform(image=img_np, mask=mask_np)
    img_t = out["image"].unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        logits = net(img_t)
        pred256 = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    native_hw = mask_np.shape
    return cv2.resize(pred256, (native_hw[1], native_hw[0]), interpolation=cv2.INTER_NEAREST)


def predict_3d(model_name: str, model_root: str, eye_id: str, slice_idx: int, fold: int, device, window_k=7) -> np.ndarray:
    ckpt = find_fold_checkpoint_3d(Path(model_root), fold)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint for {model_name} fold {fold} under {model_root}")

    net = build_model_3d(model_name, window_k=window_k)
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    net.to(device)
    net.eval()

    ds = D3Dataset(root_dir=str(DATA_ROOT), split="test", fold=fold, transform=False)
    idx = ds.eye_ids.index(eye_id)
    sample = ds[idx]
    img = sample["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
    gt = sample["mask"].unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        logits = net(img)
        pred = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
        pred = match_depth(pred, gt)
        pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)

    native_gray = load_native_gray(eye_id, slice_idx)
    pred_vol_native = upsample_pred_volume(pred_np, native_gray.shape)
    return pred_vol_native[slice_idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=str, default="qualitative/images")
    ap.add_argument("--n_per_category", type=int, default=2,
                     help="How many worst/median/best eyes to include (default: 2 each, 6 total)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(METADATA_PATH, dtype={"patient_id": str})
    metadata["patient_id"] = metadata["patient_id"].astype(str).str.strip().str.zfill(3)

    eyes = pick_representative_eyes(n_per_category=args.n_per_category)
    manifest = {"eyes": []}

    for tag, eye_id, nnunet_2d_dice in eyes:
        fold = fold_for_eye(metadata, eye_id)
        slice_idx = central_slice_for_eye(eye_id)
        print(f"[{tag}] eye={eye_id} fold={fold} slice={slice_idx}/{N_SLICES - 1} nnunet_2d_dice={nnunet_2d_dice:.4f}")

        gray = load_native_gray(eye_id, slice_idx)
        gt01 = load_native_mask01(eye_id, slice_idx)

        cv2.imwrite(str(out_dir / f"{tag}_{eye_id}_image.png"), gray)
        cv2.imwrite(str(out_dir / f"{tag}_{eye_id}_gt.png"), overlay_contour(gray, gt01, bgr=(0, 255, 255)))

        for model_name, model_root in MODELS_2D.items():
            pred01 = predict_2d(model_name, model_root, eye_id, slice_idx, fold, device)
            cv2.imwrite(str(out_dir / f"{tag}_{eye_id}_{model_name}.png"), overlay_contour(gray, pred01, bgr=(0, 255, 0)))
            print(f"    saved {model_name}")

        for model_name, model_root in MODELS_3D.items():
            pred01 = predict_3d(model_name, model_root, eye_id, slice_idx, fold, device)
            cv2.imwrite(str(out_dir / f"{tag}_{eye_id}_{model_name}.png"), overlay_contour(gray, pred01, bgr=(0, 255, 0)))
            print(f"    saved {model_name}")

        manifest["eyes"].append({
            "tag": tag, "eye_id": eye_id, "slice_idx": slice_idx, "fold": fold,
            "nnunet_2d_dice": nnunet_2d_dice,
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[Saved] -> {out_dir} (+ {manifest_path} for qualitative_nnunet.py)")


if __name__ == "__main__":
    main()
