"""
Adds nnU-Net (2d and 3d_fullres, Dataset002_ELM3D) prediction-overlay panels
for the same eyes/slices/folds that qualitative_baselines.py picked, so all 8
models can be stitched into one comparison figure per eye.

Run this in the nnunet conda env, with `conda activate nnunet && source
nnUnet.sh` already active (same requirement as nnunet/predict_cv.py), after
qualitative_baselines.py has produced qualitative/images/manifest.json.

Usage:
  python qualitative/qualitative_nnunet.py --manifest qualitative/images/manifest.json
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nnunet.predict_cv import DATASET_NAMES, run_prediction

DATASET_ID = 2  # Dataset002_ELM3D -- both "2d" and "3d_fullres" configs train on this


def load_slice(path: Path, slice_idx: int) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))[slice_idx]


def overlay_contour(gray_uint8: np.ndarray, mask01: np.ndarray, bgr, thickness=2) -> np.ndarray:
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    mask255 = (mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(base, contours, -1, bgr, thickness)
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="qualitative/images/manifest.json")
    ap.add_argument("--out_dir", default=None, help="Default: same directory as --manifest")
    ap.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    ap.add_argument("--trainer", default="nnUNetTrainer")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--step_size", type=float, default=0.5)
    ap.add_argument("--disable_tta", action="store_true")
    args = ap.parse_args()

    for env_var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if env_var not in os.environ:
            raise RuntimeError(f"{env_var} is not set -- did you `source nnUnet.sh`?")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    out_dir = Path(args.out_dir) if args.out_dir else manifest_path.parent

    dataset_name = DATASET_NAMES[DATASET_ID]
    raw_root = Path(os.environ["nnUNet_raw"]) / dataset_name
    images_tr = raw_root / "imagesTr"
    file_ending = ".nii.gz"

    for configuration in ["2d", "3d_fullres"]:
        model_root = Path(os.environ["nnUNet_results"]) / dataset_name / f"{args.trainer}__{args.plans}__{configuration}"

        for entry in manifest["eyes"]:
            tag, eye_id, slice_idx, fold = entry["tag"], entry["eye_id"], entry["slice_idx"], entry["fold"]
            ckpt_path = model_root / f"fold_{fold}" / args.checkpoint
            if not ckpt_path.exists():
                print(f"[skip] {configuration}: missing checkpoint for fold {fold}: {ckpt_path}")
                continue

            tmp_in = Path(tempfile.mkdtemp(prefix="qual_nnunet_in_"))
            tmp_out = Path(tempfile.mkdtemp(prefix="qual_nnunet_out_"))
            try:
                run_prediction(
                    images_tr, [eye_id], file_ending, DATASET_ID, configuration,
                    args.plans, args.trainer, fold, args.checkpoint, args.step_size,
                    args.disable_tta, tmp_in, tmp_out,
                )
                pred01 = load_slice(tmp_out / f"{eye_id}{file_ending}", slice_idx) > 0
            finally:
                shutil.rmtree(tmp_in, ignore_errors=True)
                shutil.rmtree(tmp_out, ignore_errors=True)

            img_slice = load_slice(images_tr / f"{eye_id}_0000{file_ending}", slice_idx)
            gray = np.clip(img_slice, 0, 255).astype(np.uint8)

            model_tag = f"nnUNet_{configuration}"
            out_path = out_dir / f"{tag}_{eye_id}_{model_tag}.png"
            cv2.imwrite(str(out_path), overlay_contour(gray, pred01, bgr=(0, 255, 0)))
            print(f"[{tag}] eye={eye_id} {model_tag} -> {out_path}")

    print(f"\n[Saved] -> {out_dir}")


if __name__ == "__main__":
    main()
