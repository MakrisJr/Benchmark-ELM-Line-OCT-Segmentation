#!/usr/bin/env python3
"""
Convert data_no_anomalies/ into an nnU-Net v2 raw dataset (Dataset002_ELM3D)
by stacking each patient's 49 slices (index 0..48, in order) into a single
3D volume, matching the assembly done by elm.dataset.D3Dataset.

All 99 patients go into imagesTr/labelsTr -- see prepare_2d.py's docstring
for why there's no fixed imagesTs split here.

NOTE ON SPACING: default --spacing below is 0.00547 x 0.00387 x 0.029 mm
(5.47um x 3.87um x 29um in x/width, y/height, z/slice order), the real
acquisition geometry for this OCT dataset. This is fairly anisotropic
(z spacing ~5-7x the in-plane spacing), which nnU-Net's planner will pick
up on and likely handle with anisotropic resampling for 3d_fullres.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

DATASET_NAME = "Dataset002_ELM3D"
N_SLICES = 49

# load all 49 slices for a given patient into a single 3D volume (numpy array)
# returns (49, height, width) array for both image and mask
def load_volume(image_dir: Path, mask_dir: Path, pid: str):
    imgs, masks = [], []
    for slice_idx in range(N_SLICES):
        img_path = image_dir / f"{pid}-{slice_idx}.png"
        mask_path = mask_dir / f"{pid}-{slice_idx}.png"
        if not img_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing slice for patient {pid}, index {slice_idx}")
        imgs.append(np.array(Image.open(img_path).convert("L")))
        masks.append((np.array(Image.open(mask_path)) > 0).astype(np.uint8))
    return np.stack(imgs), np.stack(masks)

# write a 3D volume (numpy array) to a NIfTI file with the given spacing
def write_nifti(arr: np.ndarray, path: Path, spacing, is_label: bool) -> None:
    dtype = np.uint8 if is_label else np.float32
    img = sitk.GetImageFromArray(arr.astype(dtype))
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_no_anomalies")
    parser.add_argument("--nnunet-raw", default="nnunet/nnUNet_raw")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(0.00547, 0.00387, 0.029),
        metavar=("X", "Y", "Z"),
        help="Voxel spacing in mm, SimpleITK (x/width, y/height, z/slice) order. "
        "Default is this dataset's real acquisition geometry "
        "(5.47um x 3.87um x 29um).",
    )
    parser.add_argument(
        "--limit-patients",
        type=int,
        default=None,
        help="Only convert the first N patients (for a quick smoke test).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    image_dir = data_root / "all" / "image"
    mask_dir = data_root / "all" / "mask"

    meta = pd.read_csv(data_root / "metadata.csv", dtype={"patient_id": str})
    meta["patient_id"] = meta["patient_id"].str.strip().str.zfill(3) # pad to 3 digits
    patient_ids = sorted(meta["patient_id"].unique())
    if args.limit_patients:
        patient_ids = patient_ids[: args.limit_patients]

    out_root = Path(args.nnunet_raw) / args.dataset_name
    images_tr = out_root / "imagesTr"
    labels_tr = out_root / "labelsTr"
    for d in (images_tr, labels_tr):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    spacing = tuple(args.spacing)
    n_train = 0
    for pid in tqdm(patient_ids, desc="patients", unit="patient"):
        img_vol, mask_vol = load_volume(image_dir, mask_dir, pid)
        write_nifti(img_vol, images_tr / f"{pid}_0000.nii.gz", spacing, is_label=False)
        write_nifti(mask_vol, labels_tr / f"{pid}.nii.gz", spacing, is_label=True)
        n_train += 1

    dataset_json = {
        "channel_names": {"0": "OCT"},
        "labels": {"background": 0, "ELM": 1},
        "numTraining": n_train,
        "file_ending": ".nii.gz",
    }
    with open(out_root / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Wrote {n_train} training volumes to {out_root} (spacing={spacing})")


if __name__ == "__main__":
    main()
