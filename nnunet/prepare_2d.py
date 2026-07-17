#!/usr/bin/env python3
"""
Convert data_no_anomalies/ (individual 2D OCT slices) into an nnU-Net v2 raw dataset (Dataset001_ELM2D), one training case per slice.

All 99 patients / 4851 slices go into imagesTr/labelsTr (not split into Tr/Ts here) because this repo's train/val/test assignment rotates per fold (metadata.csv split_fold0..split_fold4) rather than being fixed.
The actual 5-fold train/val split is written separately by make_splits.py as nnU-Net's splits_final.json; slices held out as "test" for a given fold simply aren't referenced by that fold's split and get evaluated later with nnUNetv2_predict + this repo's own metrics.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATASET_NAME = "Dataset001_ELM2D"
N_SLICES = 49

# converts image from 0..255 to {0,1}
def convert_mask(src: Path, dst: Path) -> None:
    arr = np.array(Image.open(src))
    arr = (arr > 0).astype(np.uint8)
    Image.fromarray(arr).save(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_no_anomalies")
    parser.add_argument("--nnunet-raw", default="nnunet/nnUNet_raw")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
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
    meta["patient_id"] = meta["patient_id"].str.strip().str.zfill(3)
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

    n_train = 0
    for pid in tqdm(patient_ids, desc="patients", unit="patient"):
        for slice_idx in range(N_SLICES):
            img_src = image_dir / f"{pid}-{slice_idx}.png"
            mask_src = mask_dir / f"{pid}-{slice_idx}.png"
            if not img_src.exists() or not mask_src.exists():
                raise FileNotFoundError(
                    f"Missing slice for patient {pid}, index {slice_idx}"
                )
            # if slice exists:
            case_id = f"{pid}-{slice_idx}"
            shutil.copy(img_src, images_tr / f"{case_id}_0000.png") # each channel is a separate file, so we add _0000 for the first and only channel (OCT)
            convert_mask(mask_src, labels_tr / f"{case_id}.png")
            n_train += 1

    dataset_json = {
        "channel_names": {"0": "OCT"},
        "labels": {"background": 0, "ELM": 1},
        "numTraining": n_train,
        "file_ending": ".png",
    }
    with open(out_root / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Wrote {n_train} training cases ({len(patient_ids)} patients) to {out_root}")


if __name__ == "__main__":
    main()
