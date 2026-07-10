#!/usr/bin/env python3
"""
Generates nnU-Net's splits_final.json for Dataset001_ELM2D and
Dataset002_ELM3D from this repo's own metadata.csv split_fold0..4 columns,
so nnU-Net's 5 folds match the folds used to benchmark the other models
(see make_cv_splits.py).

nnU-Net's CV only knows train/val -- there's no per-fold "test" slot in
splits_final.json. Since data_no_anomalies/metadata.csv rotates a distinct
20% test set per fold, patients marked "test" for fold k are simply left
out of fold k's train+val lists (they still exist in imagesTr/labelsTr,
just unused for that fold). This script also writes
test_cases_per_fold.json next to the dataset so those held-out cases are
easy to find later for nnUNetv2_predict + evaluation.

Run this after nnunet/prepare_2d.py and nnunet/prepare_3d.py. It creates
nnUNet_preprocessed/<dataset>/ if it doesn't exist yet, so it's fine to run
either before or after nnUNetv2_plan_and_preprocess.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

N_SLICES = 49


def build_splits(meta: pd.DataFrame, mode: str):
    folds = []
    test_per_fold = []
    for k in range(5):
        col = f"split_fold{k}"
        train_ids = meta.loc[meta[col] == "train", "patient_id"].tolist()
        val_ids = meta.loc[meta[col] == "val", "patient_id"].tolist()
        test_ids = meta.loc[meta[col] == "test", "patient_id"].tolist()

        if mode == "2d":
            to_cases = lambda ids: sorted(f"{pid}-{s}" for pid in ids for s in range(N_SLICES))
        else:
            to_cases = lambda ids: sorted(ids)

        folds.append({"train": to_cases(train_ids), "val": to_cases(val_ids)})
        test_per_fold.append({"fold": k, "test_cases": to_cases(test_ids)})
    return folds, test_per_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_no_anomalies")
    parser.add_argument("--nnunet-preprocessed", default="nnUNet_preprocessed")
    parser.add_argument("--dataset-2d", default="Dataset001_ELM2D")
    parser.add_argument("--dataset-3d", default="Dataset002_ELM3D")
    args = parser.parse_args()

    meta = pd.read_csv(Path(args.data_root) / "metadata.csv", dtype={"patient_id": str})
    meta["patient_id"] = meta["patient_id"].str.strip().str.zfill(3)

    for dataset_name, mode in [(args.dataset_2d, "2d"), (args.dataset_3d, "3d")]:
        splits, test_per_fold = build_splits(meta, mode)
        out_dir = Path(args.nnunet_preprocessed) / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "splits_final.json", "w") as f:
            json.dump(splits, f, indent=2)
        with open(out_dir / "test_cases_per_fold.json", "w") as f:
            json.dump(test_per_fold, f, indent=2)

        summary = ", ".join(
            f"fold{i}: train={len(s['train'])}/val={len(s['val'])}/test={len(t['test_cases'])}"
            for i, (s, t) in enumerate(zip(splits, test_per_fold))
        )
        print(f"{dataset_name}: {summary}")
        print(f"  -> {out_dir / 'splits_final.json'}")
        print(f"  -> {out_dir / 'test_cases_per_fold.json'}")


if __name__ == "__main__":
    main()
