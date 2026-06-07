#!/usr/bin/env python3
"""
Generate 5-fold train/val/test splits with 70/10/20 proportions.

- Test is rotated across folds (each fold gets its own 20% test set).
- Validation is sampled from the remaining 80% to be ~10% of the full dataset.
- Output columns: split_fold0 .. split_fold4 in metadata.csv
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 5-fold 70/10/20 splits in metadata.csv"
    )
    parser.add_argument("--metadata", type=str, default="data_no_anomalies/metadata.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    df = pd.read_csv(meta_path, dtype={"patient_id": str})

    if "patient_id" not in df.columns:
        raise ValueError("metadata.csv must have a 'patient_id' column")

    df["patient_id"] = df["patient_id"].astype(str).str.strip().str.zfill(3)
    base = df.drop_duplicates("patient_id").copy()
    patients = sorted(base["patient_id"].unique().tolist())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(patients)

    n = len(patients)
    n_test = int(round(n * 0.20))
    n_val = int(round(n * 0.10))

    # Split into 5 roughly equal chunks for rotating test sets.
    chunks = np.array_split(patients, 5)

    split_cols = {}
    for fold in range(5):
        test_ids = set(chunks[fold])
        remaining = [pid for pid in patients if pid not in test_ids]

        rng_fold = np.random.default_rng(args.seed + fold)
        rng_fold.shuffle(remaining)

        val_ids = set(remaining[:n_val])
        train_ids = set(remaining[n_val:])

        col = f"split_fold{fold}"
        split_cols[col] = []
        for pid in patients:
            if pid in test_ids:
                split_cols[col].append("test")
            elif pid in val_ids:
                split_cols[col].append("val")
            else:
                split_cols[col].append("train")

    out = base.set_index("patient_id").reindex(patients).reset_index()
    for col, values in split_cols.items():
        out[col] = values

    out.to_csv(meta_path, index=False)
    print(f"Wrote splits to {meta_path} with columns: {', '.join(split_cols.keys())}")


if __name__ == "__main__":
    main()
