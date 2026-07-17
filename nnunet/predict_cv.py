#!/usr/bin/env python3
"""
Predict + score nnU-Net's 5-fold CV test splits.

Runs, for one trained (dataset, configuration) combo, one nnUNetv2_predict
call per fold -- using only that fold's model on that fold's held-out test
cases (nnunet/nnUNet_preprocessed/<dataset>/test_cases_per_fold.json) -- then scores
the predictions against labelsTr with the same metric set and per-eye
aggregation as predict_cv2d.py / predict_cv3d.py, so nnU-Net's numbers plug
directly into this repo's existing comparison tooling (compare_csvs.py,
paired_*_summary.csv), no adjustment needed.

Dataset001_ELM2D cases are per-slice ("<pid>-<slice>", one PNG each) --
scored like predict_cv2d.py: per-slice metrics, then pooled per-eye
aggregation across that eye's slices.
Dataset002_ELM3D cases are per-eye volumes ("<pid>", one NIfTI each,
whichever of "2d"/"3d_fullres" was trained) -- scored like predict_cv3d.py:
metrics computed directly per volume.

Requires nnunet/submit_folds.sh to have finished (checkpoint_final.pth
present for every fold) and `conda activate nnunet && source nnUnet.sh` to
already be active in this shell, same as nnunet/train.sbatch.

Usage:
  python nnunet/predict_cv.py --dataset_id 1 --configuration 2d
  python nnunet/predict_cv.py --dataset_id 2 --configuration 2d
  python nnunet/predict_cv.py --dataset_id 2 --configuration 3d_fullres
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from elm.metrics import (
    confusion_counts,
    dice_iou_sen_fpr,
    rmse,
    boundary_f1_2d,
    surface_dice_2d,
    assd_hd_hd95_2d,
    boundary_f1_3d,
    surface_dice_3d,
    assd_hd_hd95_3d,
    summarize_list,
    summarize_rows,
)
from elm.hole_metrics import (
    analyze_slice,
    analyze_slice_3d,
    gap_result_to_row_fields,
    summarize_gap_geometry,
    summarize_spurious_gaps,
)

DATASET_NAMES = {1: "Dataset001_ELM2D", 2: "Dataset002_ELM3D"}


def write_csv(path: Path, rows: list, fieldnames: list = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_prediction(images_tr, case_ids, file_ending, dataset_id, configuration,
                    plans, trainer, fold, checkpoint, step_size, disable_tta,
                    in_dir: Path, out_dir: Path):
    """
    Runs nnUNetv2_predict command through execvp on the given case_ids, writing predictions to out_dir.
    """
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid in case_ids:
        src = (images_tr / f"{cid}_0000{file_ending}").resolve()
        dst = in_dir / f"{cid}_0000{file_ending}"
        if not dst.exists():
            dst.symlink_to(src)

    cmd = [
        "nnUNetv2_predict",
        "-i", str(in_dir),
        "-o", str(out_dir),
        "-d", str(dataset_id),
        "-c", configuration,
        "-p", plans,
        "-tr", trainer,
        "-f", str(fold),
        "-chk", checkpoint,
        "-step_size", str(step_size),
    ]
    if disable_tta:
        cmd.append("--disable_tta")
    subprocess.run(cmd, check=True)


def score_2d_fold(case_ids, pred_dir, labels_tr, file_ending, fold, tol,
                   hole_decomposition=False, min_gap_width=5):
    """Per-slice metrics + pooled per-eye aggregation, matching predict_cv2d.py."""
    slice_rows = []
    patient_agg = defaultdict(
        lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "rmse": [], "assd": [],
                  "hd95": [], "bf1_tol2": [], "surf_dice_tol2": [], "n_slices": 0,
                  "gap_geoms": [], "spurious_records": []}
    )

    for cid in case_ids:
        pid, slice_idx = cid.rsplit("-", 1)
        pred = np.array(Image.open(pred_dir / f"{cid}{file_ending}"))
        gt = np.array(Image.open(labels_tr / f"{cid}{file_ending}"))
        pred01 = (pred > 0).astype(np.uint8)
        gt01 = (gt > 0).astype(np.uint8)

        tp, fp, tn, fn = confusion_counts(pred01, gt01)
        dice, iou, sen, fpr = dice_iou_sen_fpr(tp, fp, tn, fn)
        r = rmse(pred01, gt01)
        assd, hd, hd95 = assd_hd_hd95_2d(pred01, gt01)
        bf1 = boundary_f1_2d(pred01, gt01, tol=tol)
        sdice = surface_dice_2d(pred01, gt01, tol=tol)

        slice_row = {
            "fold": fold, "eye_id": pid, "slice_idx": int(slice_idx), "image_name": cid,
            "dice": dice, "iou": iou, "sen": sen, "fpr": fpr, "rmse": r,
            "assd": assd, "hd": hd, "hd95": hd95, "bf1_tol2": bf1, "surf_dice_tol2": sdice,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

        pa = patient_agg[pid]
        pa["tp"] += tp; pa["fp"] += fp; pa["tn"] += tn; pa["fn"] += fn
        pa["rmse"].append(r); pa["assd"].append(assd); pa["hd95"].append(hd95)
        pa["bf1_tol2"].append(bf1); pa["surf_dice_tol2"].append(sdice)
        pa["n_slices"] += 1

        if hole_decomposition:
            gap_r = analyze_slice(pred01, gt01, min_gap_width=min_gap_width)
            slice_row.update(gap_result_to_row_fields(gap_r))
            if gap_r is not None:
                if gap_r["gt_has_gap"]:
                    pa["gap_geoms"].append(gap_r)
                else:
                    pa["spurious_records"].append(gap_r)

        slice_rows.append(slice_row)

    patient_rows = []
    for pid, pa in patient_agg.items():
        pdice, piou, psen, pfpr = dice_iou_sen_fpr(pa["tp"], pa["fp"], pa["tn"], pa["fn"])
        m_rmse, _ = summarize_list(pa["rmse"])
        m_assd, _ = summarize_list(pa["assd"])
        m_hd95, _ = summarize_list(pa["hd95"])
        m_bf1, _ = summarize_list(pa["bf1_tol2"])
        m_sdice, _ = summarize_list(pa["surf_dice_tol2"])
        patient_row = {
            "fold": fold, "eye_id": pid, "tp": pa["tp"], "fp": pa["fp"], "tn": pa["tn"], "fn": pa["fn"],
            "vol_dice_pooled": pdice, "vol_iou_pooled": piou, "vol_sen_pooled": psen, "vol_fpr_pooled": pfpr,
            "vol_rmse_mean": m_rmse, "vol_assd_mean": m_assd, "vol_hd95_mean": m_hd95,
            "vol_bf1_mean": m_bf1, "vol_sdice_mean": m_sdice, "n_slices": pa["n_slices"],
        }
        if hole_decomposition:
            patient_row.update(summarize_gap_geometry(pa["gap_geoms"]))
            patient_row.update(summarize_spurious_gaps(pa["spurious_records"]))
        patient_rows.append(patient_row)
    patient_rows.sort(key=lambda r: r["eye_id"])
    return slice_rows, patient_rows


def score_3d_fold(case_ids, pred_dir, labels_tr, file_ending, fold, tol, spacing,
                   hole_decomposition=False, min_gap_width=5):
    """Per-volume metrics, matching predict_cv3d.py."""
    volume_rows = []
    gap_rows = []
    for cid in case_ids:
        pred_img = sitk.ReadImage(str(pred_dir / f"{cid}{file_ending}"))
        gt_img = sitk.ReadImage(str(labels_tr / f"{cid}{file_ending}"))
        pred01 = (sitk.GetArrayFromImage(pred_img) > 0).astype(np.uint8)
        gt01 = (sitk.GetArrayFromImage(gt_img) > 0).astype(np.uint8)

        tp, fp, tn, fn = confusion_counts(pred01, gt01)
        dice, iou, sen, fpr = dice_iou_sen_fpr(tp, fp, tn, fn)
        r = rmse(pred01, gt01)
        assd, hd, hd95 = assd_hd_hd95_3d(pred01, gt01, spacing=spacing)
        bf1 = boundary_f1_3d(pred01, gt01, tol_vox=tol, spacing=spacing)
        sdice = surface_dice_3d(pred01, gt01, tol_vox=tol, spacing=spacing)

        volume_row = {
            "fold": fold, "eye_id": cid, "dice": dice, "iou": iou, "sen": sen, "fpr": fpr,
            "rmse": r, "assd": assd, "hd": hd, "hd95": hd95, "bf1_tol2": bf1, "surf_dice_tol2": sdice,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "depth": int(pred01.shape[0]), "height": int(pred01.shape[1]), "width": int(pred01.shape[2]),
        }

        if hole_decomposition:
            records = analyze_slice_3d(pred01, gt01, min_gap_width=min_gap_width)
            hole_recs = [g for _, g in records if g["gt_has_gap"]]
            cont_recs = [g for _, g in records if not g["gt_has_gap"]]
            volume_row.update(summarize_gap_geometry(hole_recs))
            volume_row.update(summarize_spurious_gaps(cont_recs))
            for slice_idx, g in records:
                gap_rows.append({
                    "fold": fold, "eye_id": cid, "slice_idx": slice_idx,
                    **gap_result_to_row_fields(g),
                })

        volume_rows.append(volume_row)

    volume_rows.sort(key=lambda r: r["eye_id"])
    patient_rows = [{
        "fold": r["fold"], "eye_id": r["eye_id"], "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"],
        "vol_dice_pooled": r["dice"], "vol_iou_pooled": r["iou"], "vol_sen_pooled": r["sen"], "vol_fpr_pooled": r["fpr"],
        "vol_rmse_mean": r["rmse"], "vol_assd_mean": r["assd"], "vol_hd_mean": r["hd"], "vol_hd95_mean": r["hd95"],
        "vol_bf1_mean": r["bf1_tol2"], "vol_sdice_mean": r["surf_dice_tol2"],
        "depth": r["depth"], "height": r["height"], "width": r["width"],
        **({k: r[k] for k in (
            "n_hole_slices", "n_bridged", "bridged_frac", "mean_width_error",
            "mean_abs_width_error", "mean_left_margin_error", "mean_right_margin_error",
            "n_continuous_slices", "n_spurious_gaps", "spurious_gap_frac", "mean_spurious_gap_width",
        )} if hole_decomposition else {}),
    } for r in volume_rows]
    return volume_rows, patient_rows, gap_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_id", type=int, required=True, choices=[1, 2])
    ap.add_argument("--configuration", required=True, choices=["2d", "3d_fullres"])
    ap.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    ap.add_argument("--trainer", default="nnUNetTrainer")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--num_folds", type=int, default=5)
    ap.add_argument("--step_size", type=float, default=0.5)
    ap.add_argument("--disable_tta", action="store_true")
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument(
        "--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("Z", "Y", "X"),
        help="Voxel spacing (z,y,x, SimpleITK array order) for 3D distance metrics "
             "(Dataset002_ELM3D only). Default is voxel units (1,1,1); pass this "
             "dataset's real acquisition spacing (0.029 0.00387 0.00547) for "
             "physically calibrated distances instead.",
    )
    ap.add_argument("--out_dir", default=None, help="Default: <nnunet/nnUNet_results model dir>/cv_eval")
    ap.add_argument("--keep_predictions", action="store_true",
                     help="Keep each fold's raw predicted segmentations under <out_dir>/fold_<k>_predictions")
    ap.add_argument(
        "--hole_decomposition", action="store_true",
        help="On slices that cross the macular hole (an interior gap in the "
             "annotated ELM line), compare the model's own predicted gap against "
             "the GT gap: whether it bridged straight across (no gap at all), and "
             "if not, how its width and margins (ELM termination points) compare. "
             "On slices where the annotated line is continuous, also checks whether "
             "the model predicts a spurious gap anyway (a false-positive hole). Adds "
             "gap_* fields to the per-slice/per-volume and per-patient CSVs (plus a "
             "dedicated gap-analysis CSV for the 3D/volume case).",
    )
    ap.add_argument(
        "--min_gap_width", type=int, default=5,
        help="Minimum run length (columns) of missing GT columns to count as a "
             "hole, filtering out annotation jitter (only used with "
             "--hole_decomposition).",
    )
    args = ap.parse_args()

    for env_var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if env_var not in os.environ:
            raise RuntimeError(f"{env_var} is not set -- did you `source nnUnet.sh`?")

    dataset_name = DATASET_NAMES[args.dataset_id]
    is_2d_dataset = args.dataset_id == 1

    raw_root = Path(os.environ["nnUNet_raw"]) / dataset_name
    preprocessed_root = Path(os.environ["nnUNet_preprocessed"]) / dataset_name
    model_root = Path(os.environ["nnUNet_results"]) / dataset_name / f"{args.trainer}__{args.plans}__{args.configuration}"

    dataset_json = json.loads((raw_root / "dataset.json").read_text())
    file_ending = dataset_json["file_ending"]
    images_tr = raw_root / "imagesTr"
    labels_tr = raw_root / "labelsTr"

    test_cases_per_fold = {
        e["fold"]: e["test_cases"]
        for e in json.loads((preprocessed_root / "test_cases_per_fold.json").read_text())
    }

    default_dir_name = "cv_eval_hole" if args.hole_decomposition else "cv_eval"
    out_dir = Path(args.out_dir) if args.out_dir else (model_root / default_dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    spacing = tuple(args.spacing)

    metric_names = ["dice", "iou", "sen", "fpr", "rmse", "assd", "hd95", "bf1_tol2", "surf_dice_tol2"]
    if not is_2d_dataset:
        metric_names.insert(6, "hd")  # predict_cv3d.py also tracks plain HD

    fold_summary_rows = []
    fold_metric_means = defaultdict(list)
    all_patient_rows = []
    all_slice_rows = []
    all_gap_rows = []

    for fold in tqdm(range(args.num_folds), desc="Folds"):
        case_ids = test_cases_per_fold[fold]
        ckpt_path = model_root / f"fold_{fold}" / args.checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {ckpt_path}")

        tmp_in = Path(tempfile.mkdtemp(prefix=f"nnunet_predict_in_f{fold}_"))
        if args.keep_predictions:
            pred_dir = out_dir / f"fold_{fold}_predictions"
            if pred_dir.exists():
                shutil.rmtree(pred_dir)
            tmp_out = None
        else:
            tmp_out = Path(tempfile.mkdtemp(prefix=f"nnunet_predict_out_f{fold}_"))
            pred_dir = tmp_out

        try:
            run_prediction(
                images_tr, case_ids, file_ending, args.dataset_id, args.configuration,
                args.plans, args.trainer, fold, args.checkpoint, args.step_size,
                args.disable_tta, tmp_in, pred_dir,
            )

            if is_2d_dataset:
                slice_rows, patient_rows = score_2d_fold(
                    case_ids, pred_dir, labels_tr, file_ending, fold, args.tol,
                    hole_decomposition=args.hole_decomposition, min_gap_width=args.min_gap_width,
                )
                all_slice_rows.extend(slice_rows)
                write_csv(out_dir / f"fold_{fold}_per_slice.csv", slice_rows)
                n_extra = {"n_slices": len(slice_rows)}
            else:
                volume_rows, patient_rows, gap_rows = score_3d_fold(
                    case_ids, pred_dir, labels_tr, file_ending, fold, args.tol, spacing,
                    hole_decomposition=args.hole_decomposition, min_gap_width=args.min_gap_width,
                )
                write_csv(out_dir / f"fold_{fold}_per_volume.csv", volume_rows)
                if args.hole_decomposition:
                    write_csv(out_dir / f"fold_{fold}_gap_analysis.csv", gap_rows)
                    all_gap_rows.extend(gap_rows)
                n_extra = {}
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)
            if tmp_out is not None:
                shutil.rmtree(tmp_out, ignore_errors=True)

        write_csv(out_dir / f"fold_{fold}_per_patient.csv", patient_rows)
        all_patient_rows.extend(patient_rows)

        def fold_mean(vol_key):
            vals = np.array([r[vol_key] for r in patient_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            return float(vals.mean()) if vals.size else float("nan")

        vol_key_map = {
            "dice": "vol_dice_pooled", "iou": "vol_iou_pooled", "sen": "vol_sen_pooled", "fpr": "vol_fpr_pooled",
            "rmse": "vol_rmse_mean", "assd": "vol_assd_mean", "hd": "vol_hd_mean", "hd95": "vol_hd95_mean",
            "bf1_tol2": "vol_bf1_mean", "surf_dice_tol2": "vol_sdice_mean",
        }
        fold_row = {"fold": fold, "checkpoint": str(ckpt_path), "n_patients": len(patient_rows), **n_extra}
        for m in metric_names:
            fold_row[m] = fold_mean(vol_key_map[m])
        fold_summary_rows.append(fold_row)
        for m in metric_names:
            fold_metric_means[m].append(fold_row[m])

        print(
            f"Fold {fold}: patients={fold_row['n_patients']} \n"
            f"Dice={fold_row['dice']:.6f} IoU={fold_row['iou']:.6f} "
            f"SEN={fold_row['sen']:.6f} FPR={fold_row['fpr']:.6f} "
            f"ASSD={fold_row['assd']:.6f} HD95={fold_row['hd95']:.6f} "
            f"BF1_tol2={fold_row['bf1_tol2']:.6f} SurfDice_tol2={fold_row['surf_dice_tol2']:.6f} "
        )

    unit = "per-patient" if is_2d_dataset else "per-volume"
    print(f"\n=== 5-fold CV ({dataset_name}/{args.configuration}, {unit} macro, mean±std across folds) ===")
    for m in metric_names:
        mu, sd = summarize_list(fold_metric_means[m])
        print(f"{m:14s}: {mu:.6f} ± {sd:.6f}")

    print("\n=== OOF (per-eye) mean±std across eyes ===")
    eye_summary_rows = []
    for m in metric_names:
        mu, sd = summarize_rows(all_patient_rows, vol_key_map[m])
        print(f"{m:14s}: {mu:.6f} ± {sd:.6f}")
        eye_summary_rows.append({"metric": m, "mean": mu, "std": sd, "n_eyes": len(all_patient_rows)})

    fold_summary_rows.sort(key=lambda r: r["fold"])
    all_patient_rows.sort(key=lambda r: (r["fold"], r["eye_id"]))
    write_csv(out_dir / "cv_fold_summary.csv", fold_summary_rows)
    write_csv(out_dir / "cv_per_patient_all_folds.csv", all_patient_rows)
    write_csv(out_dir / "cv_oof_eye_summary.csv", eye_summary_rows, fieldnames=["metric", "mean", "std", "n_eyes"])
    if is_2d_dataset:
        all_slice_rows.sort(key=lambda r: (r["fold"], r["eye_id"], r["slice_idx"]))
        write_csv(out_dir / "cv_per_slice_all_folds.csv", all_slice_rows)
    elif args.hole_decomposition:
        all_gap_rows.sort(key=lambda r: (r["fold"], r["eye_id"], r["slice_idx"]))
        write_csv(out_dir / "cv_gap_analysis_all_folds.csv", all_gap_rows)

    print(f"\n[Saved] -> {out_dir}")


if __name__ == "__main__":
    main()
