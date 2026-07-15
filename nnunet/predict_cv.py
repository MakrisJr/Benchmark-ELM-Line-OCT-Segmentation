#!/usr/bin/env python3
"""
Predict + score nnU-Net's 5-fold CV test splits.

Runs, for one trained (dataset, configuration) combo, one nnUNetv2_predict
call per fold -- using only that fold's model on that fold's held-out test
cases (nnUNet_preprocessed/<dataset>/test_cases_per_fold.json) -- then scores
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
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

DATASET_NAMES = {1: "Dataset001_ELM2D", 2: "Dataset002_ELM3D"}


def safe_div(n, d, eps=1e-12):
    return float(n) / float(d + eps)


def confusion_counts(pred01: np.ndarray, gt01: np.ndarray):
    tp = int(np.logical_and(pred01 == 1, gt01 == 1).sum())
    fp = int(np.logical_and(pred01 == 1, gt01 == 0).sum())
    tn = int(np.logical_and(pred01 == 0, gt01 == 0).sum())
    fn = int(np.logical_and(pred01 == 0, gt01 == 1).sum())
    return tp, fp, tn, fn


def dice_iou_sen_fpr(tp, fp, tn, fn, eps=1e-12):
    dice = safe_div(2 * tp, 2 * tp + fp + fn, eps)
    iou = safe_div(tp, tp + fp + fn, eps)
    sen = safe_div(tp, tp + fn, eps)
    fpr = safe_div(fp, fp + tn, eps)
    return dice, iou, sen, fpr


def rmse(pred01: np.ndarray, gt01: np.ndarray):
    diff = pred01.astype(np.float32) - gt01.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))


def summarize_list(xs):
    xs = np.array(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(xs.std(ddof=1)) if xs.size > 1 else 0.0


def summarize_rows(rows, key):
    return summarize_list([r.get(key, float("nan")) for r in rows])


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


# ---- 2D boundary metrics (identical to predict_cv2d.py) ----

def extract_boundary_2d(mask01: np.ndarray):
    mask255 = mask01.astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    er = cv2.erode(mask255, k, iterations=1)
    return (cv2.subtract(mask255, er) > 0).astype(np.uint8)


def boundary_f1_2d(pred01, gt01, tol=2):
    pb, gb = extract_boundary_2d(pred01), extract_boundary_2d(gt01)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    pb_d, gb_d = cv2.dilate(pb, k, iterations=1), cv2.dilate(gb, k, iterations=1)
    prec = safe_div(np.logical_and(pb == 1, gb_d == 1).sum(), pb.sum())
    rec = safe_div(np.logical_and(gb == 1, pb_d == 1).sum(), gb.sum())
    return float(safe_div(2 * prec * rec, prec + rec))


def surface_dice_2d(pred01, gt01, tol=2):
    pb, gb = extract_boundary_2d(pred01), extract_boundary_2d(gt01)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    pb_d, gb_d = cv2.dilate(pb, k, iterations=1), cv2.dilate(gb, k, iterations=1)
    inter = np.logical_and(pb == 1, gb_d == 1).sum() + np.logical_and(gb == 1, pb_d == 1).sum()
    return safe_div(inter, pb.sum() + gb.sum())


def assd_hd_hd95_2d(pred01, gt01):
    pb, gb = extract_boundary_2d(pred01), extract_boundary_2d(gt01)
    if pb.sum() == 0 and gb.sum() == 0:
        return 0.0, 0.0, 0.0
    if pb.sum() == 0 or gb.sum() == 0:
        return float("inf"), float("inf"), float("inf")
    dt_g = cv2.distanceTransform((1 - gb).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)
    dt_p = cv2.distanceTransform((1 - pb).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)
    d_p_to_g = dt_g[pb == 1].astype(np.float64)
    d_g_to_p = dt_p[gb == 1].astype(np.float64)
    all_d = np.concatenate([d_p_to_g, d_g_to_p])
    return float(all_d.mean()), float(all_d.max()), float(np.percentile(all_d, 95))


# ---- 3D boundary metrics (identical to predict_cv3d.py) ----

def surface_voxels_3d(mask01: np.ndarray) -> np.ndarray:
    mask = mask01.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndi.binary_erosion(mask, structure=structure, iterations=1, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def assd_hd_hd95_3d(pred01, gt01, spacing=(1.0, 1.0, 1.0)):
    pred_s, gt_s = surface_voxels_3d(pred01), surface_voxels_3d(gt01)
    if pred_s.sum() == 0 and gt_s.sum() == 0:
        return 0.0, 0.0, 0.0
    if pred_s.sum() == 0 or gt_s.sum() == 0:
        return float("inf"), float("inf"), float("inf")
    dt_gt = ndi.distance_transform_edt(~gt_s, sampling=spacing)
    dt_pred = ndi.distance_transform_edt(~pred_s, sampling=spacing)
    d_pred_to_gt = dt_gt[pred_s].astype(np.float64)
    d_gt_to_pred = dt_pred[gt_s].astype(np.float64)
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(all_d.mean()), float(all_d.max()), float(np.percentile(all_d, 95))


def boundary_f1_3d(pred01, gt01, tol_vox=2, spacing=(1.0, 1.0, 1.0)):
    pred_s, gt_s = surface_voxels_3d(pred01), surface_voxels_3d(gt01)
    if pred_s.sum() == 0 and gt_s.sum() == 0:
        return 1.0
    if pred_s.sum() == 0 or gt_s.sum() == 0:
        return 0.0
    dt_gt = ndi.distance_transform_edt(~gt_s, sampling=spacing)
    dt_pred = ndi.distance_transform_edt(~pred_s, sampling=spacing)
    prec = safe_div((dt_gt[pred_s] <= tol_vox).sum(), pred_s.sum())
    rec = safe_div((dt_pred[gt_s] <= tol_vox).sum(), gt_s.sum())
    return float(safe_div(2 * prec * rec, prec + rec))


def surface_dice_3d(pred01, gt01, tol_vox=2, spacing=(1.0, 1.0, 1.0)):
    pred_s, gt_s = surface_voxels_3d(pred01), surface_voxels_3d(gt01)
    if pred_s.sum() == 0 and gt_s.sum() == 0:
        return 1.0
    if pred_s.sum() == 0 or gt_s.sum() == 0:
        return 0.0
    dt_gt = ndi.distance_transform_edt(~gt_s, sampling=spacing)
    dt_pred = ndi.distance_transform_edt(~pred_s, sampling=spacing)
    matched_pred = (dt_gt[pred_s] <= tol_vox).sum()
    matched_gt = (dt_pred[gt_s] <= tol_vox).sum()
    return float(safe_div(matched_pred + matched_gt, pred_s.sum() + gt_s.sum()))


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


def score_2d_fold(case_ids, pred_dir, labels_tr, file_ending, fold, tol):
    """Per-slice metrics + pooled per-eye aggregation, matching predict_cv2d.py."""
    slice_rows = []
    patient_agg = defaultdict(
        lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "rmse": [], "assd": [],
                  "hd95": [], "bf1_tol2": [], "surf_dice_tol2": [], "n_slices": 0}
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

        slice_rows.append({
            "fold": fold, "eye_id": pid, "slice_idx": int(slice_idx), "image_name": cid,
            "dice": dice, "iou": iou, "sen": sen, "fpr": fpr, "rmse": r,
            "assd": assd, "hd": hd, "hd95": hd95, "bf1_tol2": bf1, "surf_dice_tol2": sdice,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

        pa = patient_agg[pid]
        pa["tp"] += tp; pa["fp"] += fp; pa["tn"] += tn; pa["fn"] += fn
        pa["rmse"].append(r); pa["assd"].append(assd); pa["hd95"].append(hd95)
        pa["bf1_tol2"].append(bf1); pa["surf_dice_tol2"].append(sdice)
        pa["n_slices"] += 1

    patient_rows = []
    for pid, pa in patient_agg.items():
        pdice, piou, psen, pfpr = dice_iou_sen_fpr(pa["tp"], pa["fp"], pa["tn"], pa["fn"])
        m_rmse, _ = summarize_list(pa["rmse"])
        m_assd, _ = summarize_list(pa["assd"])
        m_hd95, _ = summarize_list(pa["hd95"])
        m_bf1, _ = summarize_list(pa["bf1_tol2"])
        m_sdice, _ = summarize_list(pa["surf_dice_tol2"])
        patient_rows.append({
            "fold": fold, "eye_id": pid, "tp": pa["tp"], "fp": pa["fp"], "tn": pa["tn"], "fn": pa["fn"],
            "vol_dice_pooled": pdice, "vol_iou_pooled": piou, "vol_sen_pooled": psen, "vol_fpr_pooled": pfpr,
            "vol_rmse_mean": m_rmse, "vol_assd_mean": m_assd, "vol_hd95_mean": m_hd95,
            "vol_bf1_mean": m_bf1, "vol_sdice_mean": m_sdice, "n_slices": pa["n_slices"],
        })
    patient_rows.sort(key=lambda r: r["eye_id"])
    return slice_rows, patient_rows


def score_3d_fold(case_ids, pred_dir, labels_tr, file_ending, fold, tol, spacing):
    """Per-volume metrics, matching predict_cv3d.py."""
    volume_rows = []
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

        volume_rows.append({
            "fold": fold, "eye_id": cid, "dice": dice, "iou": iou, "sen": sen, "fpr": fpr,
            "rmse": r, "assd": assd, "hd": hd, "hd95": hd95, "bf1_tol2": bf1, "surf_dice_tol2": sdice,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "depth": int(pred01.shape[0]), "height": int(pred01.shape[1]), "width": int(pred01.shape[2]),
        })

    volume_rows.sort(key=lambda r: r["eye_id"])
    patient_rows = [{
        "fold": r["fold"], "eye_id": r["eye_id"], "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"],
        "vol_dice_pooled": r["dice"], "vol_iou_pooled": r["iou"], "vol_sen_pooled": r["sen"], "vol_fpr_pooled": r["fpr"],
        "vol_rmse_mean": r["rmse"], "vol_assd_mean": r["assd"], "vol_hd_mean": r["hd"], "vol_hd95_mean": r["hd95"],
        "vol_bf1_mean": r["bf1_tol2"], "vol_sdice_mean": r["surf_dice_tol2"],
        "depth": r["depth"], "height": r["height"], "width": r["width"],
    } for r in volume_rows]
    return volume_rows, patient_rows


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
    ap.add_argument("--out_dir", default=None, help="Default: <nnUNet_results model dir>/cv_eval")
    ap.add_argument("--keep_predictions", action="store_true",
                     help="Keep each fold's raw predicted segmentations under <out_dir>/fold_<k>_predictions")
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

    out_dir = Path(args.out_dir) if args.out_dir else (model_root / "cv_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    spacing = tuple(args.spacing)

    metric_names = ["dice", "iou", "sen", "fpr", "rmse", "assd", "hd95", "bf1_tol2", "surf_dice_tol2"]
    if not is_2d_dataset:
        metric_names.insert(6, "hd")  # predict_cv3d.py also tracks plain HD

    fold_summary_rows = []
    fold_metric_means = defaultdict(list)
    all_patient_rows = []
    all_slice_rows = []

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
                slice_rows, patient_rows = score_2d_fold(case_ids, pred_dir, labels_tr, file_ending, fold, args.tol)
                all_slice_rows.extend(slice_rows)
                write_csv(out_dir / f"fold_{fold}_per_slice.csv", slice_rows)
                n_extra = {"n_slices": len(slice_rows)}
            else:
                volume_rows, patient_rows = score_3d_fold(case_ids, pred_dir, labels_tr, file_ending, fold, args.tol, spacing)
                write_csv(out_dir / f"fold_{fold}_per_volume.csv", volume_rows)
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

    print(f"\n[Saved] -> {out_dir}")


if __name__ == "__main__":
    main()
