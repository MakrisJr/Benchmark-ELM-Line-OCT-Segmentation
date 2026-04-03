import argparse
import os
import re
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from elm.dataset import BasicDataset, make_2d_transforms
from elm.model import (
    U_Net,
    AttU_Net,
    LinkNetImprove,
    U2NETP,
    R2U_Net,
    DeepLabv3_plus,
    FCN,
    SegNet,
    SwinEncoderUNet2D,
)


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


def rmse_pixel(pred01: np.ndarray, gt01: np.ndarray):
    diff = pred01.astype(np.float32) - gt01.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))


def extract_boundary(mask01: np.ndarray):
    mask255 = (mask01.astype(np.uint8) * 255)
    k = np.ones((3, 3), np.uint8)
    er = cv2.erode(mask255, k, iterations=1)
    b = cv2.subtract(mask255, er)
    return (b > 0).astype(np.uint8)


def boundary_f1(pred01: np.ndarray, gt01: np.ndarray, tol=2):
    pb = extract_boundary(pred01)
    gb = extract_boundary(gt01)

    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    pb_d = cv2.dilate(pb, k, iterations=1)
    gb_d = cv2.dilate(gb, k, iterations=1)

    prec = safe_div(np.logical_and(pb == 1, gb_d == 1).sum(), pb.sum())
    rec = safe_div(np.logical_and(gb == 1, pb_d == 1).sum(), gb.sum())
    f1 = safe_div(2 * prec * rec, (prec + rec))
    return float(f1)


def surface_dice(pred01: np.ndarray, gt01: np.ndarray, tol=2):
    pb = extract_boundary(pred01)
    gb = extract_boundary(gt01)

    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    pb_d = cv2.dilate(pb, k, iterations=1)
    gb_d = cv2.dilate(gb, k, iterations=1)

    inter = (
        np.logical_and(pb == 1, gb_d == 1).sum()
        + np.logical_and(gb == 1, pb_d == 1).sum()
    )
    denom = (pb.sum() + gb.sum())
    return safe_div(inter, denom)


def distance_transform(mask01: np.ndarray):
    return cv2.distanceTransform(mask01.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)


def assd_and_hausdorff(pred01: np.ndarray, gt01: np.ndarray):
    pb = extract_boundary(pred01)
    gb = extract_boundary(gt01)

    if pb.sum() == 0 and gb.sum() == 0:
        return 0.0, 0.0, 0.0
    if pb.sum() == 0 or gb.sum() == 0:
        return float("inf"), float("inf"), float("inf")

    dt_g = distance_transform(1 - gb)
    dt_p = distance_transform(1 - pb)

    d_p_to_g = dt_g[pb == 1].astype(np.float64)
    d_g_to_p = dt_p[gb == 1].astype(np.float64)

    all_d = np.concatenate([d_p_to_g, d_g_to_p], axis=0)
    assd = float(all_d.mean())
    hd = float(all_d.max())
    hd95 = float(np.percentile(all_d, 95))
    return assd, hd, hd95


def summarize_list(xs):
    xs = np.array(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(xs.std(ddof=1)) if xs.size > 1 else 0.0


def summarize_rows(rows: list, key: str):
    vals = [r.get(key, float("nan")) for r in rows]
    return summarize_list(vals)


def build_model(model_name: str):
    if model_name == "SwinEncoderUNet2D":
        return SwinEncoderUNet2D(
            n_channels=3,
            n_classes=1,
            backbone="swin_tiny_patch4_window7_224",
            pretrained=False,
        )
    if model_name == "U_Net":
        return U_Net(n_channels=3, n_classes=1)
    if model_name == "AttU_Net":
        return AttU_Net(n_channels=3, n_classes=1)
    if model_name == "LinkNetImprove":
        return LinkNetImprove(n_channels=3, n_classes=1)
    if model_name == "U2NETP":
        return U2NETP(in_ch=3, out_ch=1)
    if model_name == "R2U_Net":
        return R2U_Net(n_channels=3, n_classes=1, t=2)
    if model_name == "DeepLabv3_plus":
        return DeepLabv3_plus(n_channels=3, n_classes=1, os=16, pretrained=False, _print=False)
    if model_name == "FCN":
        return FCN(n_channels=3, n_classes=1)
    if model_name == "SegNet":
        return SegNet(n_channels=3, n_classes=1)
    raise ValueError(f"Unsupported model: {model_name}")


def find_fold_checkpoint(model_root: Path, fold: int):
    """Return fold checkpoint.

    Expected layout (your repo):
      elm-results/<MODEL_NAME>/fold_<k>/checkpoints/<single .pth>

    We pick the only .pth in that directory. If multiple exist, we prefer
    *_best_epoch_*.pth; otherwise we fall back to newest by mtime.
    """

    fold_dir = model_root / f"fold_{fold}" / "checkpoints"
    if not fold_dir.exists():
        return None

    candidates = sorted(fold_dir.glob("*.pth"))
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    best = [p for p in candidates if "_best_epoch_" in p.name]
    if len(best) == 1:
        return best[0]

    # fallback: newest by modification time
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


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


def main():
    parser = argparse.ArgumentParser(description="5-fold CV inference (2D) with per-patient macro metrics")
    parser.add_argument("--base_dir", type=str, default="./")
    parser.add_argument(
        "--model_root",
        type=str,
        required=True,
        help="Path to model root containing fold_0..fold_4 (e.g. ./elm-results/<model_name>)",
    )
    parser.add_argument("--model", type=str, default="SegNet")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tol", type=int, default=2)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for CSVs (default: <model_root>/cv_eval)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root = Path(args.model_root)
    out_dir = Path(args.out_dir) if args.out_dir else (model_root / "cv_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_summary_rows = []
    fold_patient_means = defaultdict(list)  # metric -> [fold_means]

    # Optional overall out-of-fold per-patient list (for a single overall mean)
    all_patient_rows = []
    all_slice_rows = []

    for fold in range(args.num_folds):
        ckpt = find_fold_checkpoint(model_root, fold)
        if ckpt is None:
            raise FileNotFoundError(
                f"Could not find checkpoint for fold {fold} under {model_root}. "
                f"Expected {model_root}/fold_{fold}/checkpoints/*.pth"
            )

        net = build_model(args.model)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        net.load_state_dict(state)
        net.to(device)
        net.eval()

        data_root = os.path.join(args.base_dir, "data_no_anomalies")
        transform_val = make_2d_transforms(train=False, out_size=(256, 256))
        ds = BasicDataset(
            root_dir=data_root,
            split="val",
            fold=fold,
            scale=1.0,
            transform=transform_val,
            single_channel=False,
        )

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # Per-slice metric storage
        slice_rows = []

        # Per-patient aggregation within fold
        patient_agg = defaultdict(
            lambda: {
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "hd95": [],
                "assd": [],
                "bf1_tol2": [],
                "surf_dice_tol2": [],
                "rmse": [],
                "n_slices": 0,
            }
        )

        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device=device, dtype=torch.float32)
                gts = batch["mask"].to(device=device, dtype=torch.float32)

                logits = net(imgs)
                prob = torch.sigmoid(logits)
                pred = (prob > args.threshold).to(torch.uint8)
                gt_bin = (gts > 0.5).to(torch.uint8)

                B = pred.shape[0]
                for i in range(B):
                    pid = batch["patient_id"][i]
                    image_name = batch.get("image_name")[i]
                    slice_idx = batch.get("slice_idx")[i]

                    pred_np = pred[i].squeeze().detach().cpu().numpy().astype(np.uint8)
                    gt_np = gt_bin[i].squeeze().detach().cpu().numpy().astype(np.uint8)

                    tp, fp, tn, fn = confusion_counts(pred_np, gt_np)
                    dice, iou, sen, fpr = dice_iou_sen_fpr(tp, fp, tn, fn)
                    rmse = rmse_pixel(pred_np, gt_np)
                    assd, hd, hd95 = assd_and_hausdorff(pred_np, gt_np)
                    bf1 = boundary_f1(pred_np, gt_np, tol=args.tol)
                    sdice = surface_dice(pred_np, gt_np, tol=args.tol)

                    slice_rows.append(
                        {
                            "fold": fold,
                            "eye_id": pid,
                            "slice_idx": int(slice_idx),
                            "image_name": image_name,
                            "dice": dice,
                            "iou": iou,
                            "sen": sen,
                            "fpr": fpr,
                            "rmse": rmse,
                            "assd": assd,
                            "hd": hd,
                            "hd95": hd95,
                            "bf1_tol2": bf1,
                            "surf_dice_tol2": sdice,
                            "tp": tp,
                            "fp": fp,
                            "tn": tn,
                            "fn": fn,
                        }
                    )

                    pa = patient_agg[pid]
                    pa["tp"] += tp
                    pa["fp"] += fp
                    pa["tn"] += tn
                    pa["fn"] += fn
                    pa["rmse"].append(rmse)
                    pa["assd"].append(assd)
                    pa["hd95"].append(hd95)
                    pa["bf1_tol2"].append(bf1)
                    pa["surf_dice_tol2"].append(sdice)
                    pa["n_slices"] += 1

        # Convert patient aggregations into patient-level metrics (one row per patient)
        patient_rows = []
        for pid, pa in patient_agg.items():
            pdice, piou, psen, pfpr = dice_iou_sen_fpr(pa["tp"], pa["fp"], pa["tn"], pa["fn"])
            m_rmse, _ = summarize_list(pa["rmse"])
            m_assd, _ = summarize_list(pa["assd"])
            m_hd95, _ = summarize_list(pa["hd95"])
            m_bf1, _ = summarize_list(pa["bf1_tol2"])
            m_sdice, _ = summarize_list(pa["surf_dice_tol2"])

            patient_rows.append(
                {
                    "fold": fold,
                    "eye_id": pid,
                    "tp": pa["tp"],
                    "fp": pa["fp"],
                    "tn": pa["tn"],
                    "fn": pa["fn"],
                    "vol_dice_pooled": pdice,
                    "vol_iou_pooled": piou,
                    "vol_sen_pooled": psen,
                    "vol_fpr_pooled": pfpr,
                    "vol_rmse_mean": m_rmse,
                    "vol_assd_mean": m_assd,
                    "vol_hd95_mean": m_hd95,
                    "vol_bf1_mean": m_bf1,
                    "vol_sdice_mean": m_sdice,
                    "n_slices": pa["n_slices"],
                }
            )

        patient_rows.sort(key=lambda r: r["eye_id"])

        # Fold-level macro means over patients
        def fold_mean(key):
            vals = [r[key] for r in patient_rows]
            vals = np.array(vals, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            return float(vals.mean()) if vals.size else float("nan")

        fold_row = {
            "fold": fold,
            "checkpoint": str(ckpt),
            "n_patients": len(patient_rows),
            "n_slices": len(slice_rows),
            "dice": fold_mean("vol_dice_pooled"),
            "iou": fold_mean("vol_iou_pooled"),
            "sen": fold_mean("vol_sen_pooled"),
            "fpr": fold_mean("vol_fpr_pooled"),
            "rmse": fold_mean("vol_rmse_mean"),
            "assd": fold_mean("vol_assd_mean"),
            "hd95": fold_mean("vol_hd95_mean"),
            "bf1_tol2": fold_mean("vol_bf1_mean"),
            "surf_dice_tol2": fold_mean("vol_sdice_mean"),
        }
        fold_summary_rows.append(fold_row)

        for metric_key, fold_key in [
            ("dice", "dice"),
            ("iou", "iou"),
            ("sen", "sen"),
            ("fpr", "fpr"),
            ("rmse", "rmse"),
            ("assd", "assd"),
            ("hd95", "hd95"),
            ("bf1_tol2", "bf1_tol2"),
            ("surf_dice_tol2", "surf_dice_tol2"),
        ]:
            fold_patient_means[metric_key].append(fold_row[fold_key])

        # Write per-fold CSVs
        write_csv(out_dir / f"fold_{fold}_per_patient.csv", patient_rows)
        write_csv(out_dir / f"fold_{fold}_per_slice.csv", slice_rows)

        all_patient_rows.extend(patient_rows)
        all_slice_rows.extend(slice_rows)

        print(
            f"Fold {fold}: patients={fold_row['n_patients']} slices={fold_row['n_slices']} "
            f"Dice={fold_row['dice']:.6f} IoU={fold_row['iou']:.6f} SEN={fold_row['sen']:.6f} FPR={fold_row['fpr']:.6f}"
        )

    # CV mean ± std across folds (macro over folds)
    print("\n=== 5-fold CV (per-patient macro, mean±std across folds) ===")
    for metric in ["dice", "iou", "sen", "fpr", "rmse", "assd", "hd95", "bf1_tol2", "surf_dice_tol2"]:
        mu, sd = summarize_list(fold_patient_means[metric])
        print(f"{metric:14s}: {mu:.6f} ± {sd:.6f}")

    # OOF mean ± std across eyes (patients)
    # Each eye is evaluated exactly once (on its own validation fold), so this is a
    # standard deviation across eyes, not across folds.
    print("\n=== OOF (per-eye) mean±std across eyes ===")
    eye_summary_rows = []
    for name, key in [
        ("dice", "vol_dice_pooled"),
        ("iou", "vol_iou_pooled"),
        ("sen", "vol_sen_pooled"),
        ("fpr", "vol_fpr_pooled"),
        ("rmse", "vol_rmse_mean"),
        ("assd", "vol_assd_mean"),
        ("hd95", "vol_hd95_mean"),
        ("bf1_tol2", "vol_bf1_mean"),
        ("surf_dice_tol2", "vol_sdice_mean"),
    ]:
        mu, sd = summarize_rows(all_patient_rows, key)
        print(f"{name:14s}: {mu:.6f} ± {sd:.6f}")
        eye_summary_rows.append({"metric": name, "mean": mu, "std": sd, "n_eyes": len(all_patient_rows)})

    # Save summary CSVs
    fold_summary_rows.sort(key=lambda r: r["fold"])
    write_csv(out_dir / "cv_fold_summary.csv", fold_summary_rows)
    all_patient_rows.sort(key=lambda r: (r["fold"], r["eye_id"]))
    all_slice_rows.sort(key=lambda r: (r["fold"], r["eye_id"], (r["slice_idx"] if r["slice_idx"] is not None else -1)))
    write_csv(out_dir / "cv_per_patient_all_folds.csv", all_patient_rows)
    write_csv(out_dir / "cv_per_slice_all_folds.csv", all_slice_rows)
    write_csv(out_dir / "cv_oof_eye_summary.csv", eye_summary_rows, fieldnames=["metric", "mean", "std", "n_eyes"])

    print(f"\n[Saved] fold summary -> {out_dir / 'cv_fold_summary.csv'}")
    print(f"[Saved] per-patient (all folds) -> {out_dir / 'cv_per_patient_all_folds.csv'}")
    print(f"[Saved] per-slice (all folds) -> {out_dir / 'cv_per_slice_all_folds.csv'}")
    print(f"[Saved] eye summary -> {out_dir / 'cv_oof_eye_summary.csv'}")


if __name__ == "__main__":
    main()

"""
SegNet:
python predict_cv2d.py --model_root elm-results/SegNet_Apr-01-2026_1639_model --model SegNet

SwinEncoderUNet2D:
python predict_cv2d.py --model_root elm-results/SwinEncoderUNet2D_Apr-01-2026_1633_model --model SwinEncoderUNet2D
"""
