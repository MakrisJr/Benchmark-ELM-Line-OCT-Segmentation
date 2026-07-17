import argparse
import os
import re
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from elm.metrics import (
    confusion_counts,
    dice_iou_sen_fpr,
    rmse as rmse_pixel,
    boundary_f1_2d as boundary_f1,
    surface_dice_2d as surface_dice,
    assd_hd_hd95_2d as assd_and_hausdorff,
    summarize_list,
    summarize_rows,
)
from elm.hole_metrics import analyze_slice, gap_result_to_row_fields, summarize_gap_geometry, summarize_spurious_gaps


def load_native_mask(mask_dir: Path, image_name: str):
    return (np.array(Image.open(mask_dir / image_name)) > 0).astype(np.uint8)


def upsample_pred_2d(pred01: np.ndarray, target_hw):
    h, w = target_hw
    return cv2.resize(pred01, (w, h), interpolation=cv2.INTER_NEAREST)


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
        "--native_res",
        action="store_true",
        help="Score against native per-slice resolution instead of the 256x256 eval "
             "size: reloads ground-truth masks from disk at their original size and "
             "upsamples predictions to match (nearest-neighbor), for comparability "
             "with nnU-Net's native-resolution evaluation (see nnunet/predict_cv.py).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for CSVs (default: <model_root>/cv_eval)",
    )
    parser.add_argument(
        "--hole_decomposition",
        action="store_true",
        help="On slices that cross the macular hole (an interior gap in the "
             "annotated ELM line), compare the model's own predicted gap against "
             "the GT gap: whether it bridged straight across (no gap at all), and "
             "if not, how its width and margins (ELM termination points) compare. "
             "On slices where the annotated line is continuous, also checks "
             "whether the model predicts a spurious gap anyway (a false-positive "
             "hole). Adds gap_* fields to the per-slice CSV and summary fields to "
             "the per-patient CSV. Requires --native_res, since the hole is defined on "
             "native-resolution columns.",
    )
    parser.add_argument(
        "--min_gap_width", type=int, default=5,
        help="Minimum run length (native-res columns) of missing GT columns to "
             "count as a hole, filtering out annotation jitter (only used with "
             "--hole_decomposition).",
    )
    args = parser.parse_args()

    if args.hole_decomposition and not args.native_res:
        raise ValueError("--hole_decomposition requires --native_res")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root = Path(args.model_root)
    default_dir_name = "cv_eval_native" if args.native_res else "cv_eval"
    out_dir = Path(args.out_dir) if args.out_dir else (model_root / default_dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_summary_rows = []
    fold_patient_means = defaultdict(list)  # metric -> [fold_means]

    # Optional overall out-of-fold per-patient list (for a single overall mean)
    all_patient_rows = []
    all_slice_rows = []

    for fold in tqdm(range(args.num_folds), desc="Folds"):
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
        native_mask_dir = Path(data_root) / "all" / "mask"
        transform_val = make_2d_transforms(train=False, out_size=(256, 256))
        ds = BasicDataset(
            root_dir=data_root,
            split="test",
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
                "gap_geoms": [],
                "spurious_records": [],
            }
        )

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  Fold {fold} batches", leave=False):
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

                    if args.native_res:
                        gt_np = load_native_mask(native_mask_dir, image_name)
                        pred_np = upsample_pred_2d(pred_np, gt_np.shape)

                    tp, fp, tn, fn = confusion_counts(pred_np, gt_np)
                    dice, iou, sen, fpr = dice_iou_sen_fpr(tp, fp, tn, fn)
                    rmse = rmse_pixel(pred_np, gt_np)
                    assd, hd, hd95 = assd_and_hausdorff(pred_np, gt_np)
                    bf1 = boundary_f1(pred_np, gt_np, tol=args.tol)
                    sdice = surface_dice(pred_np, gt_np, tol=args.tol)

                    slice_row = {
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

                    if args.hole_decomposition:
                        r = analyze_slice(pred_np, gt_np, min_gap_width=args.min_gap_width)
                        slice_row.update(gap_result_to_row_fields(r))
                        if r is not None:
                            if r["gt_has_gap"]:
                                pa["gap_geoms"].append(r)
                            else:
                                pa["spurious_records"].append(r)

                    slice_rows.append(slice_row)

        # Convert patient aggregations into patient-level metrics (one row per patient)
        patient_rows = []
        for pid, pa in patient_agg.items():
            pdice, piou, psen, pfpr = dice_iou_sen_fpr(pa["tp"], pa["fp"], pa["tn"], pa["fn"])
            m_rmse, _ = summarize_list(pa["rmse"])
            m_assd, _ = summarize_list(pa["assd"])
            m_hd95, _ = summarize_list(pa["hd95"])
            m_bf1, _ = summarize_list(pa["bf1_tol2"])
            m_sdice, _ = summarize_list(pa["surf_dice_tol2"])

            patient_row = {
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

            if args.hole_decomposition:
                patient_row.update(summarize_gap_geometry(pa["gap_geoms"]))
                patient_row.update(summarize_spurious_gaps(pa["spurious_records"]))

            patient_rows.append(patient_row)

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
    # Each eye is evaluated exactly once (on its own test fold), so this is a
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
python predict_cv2d.py --model_root elm-results/SegNet_Jun-04-2026_1243_model --model SegNet

R2U_Net:
python predict_cv2d.py --model_root elm-results/R2U_Net_Apr-02-2026_0637_model --model R2U_Net
python predict_cv2d.py --model_root elm-results/R2U_Net_Jun-04-2026_0441_model --model R2U_Net

SwinEncoderUNet2D:
python predict_cv2d.py --model_root elm-results/SwinEncoderUNet2D_Apr-01-2026_1633_model --model SwinEncoderUNet2D
python predict_cv2d.py --model_root elm-results/SwinEncoderUNet2D_Jun-03-2026_1639_model --model SwinEncoderUNet2D

"""
