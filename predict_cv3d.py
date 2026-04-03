import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

from elm.dataset import D3Dataset
from elm.model import (
    CSAM_UNet2p5D,
    SwinUNETR3D,
    UNet2DEnc3DDec,
    UNet2p5D_SlidingWindow,
    UNet3D,
    UNet3D_Aniso,
    UNet3D_Aniso2,
    UNet3DFrawley,
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


def rmse_voxel(pred01: np.ndarray, gt01: np.ndarray):
    diff = pred01.astype(np.float32) - gt01.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))


def summarize_list(xs):
    xs = np.array(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(xs.std(ddof=1)) if xs.size > 1 else 0.0


def summarize_rows(rows: list, key: str):
    vals = [r.get(key, float("nan")) for r in rows]
    return summarize_list(vals)


def surface_voxels_3d(mask01: np.ndarray) -> np.ndarray:
    mask = mask01.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndi.binary_erosion(mask, structure=structure, iterations=1, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def assd_hd_hd95_3d(pred01: np.ndarray, gt01: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)

    pred_surface = surface_voxels_3d(pred)
    gt_surface = surface_voxels_3d(gt)

    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 0.0, 0.0, 0.0
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return float("inf"), float("inf"), float("inf")

    dt_gt = ndi.distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)

    d_pred_to_gt = dt_gt[pred_surface].astype(np.float64)
    d_gt_to_pred = dt_pred[gt_surface].astype(np.float64)
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)

    assd = float(all_d.mean())
    hd = float(all_d.max())
    hd95 = float(np.percentile(all_d, 95))
    return assd, hd, hd95


def boundary_f1_3d(pred01: np.ndarray, gt01: np.ndarray, tol_vox=2, spacing=(1.0, 1.0, 1.0)):
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)

    pred_surface = surface_voxels_3d(pred)
    gt_surface = surface_voxels_3d(gt)

    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 1.0
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return 0.0

    tol_dist = float(tol_vox)
    dt_gt = ndi.distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)

    prec = safe_div((dt_gt[pred_surface] <= tol_dist).sum(), pred_surface.sum())
    rec = safe_div((dt_pred[gt_surface] <= tol_dist).sum(), gt_surface.sum())
    return float(safe_div(2 * prec * rec, prec + rec))


def surface_dice_3d(pred01: np.ndarray, gt01: np.ndarray, tol_vox=2, spacing=(1.0, 1.0, 1.0)):
    pred = pred01.astype(bool)
    gt = gt01.astype(bool)

    pred_surface = surface_voxels_3d(pred)
    gt_surface = surface_voxels_3d(gt)

    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 1.0
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return 0.0

    tol_dist = float(tol_vox)
    dt_gt = ndi.distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)

    matched_pred = (dt_gt[pred_surface] <= tol_dist).sum()
    matched_gt = (dt_pred[gt_surface] <= tol_dist).sum()
    return float(safe_div(matched_pred + matched_gt, pred_surface.sum() + gt_surface.sum()))


def match_depth(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred_depth = pred.shape[2]
    gt_depth = gt.shape[2]
    if pred_depth == gt_depth:
        return pred
    if pred_depth > gt_depth:
        start = (pred_depth - gt_depth) // 2
        return pred[:, :, start:start + gt_depth, :, :]

    pad_total = gt_depth - pred_depth
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return torch.nn.functional.pad(
        pred,
        (0, 0, 0, 0, pad_left, pad_right),
        mode="constant",
        value=0,
    )


def build_model(model_name: str, pretrained_path: str = None, window_k: int = 7):
    if model_name == "UNet3D":
        return UNet3D(in_channels=1, out_channels=1)
    if model_name == "UNet3D_Aniso":
        return UNet3D_Aniso(in_channels=1, out_channels=1)
    if model_name == "UNet3D_Aniso2":
        return UNet3D_Aniso2(in_channels=1, out_channels=1)
    if model_name == "UNet3DFrawley":
        return UNet3DFrawley(in_channels=1, out_channels=1)
    if model_name == "UNet2DEnc3DDec":
        return UNet2DEnc3DDec(in_channels=1, out_channels=1)
    if model_name == "CSAM_UNet2p5D":
        return CSAM_UNet2p5D(
            in_channels=1,
            out_channels=1,
            num_layers=3,
            base_num=32,
            semantic=True,
            positional=True,
            slice_att=True,
        )
    if model_name == "UNet2p5D_SlidingWindow":
        return UNet2p5D_SlidingWindow(
            k=window_k,
            out_channels=1,
            num_layers=3,
            base_num=32,
            pad_mode="replicate",
        )
    if model_name == "SwinUNETR3D":
        return SwinUNETR3D(
            in_channels=1,
            n_classes=1,
            pretrained_path=pretrained_path,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def find_fold_checkpoint(model_root: Path, fold: int):
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

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def write_csv(path: Path, rows: list, fieldnames: list = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="5-fold CV inference (3D) with per-volume metrics for 3D models"
    )
    parser.add_argument("--base_dir", type=str, default="./")
    parser.add_argument(
        "--model_root",
        type=str,
        required=True,
        help="Path to model root containing fold_0..fold_N-1 checkpoints",
    )
    parser.add_argument("--model", type=str, default="SwinUNETR3D")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tol", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--window_k", type=int, default=7)
    parser.add_argument("--pretrained_path", type=str, default="./checkpoint/model_swinvit_UNETR.pt")
    parser.add_argument("--spacing_z", type=float, default=1.0)
    parser.add_argument("--spacing_y", type=float, default=1.0)
    parser.add_argument("--spacing_x", type=float, default=1.0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for CSVs (default: <model_root>/cv_eval_3d)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root = Path(args.model_root)
    out_dir = Path(args.out_dir) if args.out_dir else (model_root / "cv_eval_3d")
    out_dir.mkdir(parents=True, exist_ok=True)
    spacing = (args.spacing_z, args.spacing_y, args.spacing_x)

    fold_summary_rows = []
    fold_volume_means = defaultdict(list)
    all_volume_rows = []
    all_eye_summary_rows = []

    data_root = Path(args.base_dir) / "data_no_anomalies"

    for fold in range(args.num_folds):
        ckpt = find_fold_checkpoint(model_root, fold)
        if ckpt is None:
            raise FileNotFoundError(
                f"Could not find checkpoint for fold {fold} under {model_root}. "
                f"Expected {model_root}/fold_{fold}/checkpoints/*.pth"
            )

        net = build_model(
            model_name=args.model,
            pretrained_path=args.pretrained_path,
            window_k=args.window_k,
        )
        state = torch.load(ckpt, map_location=device, weights_only=True)
        net.load_state_dict(state)
        net.to(device)
        net.eval()

        ds = D3Dataset(
            root_dir=str(data_root),
            split="val",
            fold=fold,
            scale=args.scale,
            transform=False,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        volume_rows = []

        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device=device, dtype=torch.float32)
                gts = batch["mask"].to(device=device, dtype=torch.float32)

                logits = net(imgs)
                probs = torch.sigmoid(logits)
                pred = (probs > args.threshold).to(torch.uint8)
                pred = match_depth(pred, gts)
                gt_bin = (gts > 0.5).to(torch.uint8)

                batch_size = pred.shape[0]
                for i in range(batch_size):
                    eye_id = batch["patient_id"][i]
                    pred_np = pred[i, 0].detach().cpu().numpy().astype(np.uint8)
                    gt_np = gt_bin[i, 0].detach().cpu().numpy().astype(np.uint8)

                    tp, fp, tn, fn = confusion_counts(pred_np, gt_np)
                    dice, iou, sen, fpr = dice_iou_sen_fpr(tp, fp, tn, fn)
                    rmse = rmse_voxel(pred_np, gt_np)
                    assd, hd, hd95 = assd_hd_hd95_3d(pred_np, gt_np, spacing=spacing)
                    bf1 = boundary_f1_3d(pred_np, gt_np, tol_vox=args.tol, spacing=spacing)
                    sdice = surface_dice_3d(pred_np, gt_np, tol_vox=args.tol, spacing=spacing)

                    volume_rows.append(
                        {
                            "fold": fold,
                            "eye_id": eye_id,
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
                            "depth": int(pred_np.shape[0]),
                            "height": int(pred_np.shape[1]),
                            "width": int(pred_np.shape[2]),
                        }
                    )

        volume_rows.sort(key=lambda r: r["eye_id"])

        def fold_mean(key):
            vals = [r[key] for r in volume_rows]
            vals = np.array(vals, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            return float(vals.mean()) if vals.size else float("nan")

        fold_row = {
            "fold": fold,
            "checkpoint": str(ckpt),
            "n_volumes": len(volume_rows),
            "dice": fold_mean("dice"),
            "iou": fold_mean("iou"),
            "sen": fold_mean("sen"),
            "fpr": fold_mean("fpr"),
            "rmse": fold_mean("rmse"),
            "assd": fold_mean("assd"),
            "hd": fold_mean("hd"),
            "hd95": fold_mean("hd95"),
            "bf1_tol2": fold_mean("bf1_tol2"),
            "surf_dice_tol2": fold_mean("surf_dice_tol2"),
        }
        fold_summary_rows.append(fold_row)

        for metric in [
            "dice",
            "iou",
            "sen",
            "fpr",
            "rmse",
            "assd",
            "hd",
            "hd95",
            "bf1_tol2",
            "surf_dice_tol2",
        ]:
            fold_volume_means[metric].append(fold_row[metric])

        eye_summary_rows = []
        for row in volume_rows:
            eye_summary_rows.append(
                {
                    "fold": row["fold"],
                    "eye_id": row["eye_id"],
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "tn": row["tn"],
                    "fn": row["fn"],
                    "vol_dice_pooled": row["dice"],
                    "vol_iou_pooled": row["iou"],
                    "vol_sen_pooled": row["sen"],
                    "vol_fpr_pooled": row["fpr"],
                    "vol_rmse_mean": row["rmse"],
                    "vol_assd_mean": row["assd"],
                    "vol_hd_mean": row["hd"],
                    "vol_hd95_mean": row["hd95"],
                    "vol_bf1_mean": row["bf1_tol2"],
                    "vol_sdice_mean": row["surf_dice_tol2"],
                    "depth": row["depth"],
                    "height": row["height"],
                    "width": row["width"],
                }
            )

        write_csv(out_dir / f"fold_{fold}_per_volume.csv", volume_rows)
        write_csv(out_dir / f"fold_{fold}_per_patient.csv", eye_summary_rows)

        all_volume_rows.extend(volume_rows)
        all_eye_summary_rows.extend(eye_summary_rows)

        print(
            f"Fold {fold}: volumes={fold_row['n_volumes']} "
            f"Dice={fold_row['dice']:.6f} IoU={fold_row['iou']:.6f} "
            f"SEN={fold_row['sen']:.6f} FPR={fold_row['fpr']:.6f}"
        )

    print("\n=== 5-fold CV (per-volume macro, mean±std across folds) ===")
    for metric in [
        "dice",
        "iou",
        "sen",
        "fpr",
        "rmse",
        "assd",
        "hd",
        "hd95",
        "bf1_tol2",
        "surf_dice_tol2",
    ]:
        mu, sd = summarize_list(fold_volume_means[metric])
        print(f"{metric:14s}: {mu:.6f} ± {sd:.6f}")

    print("\n=== OOF (per-eye/per-volume) mean±std across eyes ===")
    oof_summary_rows = []
    for name, key in [
        ("dice", "vol_dice_pooled"),
        ("iou", "vol_iou_pooled"),
        ("sen", "vol_sen_pooled"),
        ("fpr", "vol_fpr_pooled"),
        ("rmse", "vol_rmse_mean"),
        ("assd", "vol_assd_mean"),
        ("hd", "vol_hd_mean"),
        ("hd95", "vol_hd95_mean"),
        ("bf1_tol2", "vol_bf1_mean"),
        ("surf_dice_tol2", "vol_sdice_mean"),
    ]:
        mu, sd = summarize_rows(all_eye_summary_rows, key)
        print(f"{name:14s}: {mu:.6f} ± {sd:.6f}")
        oof_summary_rows.append(
            {"metric": name, "mean": mu, "std": sd, "n_eyes": len(all_eye_summary_rows)}
        )

    fold_summary_rows.sort(key=lambda r: r["fold"])
    all_volume_rows.sort(key=lambda r: (r["fold"], r["eye_id"]))
    all_eye_summary_rows.sort(key=lambda r: (r["fold"], r["eye_id"]))

    write_csv(out_dir / "cv_fold_summary.csv", fold_summary_rows)
    write_csv(out_dir / "cv_per_volume_all_folds.csv", all_volume_rows)
    write_csv(out_dir / "cv_per_patient_all_folds.csv", all_eye_summary_rows)
    write_csv(
        out_dir / "cv_oof_eye_summary.csv",
        oof_summary_rows,
        fieldnames=["metric", "mean", "std", "n_eyes"],
    )

    print(f"\n[Saved] fold summary -> {out_dir / 'cv_fold_summary.csv'}")
    print(f"[Saved] per-volume (all folds) -> {out_dir / 'cv_per_volume_all_folds.csv'}")
    print(f"[Saved] per-patient (all folds) -> {out_dir / 'cv_per_patient_all_folds.csv'}")
    print(f"[Saved] eye summary -> {out_dir / 'cv_oof_eye_summary.csv'}")


if __name__ == "__main__":
    main()


"""
python predict_cv3d.py \
  --model_root elm-results/SwinUNETR3D_Apr-02-2026_1646_model \
  --model SwinUNETR3D

python predict_cv3d.py \
  --model_root elm-results/UNet2DEnc3DDec_Apr-02-2026_2021_model \
  --model UNet2DEnc3DDec

"""
