import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import math

from dataset import D3Dataset
from model import UNet2DEnc3DDec, UNet3DFrawley, UNet3D, UNet3D_Aniso  # keep if you want selector


# -----------------------------
# Dice helpers (recommended)
# -----------------------------
@torch.no_grad()
def dice_per_slice_mean_per_volume(pred_bin: torch.Tensor, gt_bin: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    """
    pred_bin, gt_bin: [B, 1, D, H, W] binary {0,1}
    Returns: [B] where each volume Dice is mean over slices (each slice equally weighted).
    """
    pred_bin = pred_bin.float()
    gt_bin = gt_bin.float()
    dims = (1, 3, 4)  # sum over channel,H,W keep D
    inter = (pred_bin * gt_bin).sum(dim=dims)                 # [B, D]
    denom = pred_bin.sum(dim=dims) + gt_bin.sum(dim=dims)     # [B, D]
    dice_bd = (2.0 * inter + smooth) / (denom + smooth)       # [B, D]
    return dice_bd.mean(dim=1)                                # [B]


@torch.no_grad()
def global_voxel_dice(pred_bin: torch.Tensor, gt_bin: torch.Tensor, smooth: float = 1e-7) -> float:
    """
    pred_bin, gt_bin: [1,1,D,H,W] binary
    Returns: scalar dice computed over all voxels (flattened).
    """
    inter = (pred_bin & gt_bin).sum().item()
    p_sum = pred_bin.sum().item()
    g_sum = gt_bin.sum().item()
    return float((2.0 * inter + smooth) / (p_sum + g_sum + smooth))


def save_volume_slices(pred_bin: np.ndarray, out_dir: str, eye_id: str):
    """
    pred_bin: (D,H,W) uint8 in {0,1}
    saves D slices as pngs 0..D-1
    """
    os.makedirs(out_dir, exist_ok=True)
    D = pred_bin.shape[0]
    for z in range(D):
        out_path = os.path.join(out_dir, f"{eye_id}-{z}.png")
        cv2.imwrite(out_path, (pred_bin[z] * 255).astype(np.uint8))


# -----------------------------
# Window inference + aggregation
# -----------------------------
@torch.no_grad()
def predict_full_volume_from_windows(
    net: torch.nn.Module,
    vol: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
    win_depth: int = 7,
    windows_batch: int = 8,
):
    """
    vol: [1, 1, 49, H, W] float
    Returns:
      prob_full: [1, 1, 49, H, W] float in [0,1] (aggregated mean prob)
      std_full : [1, 1, 49, H, W] float >=0 (aggregated per-voxel std over window probs)
      pred_bin : [1, 1, 49, H, W] uint8 {0,1}
    """
    assert vol.ndim == 5 and vol.shape[0] == 1 and vol.shape[1] == 1, f"Expected [1,1,49,H,W], got {vol.shape}"
    D = vol.shape[2]
    assert D == 49, f"Expected depth 49, got {D}"
    assert win_depth == 7, "This script assumes 7-slice windows."
    nW = D - win_depth + 1  # 43

    _, _, _, H, W = vol.shape

    prob_sum = torch.zeros((1, 1, D, H, W), device=device, dtype=torch.float32)
    prob2_sum = torch.zeros((1, 1, D, H, W), device=device, dtype=torch.float32)
    count = torch.zeros((1, 1, D, H, W), device=device, dtype=torch.float32)

    starts = list(range(nW))

    for j in range(0, nW, windows_batch):
        chunk_starts = starts[j:j + windows_batch]

        x_list = []
        for s in chunk_starts:
            xw = vol[:, :, s:s+win_depth, :, :]   # [1,1,7,H,W]
            x_list.append(xw.squeeze(0))          # [1,7,H,W] (channel kept)
        x = torch.stack(x_list, dim=0).to(device=device, dtype=torch.float32)  # [m,1,7,H,W]

        logits = net(x)
        if logits.shape != x.shape:
            raise ValueError(f"Model must output logits [m,1,7,H,W]. Got {tuple(logits.shape)} for input {tuple(x.shape)}")

        prob = torch.sigmoid(logits)  # [m,1,7,H,W]

        for k, s in enumerate(chunk_starts):
            pk = prob[k:k+1]  # [1,1,7,H,W]
            prob_sum[:, :, s:s+win_depth, :, :] += pk
            prob2_sum[:, :, s:s+win_depth, :, :] += pk * pk
            count[:, :, s:s+win_depth, :, :] += 1.0

    count_safe = torch.clamp(count, min=1.0)
    prob_full = prob_sum / count_safe
    mean2 = prob2_sum / count_safe
    var = torch.clamp(mean2 - prob_full * prob_full, min=0.0)
    std_full = torch.sqrt(var)

    pred_bin = (prob_full > threshold).to(torch.uint8)
    return prob_full, std_full, pred_bin

def _to_u8_gray(x: np.ndarray) -> np.ndarray:
    """x float in [0,1] or any range -> uint8 [0,255] with clipping."""
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)

def _grid_7x7_bgr(volume_u8: np.ndarray) -> np.ndarray:
    """
    volume_u8: (49,H,W) uint8
    returns: (7H,7W,3) BGR
    """
    assert volume_u8.ndim == 3 and volume_u8.shape[0] == 49
    D, H, W = volume_u8.shape
    canvas = np.zeros((7 * H, 7 * W, 3), dtype=np.uint8)

    for z in range(49):
        r, c = divmod(z, 7)
        tile = volume_u8[z]  # (H,W)
        tile_bgr = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        y0, y1 = r * H, (r + 1) * H
        x0, x1 = c * W, (c + 1) * W
        canvas[y0:y1, x0:x1] = tile_bgr
    return canvas

def _grid_7x7_error_overlay(orig_u8: np.ndarray, pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    """
    orig_u8: (49,H,W) uint8
    pred_bin, gt_bin: (49,H,W) uint8 {0,1}
    Overlay:
      FN (gt=1,pred=0) -> RED
      FP (gt=0,pred=1) -> GREEN
    returns: (7H,7W,3) BGR
    """
    assert orig_u8.shape[0] == 49
    D, H, W = orig_u8.shape

    canvas = np.zeros((7 * H, 7 * W, 3), dtype=np.uint8)
    for z in range(49):
        r, c = divmod(z, 7)
        base = cv2.cvtColor(orig_u8[z], cv2.COLOR_GRAY2BGR)

        p = pred_bin[z].astype(bool)
        g = gt_bin[z].astype(bool)
        fn = g & (~p)
        fp = (~g) & p

        # BGR coloring
        base[fn] = (0, 0, 255)   # red
        base[fp] = (0, 255, 0)   # green

        y0, y1 = r * H, (r + 1) * H
        x0, x1 = c * W, (c + 1) * W
        canvas[y0:y1, x0:x1] = base

    return canvas

def save_eye_montage(
    orig_vol: np.ndarray,      # (49,H,W) float or uint8
    prob_vol: np.ndarray,      # (49,H,W) float in [0,1]
    std_vol: np.ndarray,       # (49,H,W) float >=0
    pred_bin: np.ndarray,      # (49,H,W) uint8 {0,1}
    gt_bin: np.ndarray,        # (49,H,W) uint8 {0,1}
    out_dir: str,
    eye_id: str,
):
    os.makedirs(out_dir, exist_ok=True)

    # originals
    if orig_vol.dtype != np.uint8:
        # assume orig roughly [0,1]; if not, min-max normalize per-volume
        o = orig_vol.astype(np.float32)
        omin, omax = float(o.min()), float(o.max())
        if omax > omin:
            o = (o - omin) / (omax - omin)
        orig_u8 = _to_u8_gray(o)
    else:
        orig_u8 = orig_vol

    # predictions (mean prob)
    prob_u8 = _to_u8_gray(prob_vol.astype(np.float32))

    # std: normalize by max std in the volume for visualization
    s = std_vol.astype(np.float32)
    smax = float(s.max())
    if smax > 0:
        s_vis = np.clip(s / smax, 0.0, 1.0)
    else:
        s_vis = np.zeros_like(s, dtype=np.float32)
    std_u8 = _to_u8_gray(s_vis)

    grid_orig = _grid_7x7_bgr(orig_u8)
    grid_pred = _grid_7x7_bgr(prob_u8)
    grid_std  = _grid_7x7_bgr(std_u8)
    grid_err  = _grid_7x7_error_overlay(orig_u8, pred_bin, gt_bin)

    montage = np.vstack([grid_orig, grid_pred, grid_std, grid_err])

    out_path = os.path.join(out_dir, f"{eye_id}_montage.png")
    cv2.imwrite(out_path, montage)


@torch.no_grad()
def predictive_entropy_from_mean_prob(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    prob: [1,1,D,H,W] in [0,1]
    returns: same shape entropy map
    """
    p = torch.clamp(prob, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def _flatten_with_mask(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    x: (D,H,W) float
    m: (D,H,W) bool
    returns: (N,) float
    """
    return x[m].reshape(-1)


def _binary_error_mask(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    """
    pred_bin, gt_bin: (D,H,W) uint8 {0,1}
    returns: (D,H,W) bool where True indicates error (FP or FN)
    """
    return (pred_bin != gt_bin)


def _risk_coverage_curve(unc: np.ndarray, err: np.ndarray, n_points: int = 50):
    """
    unc: (N,) float uncertainty score (higher = more uncertain)
    err: (N,) {0,1} error indicator (1 = wrong)
    Returns:
      coverages: (K,) in (0,1]
      risks:     (K,) mean error among retained most-certain points
      aurc: scalar (area under risk-coverage; lower is better)
    """
    assert unc.ndim == 1 and err.ndim == 1 and unc.shape[0] == err.shape[0]
    N = unc.shape[0]
    if N == 0:
        return np.array([]), np.array([]), float("nan")

    order = np.argsort(unc)  # most certain first
    err_sorted = err[order].astype(np.float32)

    # choose K coverage levels
    K = min(n_points, N)
    idxs = np.unique(np.linspace(1, N, K, dtype=int))  # retained count
    coverages = idxs / float(N)

    risks = []
    for k in idxs:
        risks.append(float(err_sorted[:k].mean()))
    risks = np.array(risks, dtype=np.float32)

    # trapezoidal area under risk-coverage
    aurc = float(np.trapezoid(risks, coverages))
    return coverages, risks, aurc


def _decile_error_by_uncertainty(unc: np.ndarray, err: np.ndarray, n_bins: int = 10):
    """
    Bin by uncertainty quantiles; returns list of (bin_lo, bin_hi, count, error_rate).
    """
    if unc.size == 0:
        return []

    qs = np.quantile(unc, np.linspace(0, 1, n_bins + 1))
    rows = []
    for b in range(n_bins):
        lo, hi = qs[b], qs[b + 1]
        if b < n_bins - 1:
            m = (unc >= lo) & (unc < hi)
        else:
            m = (unc >= lo) & (unc <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            rows.append((float(lo), float(hi), 0, float("nan")))
        else:
            rows.append((float(lo), float(hi), cnt, float(err[m].mean())))
    return rows


def _safe_auc(y_true: np.ndarray, scores: np.ndarray, kind: str):
    """
    kind: 'roc' or 'pr'
    Returns NaN if only one class present.
    """
    y_true = y_true.astype(np.int32)
    if y_true.min() == y_true.max():
        return float("nan")

    if kind == "roc":
        return float(roc_auc_score(y_true, scores))
    elif kind == "pr":
        return float(average_precision_score(y_true, scores))
    else:
        raise ValueError(kind)
    



# -----------------------------
# Main evaluation
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth state_dict")
    parser.add_argument("--model", type=str, default="UNet2DEnc3DDec", help="Model class name selector")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--windows_batch", type=int, default=8, help="How many windows to infer per forward pass")
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./eval-window-outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Construct model
    # -----------------------------
    # For window model, you want out_channels=1 and it will output [B,1,7,H,W] for window inputs.
    if args.model == "UNet2DEnc3DDec":
        net = UNet2DEnc3DDec(in_channels=1, out_channels=1)
    elif args.model == "UNet3DFrawley":
        net = UNet3DFrawley(in_channels=1, out_channels=1)
    elif args.model == "UNet3D":
        net = UNet3D(in_channels=1, out_channels=1)
    elif args.model == "UNet3D_Aniso":
        net = UNet3D_Aniso(in_channels=1, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.to(device)
    net.eval()

    # -----------------------------
    # Dataset / loader (TEST)
    # -----------------------------
    test_img_dir = os.path.join(args.base_dir, "data/test/image/")
    test_mask_dir = os.path.join(args.base_dir, "data/test/mask/")
    test_dataset = D3Dataset(test_img_dir, test_mask_dir, scale=1, transform=False)

    # Use batch_size=1 because window aggregation is per-volume
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    eye_ids = test_dataset.eye_ids

    per_volume_dice = []
    per_volume_global = []

    uq_rows = []  # per-eye metrics
    all_err = []
    all_std = []
    all_ent = []

    aurc_std_list = []
    aurc_ent_list = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating (window-agg)")):
            imgs = batch["image"].to(device=device, dtype=torch.float32)  # [1,1,49,H,W]
            gts  = batch["mask"].to(device=device, dtype=torch.float32)   # [1,1,49,H,W]

            # Aggregate predictions from all 43 windows
            prob_full, std_full, pred_bin = predict_full_volume_from_windows(
                net=net,
                vol=imgs,
                device=device,
                threshold=args.threshold,
                win_depth=7,
                windows_batch=args.windows_batch
            )# pred_bin: [1,1,49,H,W]
            gt_bin = (gts > 0.5).to(torch.uint8)
            # --- Uncertainty maps ---
            ent_full = predictive_entropy_from_mean_prob(prob_full)  # [1,1,49,H,W]

            # Move to numpy
            prob_np = prob_full[0, 0].detach().cpu().numpy().astype(np.float32)  # (49,H,W)
            std_np  = std_full[0, 0].detach().cpu().numpy().astype(np.float32)
            ent_np  = ent_full[0, 0].detach().cpu().numpy().astype(np.float32)

            pred_np = pred_bin[0, 0].detach().cpu().numpy().astype(np.uint8)
            gt_np   = gt_bin[0, 0].detach().cpu().numpy().astype(np.uint8)

            # --- Error definition ---
            err_mask = _binary_error_mask(pred_np, gt_np)  # bool (49,H,W)

            # --- Valid region mask to avoid background dominating ---
            # Default: evaluate around the structure (union of pred/gt)
            valid_mask = ((pred_np > 0) | (gt_np > 0))

            # If you prefer "whole retinal tissue" and you have it, swap valid_mask accordingly.
            # If valid is empty (rare but possible), fall back to whole volume.
            if valid_mask.sum() == 0:
                valid_mask = np.ones_like(valid_mask, dtype=bool)

            # Flatten arrays on valid region
            err_flat = _flatten_with_mask(err_mask.astype(np.uint8), valid_mask).astype(np.uint8)  # (N,)
            std_flat = _flatten_with_mask(std_np, valid_mask)  # (N,)
            ent_flat = _flatten_with_mask(ent_np, valid_mask)  # (N,)




            # Metrics
            d_mean_slices = dice_per_slice_mean_per_volume(pred_bin, gt_bin).item()
            d_global = global_voxel_dice(pred_bin, gt_bin)

            per_volume_dice.append(d_mean_slices)
            per_volume_global.append(d_global)

            # Optional save
            if args.save_preds:
                eye_id = eye_ids[i]

                orig_np = imgs[0, 0].detach().cpu().numpy()                    # (49,H,W) float
                prob_np = prob_full[0, 0].detach().cpu().numpy()               # (49,H,W) float
                std_np  = std_full[0, 0].detach().cpu().numpy()                # (49,H,W) float
                pred_np = pred_bin[0, 0].detach().cpu().numpy().astype(np.uint8)  # (49,H,W)
                gt_np   = gt_bin[0, 0].detach().cpu().numpy().astype(np.uint8)    # (49,H,W)

                # optional: still save individual slices
                vol_np = pred_np
                save_volume_slices(vol_np, args.out_dir, eye_id)

                # new: save montage per eye
                save_eye_montage(
                    orig_vol=orig_np,
                    prob_vol=prob_np,
                    std_vol=std_np,
                    pred_bin=pred_np,
                    gt_bin=gt_np,
                    out_dir=args.out_dir,
                    eye_id=eye_id
                )

                # AUCs for "error detection" using uncertainty scores
                roc_std = _safe_auc(err_flat, std_flat, kind="roc")
                pr_std  = _safe_auc(err_flat, std_flat, kind="pr")

                roc_ent = _safe_auc(err_flat, ent_flat, kind="roc")
                pr_ent  = _safe_auc(err_flat, ent_flat, kind="pr")

                # Risk–coverage + AURC
                cov_s, risk_s, aurc_std = _risk_coverage_curve(std_flat, err_flat, n_points=50)
                cov_e, risk_e, aurc_ent = _risk_coverage_curve(ent_flat, err_flat, n_points=50)

                # Decile binning tables (optional to print)
                dec_std = _decile_error_by_uncertainty(std_flat, err_flat, n_bins=10)
                dec_ent = _decile_error_by_uncertainty(ent_flat, err_flat, n_bins=10)

                uq_rows.append({
                    "eye_id": eye_ids[i],
                    "N_valid": int(valid_mask.sum()),
                    "err_rate": float(err_flat.mean()),
                    "roc_auc_std": roc_std,
                    "pr_auc_std": pr_std,
                    "aurc_std": aurc_std,
                    "roc_auc_ent": roc_ent,
                    "pr_auc_ent": pr_ent,
                    "aurc_ent": aurc_ent,
                })

                # Aggregate across all eyes (for pooled metrics)
                all_err.append(err_flat)
                all_std.append(std_flat)
                all_ent.append(ent_flat)
                aurc_std_list.append(aurc_std)
                aurc_ent_list.append(aurc_ent)

                tqdm.write(
                    f"[{i+1}/{len(test_loader)}] "
                    f"UQ(valid N={int(valid_mask.sum())}, err={float(err_flat.mean()):.4f}) | "
                    f"ROC(std)={roc_std:.3f} PR(std)={pr_std:.3f} AURC(std)={aurc_std:.3f} | "
                    f"ROC(ent)={roc_ent:.3f} PR(ent)={pr_ent:.3f} AURC(ent)={aurc_ent:.3f}"
                )

                # If you want to print deciles for the first few eyes:
                # if i < 3:
                #     print("STD deciles:", dec_std)
                #     print("ENT deciles:", dec_ent)

            tqdm.write(f"[{i+1}/{len(test_loader)}] Dice(mean per-slice): {d_mean_slices:.6f} | Dice(global vox): {d_global:.6f}")

    if len(per_volume_dice) == 0:
        print("No volumes evaluated.")
        return
    
    all_err = np.concatenate(all_err, axis=0) if len(all_err) else np.array([], dtype=np.uint8)
    all_std = np.concatenate(all_std, axis=0) if len(all_std) else np.array([], dtype=np.float32)
    all_ent = np.concatenate(all_ent, axis=0) if len(all_ent) else np.array([], dtype=np.float32)

    if all_err.size > 0:
        pooled_error_rate = float(all_err.mean())
        print(f"Pooled error rate (on valid mask): {pooled_error_rate:.6f}")
        print(f"Total valid voxels: {all_err.shape[0]}")
        pooled_roc_std = _safe_auc(all_err, all_std, kind="roc")
        pooled_pr_std  = _safe_auc(all_err, all_std, kind="pr")
        pooled_roc_ent = _safe_auc(all_err, all_ent, kind="roc")
        pooled_pr_ent  = _safe_auc(all_err, all_ent, kind="pr")

        pooled_cov_s, pooled_risk_s, pooled_aurc_std = _risk_coverage_curve(all_std, all_err, n_points=200)
        pooled_cov_e, pooled_risk_e, pooled_aurc_ent = _risk_coverage_curve(all_ent, all_err, n_points=200)

        print("\n--- Uncertainty Quality (pooled over valid voxels) ---")
        print(f"Pooled ROC-AUC (std): {pooled_roc_std:.4f} | Pooled PR-AUC (std): {pooled_pr_std:.4f} | Pooled AURC (std): {pooled_aurc_std:.4f}")
        print(f"Pooled ROC-AUC (ent): {pooled_roc_ent:.4f} | Pooled PR-AUC (ent): {pooled_pr_ent:.4f} | Pooled AURC (ent): {pooled_aurc_ent:.4f}")

        # Also report mean±std across eyes for AURC (per-eye selective performance)
        aurc_std_arr = np.array([a for a in aurc_std_list if not (isinstance(a, float) and math.isnan(a))], dtype=np.float32)
        aurc_ent_arr = np.array([a for a in aurc_ent_list if not (isinstance(a, float) and math.isnan(a))], dtype=np.float32)

        if aurc_std_arr.size > 0:
            print(f"Per-eye AURC(std): mean {float(aurc_std_arr.mean()):.4f} ± {float(aurc_std_arr.std()):.4f}")
        if aurc_ent_arr.size > 0:
            print(f"Per-eye AURC(ent): mean {float(aurc_ent_arr.mean()):.4f} ± {float(aurc_ent_arr.std()):.4f}")

    print("\n====================")
    print(f"Volumes evaluated: {len(per_volume_dice)}")
    print(f"Mean Dice (per-volume, mean per-slice): {float(np.mean(per_volume_dice)):.6f} ± {float(np.std(per_volume_dice)):.6f}")
    print(f"Mean Dice (per-volume, global vox):     {float(np.mean(per_volume_global)):.6f} ± {float(np.std(per_volume_global)):.6f}")
    print("====================\n")


if __name__ == "__main__":
    main()

"""
python predict3Dwindow.py --base_dir ./ --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet2DEnc3DDec_win7_Jan-13-2026_1451/checkpoints/UNet2DEnc3DDec_win7_Jan-13-2026_1451_best_epoch_78.pth --model UNet2DEnc3DDec --save_preds --out_dir ./eval-window-outputs

python predict3Dwindow.py --base_dir ./ --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet2DEnc3DDec_win7_Jan-13-2026_1451/checkpoints/UNet2DEnc3DDec_win7_Jan-13-2026_1451_best_epoch_78.pth --model UNet2DEnc3DDec --save_preds --out_dir ./eval-window-outputs

"""