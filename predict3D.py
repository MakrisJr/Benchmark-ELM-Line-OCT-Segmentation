import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
import  csv

from dataset import D3Dataset
from model import CSAM_UNet2p5D, UNet3DFrawley, UNet2DEnc3DDec, UNet3D, UNet3D_Aniso, UNet2p5D_SlidingWindow

try:
    from scipy import ndimage as ndi
except ImportError:
    print("Scipy not found")
    ndi = None

EXPORT_ONLY = True
EXPORT_EYES = {"919", "945", "990"}     # example
EXPORT_SLICES = {
    "919": {24, 25, 26},               # central triplet 
    "945": {24, 25, 26},               # hard case
    "990": {24, 30, 35},               # another pattern
}

# -----------------------------
# Utility functions
# -----------------------------
def to_uint8_gray(im):
    """
    Accepts grayscale or BGR image and returns grayscale uint8.
    """
    if im is None:
        return None
    if len(im.shape) == 2:
        return im.astype(np.uint8)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def overlay_tp_fp_fn(gray_uint8, pred01, gt01, alpha=0.45):
    """
    TP=green, FP=blue, FN=red
    """
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    tp = (pred01 == 1) & (gt01 == 1)
    fp = (pred01 == 1) & (gt01 == 0)
    fn = (pred01 == 0) & (gt01 == 1)

    color = np.zeros_like(base)
    color[tp] = (0, 255, 0)    # green
    color[fp] = (255, 0, 0)    # blue
    color[fn] = (0, 0, 255)    # red

    out = cv2.addWeighted(base, 1.0, color, alpha, 0)
    return out

def overlay_contours(gray_uint8, mask01, bgr=(0,255,0), thickness=1):
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    mask255 = (mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(base, contours, -1, bgr, thickness)
    return base

def overlay_dual_contours(gray_uint8, gt01, pred01,
                          gt_bgr=(0,255,255), pred_bgr=(0,255,0),
                          thickness=1):
    """
    GT in yellow, prediction in green.
    """
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)

    gt255 = (gt01.astype(np.uint8) * 255)
    pr255 = (pred01.astype(np.uint8) * 255)

    gt_contours, _ = cv2.findContours(gt255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pr_contours, _ = cv2.findContours(pr255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(base, gt_contours, -1, gt_bgr, thickness)
    cv2.drawContours(base, pr_contours, -1, pred_bgr, thickness)
    return base

def save_paper_slices_for_volume(pred01, gt01, eye_id, image_dir, paper_dir, selected_slices):
    """
    pred01, gt01: (D,H,W) uint8 {0,1}
    image_dir: directory with original 2D test images named eyeid-slice.png
    Saves raw image + GT contour + Pred contour + combined contour + TP/FP/FN.
    """
    os.makedirs(paper_dir, exist_ok=True)

    D, H, W = pred01.shape
    for z in sorted(selected_slices):
        if z < 0 or z >= D:
            print(f"[WARN] eye {eye_id}: requested slice {z} out of range 0..{D-1}")
            continue

        img_path = os.path.join(image_dir, f"{eye_id}-{z}.png")
        im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"[WARN] missing raw image for paper export: {img_path}")
            gray = np.zeros((H, W), dtype=np.uint8)
        else:
            gray = to_uint8_gray(im)
            if gray.shape != (H, W):
                gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_LINEAR)

        pred_slice = pred01[z].astype(np.uint8)
        gt_slice   = gt01[z].astype(np.uint8)

        gt_overlay   = overlay_contours(gray, gt_slice,   bgr=(0,255,255), thickness=1)
        pr_overlay   = overlay_contours(gray, pred_slice, bgr=(0,255,0),   thickness=1)
        both_overlay = overlay_dual_contours(gray, gt_slice, pred_slice,
                                             gt_bgr=(0,255,255), pred_bgr=(0,255,0), thickness=1)
        err_overlay  = overlay_tp_fp_fn(gray, pred_slice, gt_slice, alpha=0.50)

        prefix = os.path.join(paper_dir, f"{eye_id}-{z}")
        cv2.imwrite(prefix + "_A_img.png", gray)
        cv2.imwrite(prefix + "_B_gtContour.png", gt_overlay)
        cv2.imwrite(prefix + "_C_predContour.png", pr_overlay)
        cv2.imwrite(prefix + "_D_gtPredContours.png", both_overlay)
        cv2.imwrite(prefix + "_E_errTPFPFN.png", err_overlay)

def safe_div(n, d, eps=1e-12):
    return float(n) / float(d + eps)

def write_per_patient_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["eye_id"]
    with open(path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def confusion_counts_np(pred, gt):
    # pred, gt are uint8 {0,1}
    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    tn = int(np.logical_and(pred == 0, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    return tp, fp, tn, fn

def dice_iou_sen_fpr(tp, fp, tn, fn, eps=1e-12):
    dice = safe_div(2 * tp, 2 * tp + fp + fn, eps)
    iou  = safe_div(tp, tp + fp + fn, eps)
    sen  = safe_div(tp, tp + fn, eps)
    fpr  = safe_div(fp, fp + tn, eps)
    return dice, iou, sen, fpr

def rmse_voxel(pred, gt):
    diff = pred.astype(np.float32) - gt.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))

def summarize_list(xs):
    xs = np.array(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(xs.std(ddof=1)) if xs.size > 1 else 0.0

def match_depth(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Ensure pred and gt have same depth by center-cropping or padding pred.
    pred, gt: [B,1,D,H,W]
    """
    dp = pred.shape[2]
    dg = gt.shape[2]
    if dp == dg:
        return pred

    if dp > dg:
        # center crop
        start = (dp - dg) // 2
        return pred[:, :, start:start+dg, :, :]
    else:
        # pad equally on both sides
        pad_total = dg - dp
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        # pad format for 5D is (W_left, W_right, H_left, H_right, D_left, D_right)
        return torch.nn.functional.pad(pred, (0, 0, 0, 0, pad_left, pad_right),
                                       mode="constant", value=0)

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

def dice_from_binary(pred: torch.Tensor, gt: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
    """
    pred, gt: binary tensors with shape [B, 1, D, H, W] (0/1)
    returns: dice per item in batch shape [B]
    """
    pred = pred.float()
    gt = gt.float()
    dims = (1, 2, 3, 4)
    inter = (pred * gt).sum(dim=dims)
    p = pred.sum(dim=dims)
    g = gt.sum(dim=dims)
    return (2.0 * inter + smooth) / (p + g + smooth)

# -----------------------------
# 3D surface/boundary metrics
# -----------------------------
def _require_scipy():
    if ndi is None:
        raise ImportError(
            "SciPy is required for 3D ASSD/HD/HD95 and 3D boundary metrics. "
            "Install with: pip install scipy"
        )

def surface_voxels_3d(mask01: np.ndarray) -> np.ndarray:
    """
    Returns a boolean array of surface voxels for a 3D binary mask.
    surface = mask XOR erode(mask)
    """
    _require_scipy()
    mask = mask01.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)  # 6-neighbourhood
    er = ndi.binary_erosion(mask, structure=structure, iterations=1, border_value=0)
    surf = np.logical_and(mask, np.logical_not(er))
    return surf

def assd_hd_hd95_3d(pred01: np.ndarray, gt01: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    """
    ASSD, HD, HD95 computed on 3D surfaces using Euclidean distance transform.
    spacing: (z,y,x) voxel spacing in your physical unit (mm/um/etc). If you keep 1, it's in voxels.
    """
    _require_scipy()
    p = pred01.astype(bool)
    g = gt01.astype(bool)

    ps = surface_voxels_3d(p)
    gs = surface_voxels_3d(g)

    if ps.sum() == 0 and gs.sum() == 0:
        return 0.0, 0.0, 0.0
    if ps.sum() == 0 or gs.sum() == 0:
        return float("inf"), float("inf"), float("inf")

    # distance to nearest surface voxel: compute EDT on inverted surface
    dt_g = ndi.distance_transform_edt(~gs, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~ps, sampling=spacing)

    d_p_to_g = dt_g[ps].astype(np.float64)
    d_g_to_p = dt_p[gs].astype(np.float64)

    all_d = np.concatenate([d_p_to_g, d_g_to_p], axis=0)
    assd = float(all_d.mean())
    hd = float(all_d.max())
    hd95 = float(np.percentile(all_d, 95))
    return assd, hd, hd95

def boundary_f1_3d(pred01: np.ndarray, gt01: np.ndarray, tol_vox=2, spacing=(1.0, 1.0, 1.0)):
    """
    3D Boundary F1 with tolerance using distance transforms on surfaces.
    tol_vox is interpreted in *voxels* unless you also change spacing;
    actual tolerance distance is tol_vox * spacing if spacing != 1 isotropically.
    """
    _require_scipy()
    p = pred01.astype(bool)
    g = gt01.astype(bool)

    ps = surface_voxels_3d(p)
    gs = surface_voxels_3d(g)

    if ps.sum() == 0 and gs.sum() == 0:
        return 1.0
    if ps.sum() == 0 or gs.sum() == 0:
        return 0.0

    # threshold in physical units: tol_vox * 1 voxel in each direction isn't fully well-defined if anisotropic.
    # We interpret tol_vox as a *distance* in the same unit as EDT output when spacing is provided.
    tol_dist = float(tol_vox)

    dt_g = ndi.distance_transform_edt(~gs, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~ps, sampling=spacing)

    # precision: predicted surface points within tol of gt surface
    prec = safe_div((dt_g[ps] <= tol_dist).sum(), ps.sum())
    # recall: gt surface points within tol of predicted surface
    rec  = safe_div((dt_p[gs] <= tol_dist).sum(), gs.sum())

    return float(safe_div(2 * prec * rec, (prec + rec)))

def surface_dice_3d(pred01: np.ndarray, gt01: np.ndarray, tol_vox=2, spacing=(1.0, 1.0, 1.0)):
    """
    "Surface Dice" analogue: symmetric surface overlap within tolerance.
    """
    _require_scipy()
    p = pred01.astype(bool)
    g = gt01.astype(bool)

    ps = surface_voxels_3d(p)
    gs = surface_voxels_3d(g)

    if ps.sum() == 0 and gs.sum() == 0:
        return 1.0
    if ps.sum() == 0 or gs.sum() == 0:
        return 0.0

    tol_dist = float(tol_vox)

    dt_g = ndi.distance_transform_edt(~gs, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~ps, sampling=spacing)

    matched_p = (dt_g[ps] <= tol_dist).sum()
    matched_g = (dt_p[gs] <= tol_dist).sum()

    return float(safe_div(matched_p + matched_g, ps.sum() + gs.sum()))

# -----------------------------
# Main evaluation
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="UNet3DFrawley")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./eval-3d-outputs")

    # To keep metric names identical to 2D: bf1_tol2 / surf_dice_tol2, we fix tol=2 by default
    parser.add_argument("--tol", type=int, default=2)
    parser.add_argument("--spacing_z", type=float, default=1.0)
    parser.add_argument("--spacing_y", type=float, default=1.0)
    parser.add_argument("--spacing_x", type=float, default=1.0)
    args = parser.parse_args()

    spacing = (args.spacing_z, args.spacing_y, args.spacing_x)
    paper_dir = os.path.join(os.path.dirname(args.out_dir), "paper_figures")
    os.makedirs(paper_dir, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model selection
    if args.model == "UNet3DFrawley":
        net = UNet3DFrawley(in_channels=1, out_channels=1)
    elif args.model == "UNet2DEnc3DDec":
        net = UNet2DEnc3DDec(in_channels=1, out_channels=1)
    elif args.model == "UNet3D":
        net = UNet3D(in_channels=1, out_channels=1)
    elif args.model == "UNet3D_Aniso":
        net = UNet3D_Aniso(in_channels=1, out_channels=1)
    elif args.model == "CSAM_UNet2p5D":
        net = CSAM_UNet2p5D(in_channels=1, out_channels=1, num_layers=5, base_num=32, semantic=True, positional=True, slice_att=True)
    elif args.model == "UNet2p5D_SlidingWindow":
        net = UNet2p5D_SlidingWindow(k=7, out_channels=1, num_layers=3, base_num=32, pad_mode="replicate")
    else:
        raise ValueError(f"Unknown model: {args.model}")

    net.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    net.to(device)
    net.eval()

    # Dataset / loader
    test_img_dir = os.path.join(args.base_dir, "data/test/image/")
    test_mask_dir = os.path.join(args.base_dir, "data/test/mask/")
    test_dataset = D3Dataset(test_img_dir, test_mask_dir, scale=1, transform=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # --- Metric storage (names + order matching 2D) ---
    metrics = {
        "dice": [],
        "iou": [],
        "sen": [],
        "fpr": [],
        "rmse": [],
        "assd": [],
        "hd": [],
        "hd95": [],
        "bf1_tol2": [],
        "surf_dice_tol2": [],
    }

    # --- Global pooled confusion (voxel pooled, but we print "pixel-pooled") ---
    g_tp = g_fp = g_tn = g_fn = 0

    # --- Per-patient aggregation (volume summaries) ---
    # Here "patient" == eye_id from dataset ordering
    patient_metrics = {}
    for eid in test_dataset.eye_ids:
        patient_metrics[eid] = {
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "hd95": [], "assd": [], "bf1_tol2": [], "surf_dice_tol2": [],
            "n_vols": 0
        }

    eye_ids = test_dataset.eye_ids

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            imgs = batch["image"].to(device=device, dtype=torch.float32)  # [B,1,D,H,W]
            gts  = batch["mask"].to(device=device, dtype=torch.float32)   # [B,1,D,H,W]

            logits = net(imgs)
            probs = torch.sigmoid(logits)
            pred_bin_t = (probs > args.threshold).to(torch.uint8)

            pred_bin_t = match_depth(pred_bin_t, gts)
            gt_bin_t = (gts > 0.5).to(torch.uint8)

            B = pred_bin_t.shape[0]
            for b in range(B):
                eye_id = eye_ids[i * args.batch_size + b]

                pred01 = pred_bin_t[b, 0].detach().cpu().numpy().astype(np.uint8)  # (D,H,W)
                gt01   = gt_bin_t[b, 0].detach().cpu().numpy().astype(np.uint8)    # (D,H,W)

                # confusion-based
                tp, fp, tn, fn = confusion_counts_np(pred01, gt01)
                dice, iou, sen, fpr = dice_iou_sen_fpr(tp, fp, tn, fn)

                rmse = rmse_voxel(pred01, gt01)
                assd, hd, hd95 = assd_hd_hd95_3d(pred01, gt01, spacing=spacing)
                bf1 = boundary_f1_3d(pred01, gt01, tol_vox=args.tol, spacing=spacing)
                sdice = surface_dice_3d(pred01, gt01, tol_vox=args.tol, spacing=spacing)

                # store per-volume metrics
                metrics["dice"].append(dice)
                metrics["iou"].append(iou)
                metrics["sen"].append(sen)
                metrics["fpr"].append(fpr)
                metrics["rmse"].append(rmse)
                metrics["assd"].append(assd)
                metrics["hd"].append(hd)
                metrics["hd95"].append(hd95)
                metrics["bf1_tol2"].append(bf1)
                metrics["surf_dice_tol2"].append(sdice)

                # global pooled confusion
                g_tp += tp; g_fp += fp; g_tn += tn; g_fn += fn

                # per-patient aggregation (pooled confusion + mean geo metrics)
                pm = patient_metrics[eye_id]
                pm["tp"] += tp; pm["fp"] += fp; pm["tn"] += tn; pm["fn"] += fn
                pm["hd95"].append(hd95)
                pm["assd"].append(assd)
                pm["bf1_tol2"].append(bf1)
                pm["surf_dice_tol2"].append(sdice)
                pm["n_vols"] += 1

                if EXPORT_ONLY:
                    if eye_id in EXPORT_EYES:
                        selected = EXPORT_SLICES.get(eye_id, set())
                        if len(selected) > 0:
                            save_paper_slices_for_volume(
                                pred01=pred01,
                                gt01=gt01,
                                eye_id=eye_id,
                                image_dir=test_img_dir,
                                paper_dir=paper_dir,
                                selected_slices=selected
                            )

                if args.save_preds:
                    save_volume_slices(pred01, args.out_dir, eye_id)

    # -----------------------------
    # Report (exact style/order as 2D)
    # -----------------------------
    print("\n====================")
    print(f"Volumes evaluated: {len(metrics['dice'])}")

    # Per-volume mean ± std in the SAME order as 2D output
    order = ["dice","iou","sen","fpr","rmse","assd","hd","hd95","bf1_tol2","surf_dice_tol2"]
    for k in order:
        mu, sd = summarize_list(metrics[k])
        # match your spacing/alignment loosely
        print(f"{k:14s}: {mu:.6f} ± {sd:.6f}")

    # Global pooled
    g_dice, g_iou, g_sen, g_fpr = dice_iou_sen_fpr(g_tp, g_fp, g_tn, g_fn)
    print("\n--- Global (pixel-pooled) ---")
    print(f"Global Dice: {g_dice:.6f}")
    print(f"Global IoU : {g_iou:.6f}")
    print(f"Global SEN : {g_sen:.6f}")
    print(f"Global FPR : {g_fpr:.6f}")

    # Per-patient (volume) summaries (same labels as your 2D block)
    print("\n--- Per-patient (volume) summaries ---")
    vol_dice = []
    vol_iou = []
    vol_sen = []
    vol_fpr = []
    vol_hd95 = []
    vol_assd = []
    vol_bf1 = []
    vol_sdice = []

    for pid, pm in patient_metrics.items():
        if pm["n_vols"] == 0:
            continue

        pdice, piou, psen, pfpr = dice_iou_sen_fpr(pm["tp"], pm["fp"], pm["tn"], pm["fn"])
        m_hd95, _ = summarize_list(pm["hd95"])
        m_assd, _ = summarize_list(pm["assd"])
        m_bf1, _ = summarize_list(pm["bf1_tol2"])
        m_sdice, _ = summarize_list(pm["surf_dice_tol2"])

        vol_dice.append(pdice)
        vol_iou.append(piou)
        vol_sen.append(psen)
        vol_fpr.append(pfpr)
        vol_hd95.append(m_hd95)
        vol_assd.append(m_assd)
        vol_bf1.append(m_bf1)
        vol_sdice.append(m_sdice)

    def print_vol(label, arr):
        mu, sd = summarize_list(arr)
        print(f"{label:20s}: {mu:.6f} ± {sd:.6f}")

    print_vol("Vol Dice (pooled)", vol_dice)
    print_vol("Vol IoU (pooled)",  vol_iou)
    print_vol("Vol SEN (pooled)",  vol_sen)
    print_vol("Vol FPR (pooled)",  vol_fpr)
    print_vol("Vol HD95 (mean)",   vol_hd95)
    print_vol("Vol ASSD (mean)",   vol_assd)
    print_vol("Vol BF1@2px (mean)",vol_bf1)
    print_vol("Vol SurfDice@2px (mean)", vol_sdice)

    print("====================\n")


    per_patient_csv_path = os.path.join(os.path.dirname(args.out_dir), "per_patient_metrics.csv")
    per_patient_rows = []
    for pid, pm in patient_metrics.items():
        if pm["n_vols"] == 0:
            continue

        pdice, piou, psen, pfpr = dice_iou_sen_fpr(pm["tp"], pm["fp"], pm["tn"], pm["fn"])
        m_hd95, _ = summarize_list(pm["hd95"])
        m_assd, _ = summarize_list(pm["assd"])
        m_bf1, _ = summarize_list(pm["bf1_tol2"])
        m_sdice, _ = summarize_list(pm["surf_dice_tol2"])

        per_patient_rows.append({
            "eye_id": pid,
            "tp": pm["tp"], "fp": pm["fp"], "tn": pm["tn"], "fn": pm["fn"],
            "vol_dice_pooled": pdice,
            "vol_iou_pooled": piou,
            "vol_sen_pooled": psen,
            "vol_fpr_pooled": pfpr,
            "vol_hd95_mean": m_hd95,
            "vol_assd_mean": m_assd,
            "vol_bf1_mean": m_bf1,
            "vol_sdice_mean": m_sdice,
            "n_vols": pm["n_vols"]
        })

    per_patient_rows.sort(key=lambda x: x["eye_id"])

    write_per_patient_csv(per_patient_csv_path, per_patient_rows)
    print(f"[SAVED] per-patient metrics CSV -> {per_patient_csv_path}")


if __name__ == "__main__":
    main()

"""
python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet3DFrawley_Nov-25-2025_1508_model/checkpoints/UNet3DFrawley_Nov-25-2025_1508_model_best_epoch_66.pth --model UNet3DFrawley --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./eval-3d-outputs

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet2DEnc3DDec_Jan-05-2026_1240_model/checkpoints/UNet2DEnc3DDec_Jan-05-2026_1240_model_best_epoch_85.pth --model UNet2DEnc3DDec --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./eval-3d-outputs

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet3D_Nov-21-2025_1259_model/checkpoints/UNet3D_Nov-21-2025_1259_model_best_epoch_55.pth --model UNet3D --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./eval-3d-outputs

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet3D_Aniso_Nov-25-2025_1449_model/checkpoints/INTERRUPTED_UNet3D_Aniso_Nov-25-2025_1449_model.model --model UNet3D_Aniso --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./eval-3d-outputs

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet2DEnc3DDec_Jan-08-2026_1025_model/checkpoints/UNet2DEnc3DDec_Jan-08-2026_1025_model_best_epoch_77.pth --model UNet2DEnc3DDec --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./eval-3d-outputs

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet2DEnc3DDec_Jan-08-2026_1730_model/checkpoints/UNet2DEnc3DDec_Jan-08-2026_1730_model_best_epoch_69.pth --model UNet2DEnc3DDec --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./eval-3d-outputs

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet3DFrawley_Feb-20-2026_1329_model/checkpoints/UNet3DFrawley_Feb-20-2026_1329_model_best_epoch_94.pth --model UNet3DFrawley --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./elm-results/UNet3DFrawley_Feb-20-2026_1329_model/out --tol 2

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet2DEnc3DDec_Feb-20-2026_2001_model/checkpoints/UNet2DEnc3DDec_Feb-20-2026_2001_model_best_epoch_98.pth --model UNet2DEnc3DDec --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./elm-results/UNet2DEnc3DDec_Feb-20-2026_2001_model/out --tol 2

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/CSAM_UNet2p5D_Feb-25-2026_1104_model/checkpoints/CSAM_UNet2p5D_Feb-25-2026_1104_model_best_epoch_97.pth --model CSAM_UNet2p5D --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./elm-results/CSAM_UNet2p5D_Feb-25-2026_1104_model/out --tol 2

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/CSAM_UNet2p5D_Feb-25-2026_1113_model/checkpoints/CSAM_UNet2p5D_Feb-25-2026_1113_model_best_epoch_95.pth --model CSAM_UNet2p5D --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./elm-results/CSAM_UNet2p5D_Feb-25-2026_1113_model/out --tol 2

python predict3D.py --base_dir . --checkpoint /home/s2036401/Benchmark-ELM-Line-OCT-Dataset/elm-results/UNet2p5D_SlidingWindow_Feb-25-2026_1352_model/checkpoints/UNet2p5D_SlidingWindow_Feb-25-2026_1352_model_best_epoch_90.pth --model UNet2p5D_SlidingWindow --batch_size 1 --num_workers 1 --threshold 0.5 --save_preds --out_dir ./elm-results/UNet2p5D_SlidingWindow_Feb-25-2026_1352_model/out --tol 2
"""