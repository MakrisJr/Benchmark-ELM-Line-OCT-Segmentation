import argparse
import logging
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from efficientunet import *
from elm.dice_loss import dice_coeff
from elm.model import SwinEncoderUNet2D, U_Net,AttU_Net, LinkNetImprove, U2NETP,R2U_Net,DeepLabv3_plus,FCN,SegNet
import re
from collections import defaultdict
import csv

EXPORT_ONLY = True
EXPORT_EYES = {"919", "945", "990"}     # example
EXPORT_SLICES = {
    "919": {24, 25, 26},               # central triplet
    "945": {24, 25, 26},               # hard case
    "990": {24, 30, 35},               # another pattern
}

#model = get_efficientunet_b3(n_classes=1, concat_input=True, pretrained=True)
#model = AttU_Net(n_channels=3, n_classes=1)
# model = U_Net(n_channels=3, n_classes=1)
#model = LinkNetImprove(n_channels=3, n_classes=1)
#model = U2NETP(n_channels=3,n_classes=1)
# model = R2U_Net(n_channels=3, n_classes=1,t=2)
#model = DeepLabv3_plus(n_channels=3, n_classes=1, os=16, pretrained=True, _print=True)
#model = FCN(n_channels=3, n_classes=1)
# model = SegNet(n_channels=3, n_classes=1)
model = SwinEncoderUNet2D(
        n_channels=3,
        n_classes=1,
        backbone="swin_tiny_patch4_window7_224",
        pretrained=True,
    )


if torch.cuda.is_available():
    model.cuda()

# MODEL_NAME = 'R2U_Net_Feb-16-2026_1743_model'
MODEL_NAME = 'SegNet_Feb-16-2026_1745_model'
MODEL_NAME = 'SwinEncoderUNet2D_Mar-16-2026_1515_model'
BEST_EPOCH = 20
exp_dir = './elm-results/' + MODEL_NAME
model.load_state_dict(torch.load(os.path.join(exp_dir, 'checkpoints', MODEL_NAME+'_best_epoch_'+str(BEST_EPOCH)+'.pth')))

per_patient_csv_path = os.path.join(exp_dir, f"per_patient_metrics.csv")

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

image_dir = './data_no_anomalies/test/image/'
mask_dir  = './data_no_anomalies/test/mask/'

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

# create output directory
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

if not os.path.exists(os.path.join(exp_dir, 'test_outputs')):
    os.makedirs(os.path.join(exp_dir, 'test_outputs'))
# ---- Dice accumulators (global / total Dice) ----
smooth = 1.0

# Global pixel accumulators (for global Dice/IoU/SEN/FPR)
g_tp = g_fp = g_tn = g_fn = 0

empty_gt = 0
empty_pred = 0

# Per-image metric lists
metrics = {
    "dice": [],
    "iou": [],
    "sen": [],
    "fpr": [],
    "rmse": [],
    "hd": [],
    "hd95": [],
    "assd": [],
    "bf1_tol2": [],
    "surf_dice_tol2": [],
}

# Per-patient (volume) aggregation: store per-slice metrics + confusion counts
patient_metrics = defaultdict(lambda: {
    "dice": [], "iou": [], "sen": [], "fpr": [], "rmse": [],
    "hd": [], "hd95": [], "assd": [], "bf1_tol2": [], "surf_dice_tol2": [],
    "tp": 0, "fp": 0, "tn": 0, "fn": 0,
    "n_slices": 0, "slice_idx": [],
})

def write_per_patient_csv(path: str, rows: list):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["eye_id"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def safe_div(n, d, eps=1e-12):
    return float(n) / float(d + eps)

def confusion_counts(pred, gt):
    # pred, gt are uint8 {0,1}
    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    tn = int(np.logical_and(pred == 0, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    return tp, fp, tn, fn

def dice_iou_sen_fpr(tp, fp, tn, fn, eps=1e-12):
    dice = safe_div(2 * tp, 2 * tp + fp + fn, eps)
    iou  = safe_div(tp, tp + fp + fn, eps)
    sen  = safe_div(tp, tp + fn, eps)                    # recall
    fpr  = safe_div(fp, fp + tn, eps)
    return dice, iou, sen, fpr

def rmse_pixel(pred, gt):
    # Singh et al compute RMSE on valid pixels; here for binary, we compute pixelwise MSE over full image.
    diff = pred.astype(np.float32) - gt.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))

def extract_boundary(mask):
    # 8-connected boundary: mask - eroded(mask)
    mask255 = (mask.astype(np.uint8) * 255)
    k = np.ones((3,3), np.uint8)
    er = cv2.erode(mask255, k, iterations=1)
    b = cv2.subtract(mask255, er)
    return (b > 0).astype(np.uint8)

def boundary_f1(pred, gt, tol=2):
    """
    Boundary F1 with pixel tolerance 'tol' (in pixels).
    We dilate boundaries by tol and compute precision/recall on boundary pixels.
    """
    pb = extract_boundary(pred)
    gb = extract_boundary(gt)

    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
    pb_d = cv2.dilate(pb, k, iterations=1)
    gb_d = cv2.dilate(gb, k, iterations=1)

    # precision: predicted boundary pixels that hit dilated GT boundary
    prec = safe_div(np.logical_and(pb == 1, gb_d == 1).sum(), pb.sum())
    # recall: GT boundary pixels that hit dilated predicted boundary
    rec  = safe_div(np.logical_and(gb == 1, pb_d == 1).sum(), gb.sum())
    f1 = safe_div(2 * prec * rec, (prec + rec))
    return float(f1)

def surface_dice(pred, gt, tol=2):
    """
    "Surface Dice" approximation: Dice computed on boundary pixels, allowing tolerance via dilation.
    """
    pb = extract_boundary(pred)
    gb = extract_boundary(gt)

    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
    pb_d = cv2.dilate(pb, k, iterations=1)
    gb_d = cv2.dilate(gb, k, iterations=1)

    # boundary matches with tolerance
    inter = (np.logical_and(pb == 1, gb_d == 1).sum() +
             np.logical_and(gb == 1, pb_d == 1).sum())
    denom = (pb.sum() + gb.sum())
    return safe_div(inter, denom)

def distance_transform(mask):
    """
    cv2.distanceTransform expects 0 background, non-zero foreground.
    returns float32 distances for each pixel to nearest zero pixel.
    We'll use it to get distances from boundary sets efficiently.
    """
    return cv2.distanceTransform(mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)

def assd_and_hausdorff(pred, gt):
    """
    Compute ASSD, HD, HD95 using distance transforms of boundary maps.
    No SciPy required.
    """
    pb = extract_boundary(pred)
    gb = extract_boundary(gt)

    # Edge cases
    if pb.sum() == 0 and gb.sum() == 0:
        return 0.0, 0.0, 0.0
    if pb.sum() == 0 or gb.sum() == 0:
        # undefined/infinite distances; return large numbers
        return float("inf"), float("inf"), float("inf")

    # Distance to nearest boundary pixel:
    # Make boundary pixels = 0, everything else = 1, then distanceTransform gives distance to 0.
    dt_g = distance_transform(1 - gb)   # distances to GT boundary
    dt_p = distance_transform(1 - pb)   # distances to Pred boundary

    # Directed distances: boundary pixels sample their distances to other boundary
    d_p_to_g = dt_g[pb == 1].astype(np.float64)
    d_g_to_p = dt_p[gb == 1].astype(np.float64)

    all_d = np.concatenate([d_p_to_g, d_g_to_p], axis=0)
    assd = float(all_d.mean())
    hd = float(all_d.max())
    hd95 = float(np.percentile(all_d, 95))
    return assd, hd, hd95

def patient_id_from_filename(fname):
    """
    Expected pattern: '841-0.png' -> patient '841'
    Returns None if doesn't match.
    """
    m = re.match(r"^([0-9A-Za-z]+)-([0-9]+)\.(png|jpg|jpeg)$", fname)
    if not m:
        return None
    return m.group(1)

def to_uint8_gray(im_bgr):
    # im_bgr is cv2 BGR uint8
    g = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    return g

def overlay_mask_on_gray(gray_uint8, mask01, alpha=0.35):
    """
    gray_uint8: (H,W) uint8
    mask01: (H,W) {0,1}
    Returns BGR overlay image.
    """
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(base)
    color[:, :, 1] = (mask01 * 255).astype(np.uint8)  # green
    out = cv2.addWeighted(base, 1.0, color, alpha, 0)
    return out

def overlay_tp_fp_fn(gray_uint8, pred01, gt01, alpha=0.45):
    """
    TP=green, FP=blue, FN=red (common in papers).
    """
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    tp = (pred01 == 1) & (gt01 == 1)
    fp = (pred01 == 1) & (gt01 == 0)
    fn = (pred01 == 0) & (gt01 == 1)

    color = np.zeros_like(base)
    # BGR
    color[tp] = (0, 255, 0)    # green
    color[fp] = (255, 0, 0)    # blue
    color[fn] = (0, 0, 255)    # red

    out = cv2.addWeighted(base, 1.0, color, alpha, 0)
    return out

def overlay_contours(gray_uint8, mask01, bgr=(0,255,0), thickness=1):
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    mask255 = (mask01 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(base, contours, -1, bgr, thickness)
    return base

def patient_and_slice_from_filename(fname):
    m = re.match(r"^([0-9A-Za-z]+)-([0-9]+)\.(png|jpg|jpeg)$", fname)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))

model.eval()  # important for inference
with torch.no_grad():
    for image_name in image_filenames:
        print(image_name)

        # ---- Read + preprocess image ----
        im = cv2.imread(os.path.join(image_dir, image_name))
        if im is None:
            print(f"WARNING: could not read image: {image_name}")
            continue

        h, w, c = im.shape

        im_resized = cv2.resize(im, (256, 256))
        im_resized = im_resized / 255.0
        im_resized = np.expand_dims(im_resized, 0)
        im_resized = np.transpose(im_resized, (0, 3, 1, 2))
        im_t = torch.from_numpy(im_resized).float()
        if torch.cuda.is_available():
            im_t = im_t.cuda()

        # normalize (ImageNet)
        im_t[:, 0, :, :] = (im_t[:, 0, :, :] - 0.485) / 0.229
        im_t[:, 1, :, :] = (im_t[:, 1, :, :] - 0.456) / 0.224
        im_t[:, 2, :, :] = (im_t[:, 2, :, :] - 0.406) / 0.225

        # ---- Predict ----
        out = model(im_t)
        out = torch.sigmoid(out)
        out = (out > 0.5).to(torch.uint8)  # (1,1,256,256)

        pred = out.squeeze().cpu().numpy().astype(np.uint8)  # (256,256), {0,1}

        # ---- Save prediction (as before) ----
        pred_to_save = (pred * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(MODEL_NAME, 'test_outputs', image_name), pred_to_save)

        # ---- Read + preprocess GT mask ----
        gt_path = os.path.join(mask_dir, image_name)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"WARNING: missing GT mask for {image_name} at {gt_path}. Skipping Dice for this image.")
            continue

        gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
        gt = (gt > 127).astype(np.uint8)  # binarize

        # ---- Confusion counts ----
        tp, fp, tn, fn = confusion_counts(pred, gt)
        dice, iou, sen, fpr = dice_iou_sen_fpr(tp, fp, tn, fn)

        rmse = rmse_pixel(pred, gt)
        assd, hd, hd95 = assd_and_hausdorff(pred, gt)
        bf1 = boundary_f1(pred, gt, tol=2)
        sdice = surface_dice(pred, gt, tol=2)

        # Store per-image
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

        print(
            f"  Dice={dice:.6f} IoU={iou:.6f} SEN={sen:.6f} FPR={fpr:.6f} "
            f"RMSE={rmse:.6f} ASSD={assd:.3f} HD={hd:.3f} HD95={hd95:.3f} "
            f"BF1@2px={bf1:.6f} SurfDice@2px={sdice:.6f}"
        )

        # Global accumulators
        g_tp += tp; g_fp += fp; g_tn += tn; g_fn += fn

        empty_gt += int(gt.sum() == 0)
        empty_pred += int(pred.sum() == 0)

        # Patient-level aggregation
        pid, slice_idx = patient_and_slice_from_filename(image_name)
        if pid is not None:
            pm = patient_metrics[pid]
            pm["dice"].append(dice)
            pm["iou"].append(iou)
            pm["sen"].append(sen)
            pm["fpr"].append(fpr)
            pm["rmse"].append(rmse)
            pm["assd"].append(assd)
            pm["hd"].append(hd)
            pm["hd95"].append(hd95)
            pm["bf1_tol2"].append(bf1)
            pm["surf_dice_tol2"].append(sdice)
            pm["tp"] += tp; pm["fp"] += fp; pm["tn"] += tn; pm["fn"] += fn
            pm["n_slices"] += 1
            pm["slice_idx"].append(slice_idx)

        if EXPORT_ONLY:
            if pid not in EXPORT_EYES or slice_idx not in EXPORT_SLICES.get(pid, set()):
                continue
        paper_dir = os.path.join(exp_dir, "paper_figures")
        os.makedirs(paper_dir, exist_ok=True)

        # use the resized version for overlays (since pred/gt are 256x256)
        # but use the same grayscale base used for prediction (im_resized before normalization)
        im256 = cv2.resize(im, (256, 256))
        gray256 = to_uint8_gray(im256)

        gt_overlay  = overlay_contours(gray256, gt,  bgr=(0,255,255), thickness=1)  # yellow GT
        pr_overlay  = overlay_contours(gray256, pred, bgr=(0,255,0), thickness=1)   # green Pred
        err_overlay = overlay_tp_fp_fn(gray256, pred, gt, alpha=0.5)

        # Save individual panels (lossless)
        base_path = os.path.join(paper_dir, os.path.splitext(image_name)[0])
        cv2.imwrite(base_path + "_A_img.png", gray256)
        cv2.imwrite(base_path + "_B_gtContour.png", gt_overlay)
        cv2.imwrite(base_path + "_C_predContour.png", pr_overlay)
        cv2.imwrite(base_path + "_D_errTPFPFN.png", err_overlay)


# ---- Report ----
def summarize_list(xs):
    xs = np.array(xs, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(xs.std(ddof=1)) if xs.size > 1 else 0.0

print("\n====================")
n = len(metrics["dice"])
print(f"Images evaluated: {n}")
# sanity check
# print(f"Empty GT masks: {empty_gt}")
# print(f"Empty Pred masks: {empty_pred}")

if n == 0:
    print("No metrics computed (no GT masks found or all skipped).")
else:
    # Per-image mean ± std
    for k in ["dice","iou","sen","fpr","rmse","assd","hd","hd95","bf1_tol2","surf_dice_tol2"]:
        mu, sd = summarize_list(metrics[k])
        print(f"{k:14s}: {mu:.6f} ± {sd:.6f}")

    # Global (pixel-pooled) metrics
    # print(f"Global FP: {g_fp}, TN: {g_tn}")
    g_dice, g_iou, g_sen, g_fpr = dice_iou_sen_fpr(g_tp, g_fp, g_tn, g_fn)
    print("\n--- Global (pixel-pooled) ---")
    print(f"Global Dice: {g_dice:.6f}")
    print(f"Global IoU : {g_iou:.6f}")
    print(f"Global SEN : {g_sen:.6f}")
    print(f"Global FPR : {g_fpr:.6f}")

    # Per-patient summary (volume-level)
    # Per-patient summary (volume-level)
    if len(patient_metrics) > 0:
        print("\n--- Per-patient (volume) summaries ---")

        vol_dice, vol_iou, vol_sen, vol_fpr = [], [], [], []
        vol_hd95, vol_assd, vol_bf1, vol_sdice = [], [], [], []

        per_patient_rows = []

        for pid, pm in patient_metrics.items():
            # pooled confusion -> pooled metrics
            pdice, piou, psen, pfpr = dice_iou_sen_fpr(pm["tp"], pm["fp"], pm["tn"], pm["fn"])

            # mean geometry metrics over slices
            m_hd95, _ = summarize_list(pm["hd95"])
            m_assd, _ = summarize_list(pm["assd"])
            m_bf1, _  = summarize_list(pm["bf1_tol2"])
            m_sdice, _ = summarize_list(pm["surf_dice_tol2"])

            # lists for printing summary
            vol_dice.append(pdice); vol_iou.append(piou)
            vol_sen.append(psen);   vol_fpr.append(pfpr)
            vol_hd95.append(m_hd95); vol_assd.append(m_assd)
            vol_bf1.append(m_bf1);   vol_sdice.append(m_sdice)

            # row for CSV
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
                "n_slices": pm["n_slices"],
            })

        # Save CSV once
        if per_patient_csv_path:
            per_patient_rows.sort(key=lambda r: r["eye_id"])
            write_per_patient_csv(per_patient_csv_path, per_patient_rows)
            print(f"\n[Saved] per-patient CSV -> {per_patient_csv_path}")

        # Print per-patient summary
        def print_vol(name, arr):
            mu, sd = summarize_list(arr)
            print(f"{name:18s}: {mu:.6f} ± {sd:.6f}")

        print_vol("Vol Dice (pooled)", vol_dice)
        print_vol("Vol IoU (pooled)",  vol_iou)
        print_vol("Vol SEN (pooled)",  vol_sen)
        print_vol("Vol FPR (pooled)",  vol_fpr)
        print_vol("Vol HD95 (mean)",   vol_hd95)
        print_vol("Vol ASSD (mean)",   vol_assd)
        print_vol("Vol BF1@2px (mean)",vol_bf1)
        print_vol("Vol SurfDice@2px (mean)", vol_sdice)

        # Sanity check:
        # print("Patients found:", len(patient_metrics))
        # print("Slices per patient (min/mean/max):",
        #     min(pm["n_slices"] for pm in patient_metrics.values()),
        #     np.mean([pm["n_slices"] for pm in patient_metrics.values()]),
        #     max(pm["n_slices"] for pm in patient_metrics.values()))


print("====================\n")
