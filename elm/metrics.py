"""Segmentation metrics shared by the 2D/3D CV and nnU-Net evaluation scripts.

Consolidated from predict_cv2d.py, predict_cv3d.py, predict.py, predict3D.py,
and nnunet/predict_cv.py, which all implemented the same formulas
independently. Confusion-count metrics (dice/iou/sen/fpr) and RMSE are
dimension-agnostic and shared as-is. Boundary-based metrics (ASSD/HD/HD95,
boundary F1, surface Dice) come in separate 2D and 3D variants because they
use different boundary-extraction and distance-transform implementations:

- 2D uses OpenCV: erosion-based boundary extraction, `cv2.distanceTransform`
  (an approximate Euclidean distance transform), and tolerance matching via
  dilating boundaries with a `(2*tol+1, 2*tol+1)` ellipse structuring element.
- 3D uses SciPy: erosion-based surface extraction (6-connectivity),
  `scipy.ndimage.distance_transform_edt` (an exact Euclidean distance
  transform), and tolerance matching via direct thresholding of the distance
  values at `tol_vox`.

This 2D/3D split is intentional (matches each source file), not a bug -- the
two families were verified to agree with each other internally, but are not
numerically identical across dimensions for the same tolerance value.
"""

import numpy as np
from scipy import ndimage as ndi

try:
    import cv2
except ImportError:  # pragma: no cover - only needed for the 2D metrics
    cv2 = None


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


def summarize_rows(rows: list, key: str):
    return summarize_list([r.get(key, float("nan")) for r in rows])


# ---------------------------------------------------------------------------
# 2D (per-slice) boundary metrics -- OpenCV based
# ---------------------------------------------------------------------------

def extract_boundary_2d(mask01: np.ndarray) -> np.ndarray:
    mask255 = mask01.astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    er = cv2.erode(mask255, k, iterations=1)
    return (cv2.subtract(mask255, er) > 0).astype(np.uint8)


def boundary_f1_2d(pred01: np.ndarray, gt01: np.ndarray, tol: int = 2) -> float:
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


def surface_dice_2d(pred01: np.ndarray, gt01: np.ndarray, tol: int = 2) -> float:
    pb, gb = extract_boundary_2d(pred01), extract_boundary_2d(gt01)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    pb_d, gb_d = cv2.dilate(pb, k, iterations=1), cv2.dilate(gb, k, iterations=1)
    inter = np.logical_and(pb == 1, gb_d == 1).sum() + np.logical_and(gb == 1, pb_d == 1).sum()
    return safe_div(inter, pb.sum() + gb.sum())


def assd_hd_hd95_2d(pred01: np.ndarray, gt01: np.ndarray):
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


# ---------------------------------------------------------------------------
# 3D (per-volume) boundary metrics -- SciPy based
# ---------------------------------------------------------------------------

def surface_voxels_3d(mask01: np.ndarray) -> np.ndarray:
    mask = mask01.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndi.binary_erosion(mask, structure=structure, iterations=1, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def assd_hd_hd95_3d(pred01: np.ndarray, gt01: np.ndarray, spacing=(1.0, 1.0, 1.0)):
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


def boundary_f1_3d(pred01: np.ndarray, gt01: np.ndarray, tol_vox: float = 2, spacing=(1.0, 1.0, 1.0)) -> float:
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


def surface_dice_3d(pred01: np.ndarray, gt01: np.ndarray, tol_vox: float = 2, spacing=(1.0, 1.0, 1.0)) -> float:
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
