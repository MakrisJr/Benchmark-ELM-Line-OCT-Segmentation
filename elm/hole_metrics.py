"""Per-slice comparison of the model's predicted macular-hole gap against the
GT gap, shared by predict_cv2d.py, predict_cv3d.py, and
nnunet/predict_cv.py's --hole_decomposition flag.

The ELM line is annotated as a thin near-horizontal curve. Every eye in this
dataset has a macular hole, but the hole only crosses some of an eye's
slices; on those slices the GT curve has an interior gap (a run of columns,
strictly between the leftmost and rightmost annotated columns, with no GT
foreground). The question this module answers is not "does this eye have a
hole" (always yes) but, per slice: did the model get the gap right, in
either direction --

  - On slices that cross the hole (GT has a gap): did the model find one at
    all (or bridge straight across it), and if it found one, how well do its
    margins -- the ELM termination points on either side of the hole -- line
    up with the annotated ones. See gap_geometry().
  - On slices where the annotated line is fully continuous (GT has no gap):
    did the model predict a spurious gap anyway -- a false-positive hole,
    plausible from vessel-shadow artifacts or ordinary thin-structure
    dropout unrelated to the real hole. See spurious_gap().

analyze_slice() is the per-slice entry point that routes to whichever of the
two applies, and analyze_slice_row_fields() flattens the result into a fixed
set of gap_*-prefixed keys (always the same keys, values None where not
applicable) so every slice contributes the same CSV schema regardless of
which branch it took.
"""

from typing import Optional

import numpy as np


def _contiguous_runs(bool_arr: np.ndarray) -> list:
    """List of (start, end_inclusive) for each run of True in bool_arr."""
    runs = []
    idx = 0
    n = len(bool_arr)
    while idx < n:
        if not bool_arr[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and bool_arr[idx]:
            idx += 1
        runs.append((start, idx - 1))
    return runs


def _filter_min_width(bool_arr: np.ndarray, min_width: int) -> np.ndarray:
    out = np.zeros_like(bool_arr)
    for start, end in _contiguous_runs(bool_arr):
        if end - start + 1 >= min_width:
            out[start:end + 1] = True
    return out


def hole_columns(gt01_2d: np.ndarray, min_gap_width: int = 5) -> np.ndarray:
    """Boolean array of shape (width,), True for columns inside an interior
    gap of the GT annotation (a run of >= min_gap_width columns with no GT
    foreground, strictly between the leftmost and rightmost annotated
    columns). Slices with no annotation, or no gap that wide, return all
    False."""
    width = gt01_2d.shape[1]
    cols_with_fg = np.where(gt01_2d.any(axis=0))[0]
    mask = np.zeros(width, dtype=bool)
    if cols_with_fg.size == 0:
        return mask

    lo, hi = int(cols_with_fg.min()), int(cols_with_fg.max())
    present = np.zeros(width, dtype=bool)
    present[cols_with_fg] = True
    gap = np.zeros(width, dtype=bool)
    gap[lo:hi + 1] = ~present[lo:hi + 1]
    return _filter_min_width(gap, min_gap_width)


def gap_geometry(pred01_2d: np.ndarray, gt01_2d: np.ndarray, min_gap_width: int = 5) -> Optional[dict]:
    """Compares the model's predicted gap against the GT gap for a single
    slice. Returns None if this slice doesn't cross the hole (no qualifying
    GT gap), meaning there's nothing to compare here.

    The predicted gap is searched for only within the GT's own annotated
    column range [gt_lo, gt_hi] -- that's the window where the ELM should
    exist per the annotator, so prediction extent mismatches outside it
    aren't hole-related and shouldn't count against this metric.

    Fields:
      bridged: True if the model left no qualifying gap in that window at
        all (drew a continuous line straight through the hole).
      gt_width / pred_width: widths (columns) of the GT gap and the
        predicted gap (0 if bridged).
      width_error: pred_width - gt_width (signed; negative means the model's
        gap is narrower than the true one -- it's bridging part of the way
        in -- and bridged is the extreme case of this).
      left_margin_error / right_margin_error: |predicted margin - GT margin|
        in columns, for each side of the gap. None when bridged, since there
        are no predicted margins to compare.
    """
    width = gt01_2d.shape[1]
    gt_cols_fg = np.where(gt01_2d.any(axis=0))[0]
    if gt_cols_fg.size == 0:
        return None
    gt_lo, gt_hi = int(gt_cols_fg.min()), int(gt_cols_fg.max())

    gt_hole = hole_columns(gt01_2d, min_gap_width=min_gap_width)
    gt_runs = _contiguous_runs(gt_hole)
    if not gt_runs:
        return None
    gt_start, gt_end = max(gt_runs, key=lambda r: r[1] - r[0])
    gt_width = gt_end - gt_start + 1

    window = np.zeros(width, dtype=bool)
    window[gt_lo:gt_hi + 1] = True
    pred_present = pred01_2d.any(axis=0)
    pred_gap = _filter_min_width(window & ~pred_present, min_gap_width)
    pred_runs = _contiguous_runs(pred_gap)

    if not pred_runs:
        return {
            "bridged": True,
            "gt_width": gt_width, "pred_width": 0,
            "width_error": -gt_width,
            "left_margin_error": None, "right_margin_error": None,
        }

    def overlap(run):
        return max(0, min(run[1], gt_end) - max(run[0], gt_start) + 1)

    def center_dist(run):
        return abs((run[0] + run[1]) / 2 - (gt_start + gt_end) / 2)

    pred_start, pred_end = max(pred_runs, key=lambda r: (overlap(r), -center_dist(r)))
    pred_width = pred_end - pred_start + 1

    return {
        "bridged": False,
        "gt_width": gt_width, "pred_width": pred_width,
        "width_error": pred_width - gt_width,
        "left_margin_error": abs(pred_start - gt_start),
        "right_margin_error": abs(pred_end - gt_end),
    }


def spurious_gap(pred01_2d: np.ndarray, gt01_2d: np.ndarray, min_gap_width: int = 5) -> Optional[dict]:
    """For a slice where the annotated ELM line is fully continuous (no
    qualifying GT gap), checks whether the model predicts a gap there anyway
    -- a false-positive hole. Returns None if this slice has no GT
    annotation at all (nothing to compare), or if GT actually has a
    qualifying gap here (that's gap_geometry's territory, not this).

    Mirrors gap_geometry(): the predicted gap is searched for only within
    the GT's own annotated column range.
    """
    width = gt01_2d.shape[1]
    gt_cols_fg = np.where(gt01_2d.any(axis=0))[0]
    if gt_cols_fg.size == 0:
        return None
    if hole_columns(gt01_2d, min_gap_width=min_gap_width).any():
        return None

    gt_lo, gt_hi = int(gt_cols_fg.min()), int(gt_cols_fg.max())
    window = np.zeros(width, dtype=bool)
    window[gt_lo:gt_hi + 1] = True
    pred_present = pred01_2d.any(axis=0)
    pred_gap = _filter_min_width(window & ~pred_present, min_gap_width)
    runs = _contiguous_runs(pred_gap)

    if not runs:
        return {"spurious_gap": False, "spurious_gap_width": 0}

    start, end = max(runs, key=lambda r: r[1] - r[0])
    return {"spurious_gap": True, "spurious_gap_width": end - start + 1}


def analyze_slice(pred01_2d: np.ndarray, gt01_2d: np.ndarray, min_gap_width: int = 5) -> Optional[dict]:
    """Per-slice entry point: routes to gap_geometry() if the GT has a
    qualifying gap on this slice, or spurious_gap() if the GT line is
    continuous here. Returns None if this slice has no GT annotation at all.
    The returned dict always has a `gt_has_gap` bool so callers know which
    per-eye aggregation (summarize_gap_geometry / summarize_spurious_gaps)
    the result belongs to."""
    g = gap_geometry(pred01_2d, gt01_2d, min_gap_width=min_gap_width)
    if g is not None:
        return {"gt_has_gap": True, **g}
    s = spurious_gap(pred01_2d, gt01_2d, min_gap_width=min_gap_width)
    if s is not None:
        return {"gt_has_gap": False, **s}
    return None


_GAP_ROW_FIELDS = (
    "gap_gt_has_gap",
    "gap_bridged", "gap_gt_width", "gap_pred_width", "gap_width_error",
    "gap_left_margin_error", "gap_right_margin_error",
    "gap_spurious_gap", "gap_spurious_gap_width",
)


def gap_result_to_row_fields(r: Optional[dict]) -> dict:
    """Flattens an analyze_slice() result (or None) into a fixed set of
    gap_*-prefixed keys -- always the same keys, values None for whichever
    branch didn't apply. Use this (not a raw dict.update of the analyze_slice
    result) when building a per-slice CSV row: a variable key set per row
    (only present when a gap existed) will crash csv.DictWriter as soon as
    row order puts a narrower-schema row first."""
    row = dict.fromkeys(_GAP_ROW_FIELDS)
    if r is None:
        return row
    row["gap_gt_has_gap"] = r["gt_has_gap"]
    if r["gt_has_gap"]:
        row["gap_bridged"] = r["bridged"]
        row["gap_gt_width"] = r["gt_width"]
        row["gap_pred_width"] = r["pred_width"]
        row["gap_width_error"] = r["width_error"]
        row["gap_left_margin_error"] = r["left_margin_error"]
        row["gap_right_margin_error"] = r["right_margin_error"]
    else:
        row["gap_spurious_gap"] = r["spurious_gap"]
        row["gap_spurious_gap_width"] = r["spurious_gap_width"]
    return row


def analyze_slice_row_fields(pred01_2d: np.ndarray, gt01_2d: np.ndarray, min_gap_width: int = 5) -> dict:
    """Convenience wrapper: analyze_slice() + gap_result_to_row_fields() in
    one call. Prefer calling analyze_slice() once yourself and passing the
    result to gap_result_to_row_fields() when you also need the raw result
    (e.g. for per-eye aggregation) -- this wrapper would otherwise recompute
    it a second time."""
    return gap_result_to_row_fields(analyze_slice(pred01_2d, gt01_2d, min_gap_width=min_gap_width))


def analyze_slice_3d(pred01_3d: np.ndarray, gt01_3d: np.ndarray, min_gap_width: int = 5) -> list:
    """analyze_slice() applied per z-slice of a (D, H, W) volume. Returns a
    list of (slice_idx, result_dict) for every slice with GT annotation
    (result_dict has a `gt_has_gap` bool routing it to gap_geometry's or
    spurious_gap's schema); slices with no annotation at all are omitted."""
    out = []
    for z in range(gt01_3d.shape[0]):
        r = analyze_slice(pred01_3d[z], gt01_3d[z], min_gap_width=min_gap_width)
        if r is not None:
            out.append((z, r))
    return out


def summarize_gap_geometry(geoms: list) -> dict:
    """Aggregates a list of gap_geometry() dicts (e.g. all hole-crossing
    slices for one eye) into per-eye summary fields."""
    n = len(geoms)
    n_bridged = sum(1 for g in geoms if g["bridged"])
    width_errors = [g["width_error"] for g in geoms]
    left_errs = [g["left_margin_error"] for g in geoms if g["left_margin_error"] is not None]
    right_errs = [g["right_margin_error"] for g in geoms if g["right_margin_error"] is not None]

    def mean_or_nan(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "n_hole_slices": n,
        "n_bridged": n_bridged,
        "bridged_frac": (n_bridged / n) if n else float("nan"),
        "mean_width_error": mean_or_nan(width_errors),
        "mean_abs_width_error": mean_or_nan([abs(w) for w in width_errors]),
        "mean_left_margin_error": mean_or_nan(left_errs),
        "mean_right_margin_error": mean_or_nan(right_errs),
    }


def summarize_spurious_gaps(records: list) -> dict:
    """Aggregates a list of spurious_gap() dicts (e.g. all continuous-GT
    slices for one eye) into per-eye summary fields: how often the model
    predicts a false-positive hole where the annotation shows a continuous
    ELM line, and how wide those spurious gaps tend to be."""
    n = len(records)
    n_spurious = sum(1 for r in records if r["spurious_gap"])
    widths = [r["spurious_gap_width"] for r in records if r["spurious_gap"]]

    def mean_or_nan(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "n_continuous_slices": n,
        "n_spurious_gaps": n_spurious,
        "spurious_gap_frac": (n_spurious / n) if n else float("nan"),
        "mean_spurious_gap_width": mean_or_nan(widths),
    }
