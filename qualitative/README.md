# Qualitative Comparison Figures

Generates side-by-side qualitative comparisons of models (baselines + nnU-Net 2d + nnU-Net 3d_fullres) on a spread of worst/median/best eyes, picked by nnU-Net (2d) per-eye Dice so the selection is a stable, independently-computed reference shared across every model.

## Pipeline

Three scripts run in sequence, each depending on the previous one's output:

1. **`qualitative_baselines.py`** (elm-gpu env, run from the repo root) -- picks the representative eyes, then for each one saves the native-resolution raw image, an image+ground-truth overlay, and an image+prediction overlay for each of the baseline architectures (SegNet, R2U_Net, UNet2DEnc3DDec, UNet3DFrawley, CSAM_UNet2p5D, UNet2p5D_SlidingWindow). Each baseline is evaluated with the exact fold checkpoint that held that eye out during 5-fold CV. Writes `images/manifest.json` (eye_id, slice_idx, fold per selected eye) so the next script doesn't need to re-derive the selection.

   ```bash
   sbatch qualitative/qualitative_baselines.sbatch
   # or directly:
   python qualitative/qualitative_baselines.py --n_per_category 2
   ```

2. **`qualitative_nnunet.py`** (nnunet env, `conda activate nnunet && source nnUnet.sh` first) -- reads `images/manifest.json` and adds the nnU-Net 2d and 3d_fullres prediction-overlay panels for the same eyes/slices/folds.

   ```bash
   sbatch qualitative/qualitative_nnunet.sbatch
   # or directly:
   python qualitative/qualitative_nnunet.py
   ```

3. **`make_qualitative_grid.py`** (either conda env works -- only needs matplotlib/opencv) -- stitches the per-model overlay PNGs into one labeled comparison figure per eye (one row = one eye, one column per model), plus a combined `grid_all.png` with every selected eye stacked.

   ```bash
   python qualitative/make_qualitative_grid.py --qual_dir qualitative/images
   ```

## Layout

- `qualitative_baselines.py`, `qualitative_baselines.sbatch` -- step 1
- `qualitative_nnunet.py`, `qualitative_nnunet.sbatch` -- step 2
- `make_qualitative_grid.py` -- step 3
- `images/` -- generated output (gitignored except `manifest.json`): `<tag>_<eye_id>_image.png`, `_gt.png`, and one overlay PNG per model per eye, plus `grid_<tag>_<eye_id>.png` and `grid_all.png` from step 3

## Rerunning with a different eye selection

`--n_per_category` on `qualitative_baselines.py` controls how many worst/median/best eyes are picked (default 2 each, 6 total). Changing it regenerates `manifest.json` with a new selection, so `qualitative_nnunet.py` and `make_qualitative_grid.py` must be rerun afterward to stay in sync.
