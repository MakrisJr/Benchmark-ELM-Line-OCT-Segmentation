# nnU-Net Pipeline

Converts `data_no_anomalies/` into nnU-Net v2 raw datasets and wires up this repo's own 5-fold rotating CV split (`metadata.csv` `split_fold0..4`, see `../make_cv_splits.py`) so nnU-Net is directly comparable to the other models benchmarked in `elm-results/`.

Two datasets:

- `Dataset001_ELM2D` -- one training case per OCT slice (4851 cases), 2D config
- `Dataset002_ELM3D` -- one training case per eye (99 cases), 49 slices stacked into a NIfTI volume per patient, 3D config

`nnunetv2` (2.8.1) is already installed in its own `nnunet` conda env (not `elm-gpu` -- nnunetv2 needs Python 3.10, but `elm-gpu` is pinned to 3.9), so no extra setup is needed there. All commands in this pipeline assume `conda activate nnunet`. You'll also need to `source nnUnet.sh` (repo root) first to set the `nnUNet_raw`/`nnUNet_preprocessed`/`nnUNet_results` env vars nnU-Net's CLI tools expect -- both sbatch scripts below do this automatically, but do it yourself if running `nnUNetv2_*` commands by hand.

## Why no imagesTs

nnU-Net's raw format assumes one *fixed* test set held out from `imagesTr`. This repo's CV instead rotates a different 20% test set per fold. To keep all 5 of this repo's folds usable, every case goes into `imagesTr`/`labelsTr`, and per-fold train/val/test membership lives in:

- `nnUNet_preprocessed/<dataset>/splits_final.json` -- nnU-Net's own format, read automatically during training (`train`/`val` per fold)
- `nnUNet_preprocessed/<dataset>/test_cases_per_fold.json` -- this repo's own extra file listing each fold's held-out test case IDs, for `nnUNetv2_predict` + scoring against `labelsTr` after training

## Steps

1. **Convert + preprocess** (`nnunet/preprocess.sbatch`) -- CPU only, no GPU needed, but still don't run it on the login node (resampling 4851 slices + 99 volumes is nontrivial CPU work). Submit it once you can get a batch allocation:

   ```bash
   sbatch nnunet/preprocess.sbatch
   ```

   This runs, in order: `prepare_2d.py`, `prepare_3d.py`, `nnUNetv2_plan_and_preprocess -d 1` and `-d 2` (using `nnUNetPlannerResEncM`, since the other ResEnc planners target GPU memory budgets larger than this small dataset needs), then `make_splits.py` (which overwrites nnU-Net's auto-generated splits with this repo's own folds).

   Each conversion/split script also runs standalone if you want to inspect or tweak something, e.g. a quick 2-patient smoke test:

   ```bash
   python nnunet/prepare_2d.py --nnunet-raw /tmp/nnunet_smoke --limit-patients 2
   ```

2. **Inspect the plan** before training: open `nnUNet_preprocessed/Dataset002_ELM3D/nnUNetPlans.json` and check which 3D configuration nnU-Net actually chose (`3d_fullres` vs. a `3d_lowres` + `3d_cascade_fullres` pair) and the planned patch size / target spacing.

3. **Train** (`nnunet/train.sbatch`, GPU required) -- takes `<dataset_id> <configuration> <fold>` as arguments and trains a single fold with `-p nnUNetResEncUNetMPlans`, e.g.:

   ```bash
   sbatch nnunet/train.sbatch 1 2d 0
   ```

   To train all folds (2D x5 + 2D x5 + 3D x5) at once, each as its own job so they run in parallel across GPUs rather than queued one after another, use `nnunet/submit_folds.sh`:

   ```bash
   ./nnunet/submit_folds.sh
   ```

4. **Predict + evaluate** (`nnunet/predict_cv.py`, GPU required) -- once every fold has finished training, run this once per trained `(dataset_id, configuration)` combo. For each fold it symlinks that fold's held-out test cases (from `test_cases_per_fold.json`) into a temp input folder, runs `nnUNetv2_predict` with only that fold's model (`-f <fold>`), then scores the predictions against `labelsTr`. Dataset001_ELM2D cases are per-slice, scored like `predict_cv2d.py` (per-slice metrics + pooled per-eye aggregation); Dataset002_ELM3D cases are per-eye volumes regardless of whether `2d` or `3d_fullres` was trained, scored like `predict_cv3d.py` (per-volume metrics). Output CSVs use the same `vol_*_pooled`/`vol_*_mean` column names as those two scripts, so they plug directly into `compare_csvs.py`:

   ```bash
   python nnunet/predict_cv.py --dataset_id 1 --configuration 2d
   python nnunet/predict_cv.py --dataset_id 2 --configuration 2d
   python nnunet/predict_cv.py --dataset_id 2 --configuration 3d_fullres
   ```

   Or submit all 3 combos as sbatch jobs at once with `nnunet/submit_predictions.sh`. CSVs land in `nnUNet_results/<dataset>/<trainer>__<plans>__<configuration>/cv_eval/` (`cv_fold_summary.csv`, `cv_per_patient_all_folds.csv`, `cv_oof_eye_summary.csv`, plus per-fold per-slice/per-volume CSVs).

## Known caveats to revisit

- **3D voxel spacing** defaults to this dataset's real acquisition geometry: 5.47um x 3.87um x 29um (x/width, y/height, z/slice), converted to mm as `--spacing 0.00547 0.00387 0.029` (SimpleITK x/y/z order) in `prepare_3d.py`. This is fairly anisotropic (z spacing ~5-7x the in-plane spacing) -- expect nnU-Net's planner to pick this up and use anisotropic resampling / axis-aware patch sizing for `3d_fullres`, worth double-checking in `nnUNetPlans.json` after preprocessing.
- Per-patient image size varies across the 99 patients (confirmed uniform *within* each patient's 49 slices, but not across patients) -- nnU-Net handles variable-size cases fine, just don't assume a single fixed resolution when writing downstream analysis code.
