"""
Stitches the per-model overlay PNGs from qualitative_baselines.py and
qualitative_nnunet.py into one labeled comparison figure per eye (one row =
one eye, one column per model), plus a combined figure with all selected
eyes stacked.

Usage (either conda env works, this only needs matplotlib/opencv):
  python qualitative/make_qualitative_grid.py --qual_dir qualitative/images
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

COLUMNS = [
    ("image", "Image"),
    ("gt", "Ground Truth"),
    ("SegNet", "SegNet"),
    ("R2U_Net", "R2U_Net"),
    ("UNet2DEnc3DDec", "UNet2DEnc3DDec"),
    ("UNet3DFrawley", "UNet3DFrawley"),
    ("CSAM_UNet2p5D", "CSAM_UNet2p5D"),
    ("UNet2p5D_SlidingWindow", "UNet2p5D_SlidingWindow"),
    ("nnUNet_2d", "nnU-Net (2d)"),
    ("nnUNet_3d_fullres", "nnU-Net (3d_fullres)"),
]


def load_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_row(axes_row, qual_dir: Path, entry, show_col_titles: bool):
    tag, eye_id, slice_idx = entry["tag"], entry["eye_id"], entry.get("slice_idx", "?")
    for ax, (suffix, title) in zip(axes_row, COLUMNS):
        img = load_rgb(qual_dir / f"{tag}_{eye_id}_{suffix}.png")
        ax.axis("off")
        if img is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.imshow(img)
        if show_col_titles:
            ax.set_title(title, fontsize=9)
    axes_row[0].text(
        -0.1, 0.5, f"{tag}\n({eye_id}, slice {slice_idx})", transform=axes_row[0].transAxes,
        ha="right", va="center", fontsize=10, rotation=0,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qual_dir", default="qualitative/images")
    ap.add_argument("--manifest", default=None, help="Default: <qual_dir>/manifest.json")
    args = ap.parse_args()

    qual_dir = Path(args.qual_dir)
    manifest_path = Path(args.manifest) if args.manifest else qual_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["eyes"]

    n_cols = len(COLUMNS)

    # One figure per eye
    for entry in entries:
        fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 2.6))
        draw_row(axes, qual_dir, entry, show_col_titles=True)
        fig.suptitle(
            f"eye {entry['eye_id']}, slice {entry.get('slice_idx', '?')} -- {entry['tag']} case "
            f"(nnU-Net 2d Dice={entry.get('nnunet_2d_dice', float('nan')):.3f})"
        )
        fig.tight_layout()
        out_path = qual_dir / f"grid_{entry['tag']}_{entry['eye_id']}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Saved] -> {out_path}")

    # Combined figure, one row per eye
    fig, axes = plt.subplots(len(entries), n_cols, figsize=(2.2 * n_cols, 2.6 * len(entries)))
    if len(entries) == 1:
        axes = [axes]
    for i, entry in enumerate(entries):
        draw_row(axes[i], qual_dir, entry, show_col_titles=(i == 0))
    fig.tight_layout()
    out_path = qual_dir / "grid_all.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] -> {out_path}")


if __name__ == "__main__":
    main()
