import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from elm.dataset import D3Dataset
from elm.model import (
    CSAM_UNet2p5D,
    UNet3DFrawley,
    UNet2DEnc3DDec,
    UNet3D,
    UNet3D_Aniso,
    UNet2p5D_SlidingWindow,
    SwinUNETR3D,
)

# -----------------------------
# CONSTANTS
# -----------------------------
EYE_ID = ["919", "945", "990"]
SLICE_INDEX = [24,25,48]
# TARGET_LAYERS = ["conv_up_00", "conv_up_01", "conv_bottom_21"] # UNet3DFrawley
# TARGET_LAYERS = ["dec1", "dec2", "bottleneck3d"] # UNet2DEnc3DDec
# TARGET_LAYERS = ["encoder.attn.4", "decoder.blocks.0"] # CSAM_UNet2p5D
TARGET_LAYERS = ["model.decoder1", "model.decoder2", "model.decoder3"] # SwinUNETR3D


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return x


def gray_to_rgb(gray01: np.ndarray) -> np.ndarray:
    return np.stack([gray01, gray01, gray01], axis=-1)


def overlay_cam_on_gray(gray01: np.ndarray, cam01: np.ndarray, alpha=0.35) -> np.ndarray:
    """
    gray01: [H,W] in [0,1]
    cam01:  [H,W] in [0,1]
    returns RGB uint8
    """
    base = gray_to_rgb(gray01)
    # convert cam to heatmap in [0,1]
    heatmap = cv2.applyColorMap(np.uint8(np.clip(cam01 * 255.0, 0, 255)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = np.clip((1.0 - alpha) * base + alpha * heatmap, 0, 1)
    return np.uint8(overlay * 255.0)


def draw_mask_contour_rgb(base_rgb_u8: np.ndarray, mask01: np.ndarray, color=(0, 255, 0), thickness=1):
    out = base_rgb_u8.copy()
    mask255 = (mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def resolve_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if module_name not in modules:
        available = list(modules.keys())
        raise ValueError(
            f"Target layer '{module_name}' not found.\n"
            f"Try one of these examples:\n" +
            "\n".join(available[:80])
        )
    return modules[module_name]


def auto_pick_target_layer(model: nn.Module):
    """
    Picks the second-to-last Conv3d as a reasonable default.
    """
    conv3d_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv3d)]
    if len(conv3d_layers) == 0:
        raise ValueError("No Conv3d layers found.")
    if len(conv3d_layers) == 1:
        return conv3d_layers[0]
    return conv3d_layers[-2]


# -----------------------------
# 3D Grad-CAM targets
# -----------------------------
class SliceTarget3D:
    """
    Target one output slice from a 3D segmentation output.

    model_output: [B, C, D, H, W]
    mask:         [H, W]
    """
    def __init__(self, category=0, slice_index=None, mask=None):
        self.category = category
        self.slice_index = slice_index
        self.mask = mask

    def __call__(self, model_output):
        output = model_output[0, self.category]   # [D,H,W]

        if self.slice_index is None:
            raise ValueError("slice_index must be provided")

        slice_logits = output[self.slice_index]   # [H,W]

        if self.mask is None:
            return slice_logits.mean()

        return (slice_logits * self.mask).sum()


class VolumeTarget3D:
    """
    Explain the whole predicted 3D foreground volume at once.

    model_output: [B, C, D, H, W]
    mask:         [D, H, W]
    """
    def __init__(self, category=0, mask=None):
        self.category = category
        self.mask = mask

    def __call__(self, model_output):
        output = model_output[0, self.category]   # [D,H,W]
        if self.mask is None:
            return output.mean()
        return (output * self.mask).sum()


# -----------------------------
# Manual Grad-CAM for 3D
# -----------------------------
class GradCAM3D:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = None

    def _forward_hook(self, module, inputs, output):
        self.activations = output

        def _save_grad(grad):
            self.gradients = grad

        output.register_hook(_save_grad)

    def __enter__(self):
        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fwd_handle is not None:
            self.fwd_handle.remove()

    def __call__(self, target_scalar: torch.Tensor, upsample_size, batch_size=1, depth_size=49):
        """
        target_scalar: scalar tensor
        upsample_size: (D,H,W)
        returns: cam volume [D,H,W] in [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        target_scalar.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        # Case 1: true 5D layer, e.g. encoder.attn.*
        if self.activations.ndim == 5 and self.gradients.ndim == 5:
            # encoder.attn.* is [B,D,C,H,W] -> convert to [B,C,D,H,W]
            acts = self.activations.permute(0, 2, 1, 3, 4).contiguous()
            grads = self.gradients.permute(0, 2, 1, 3, 4).contiguous()

        # Case 2: 4D layer, e.g. decoder.blocks.0 or head
        elif self.activations.ndim == 4 and self.gradients.ndim == 4:
            if batch_size is None or depth_size is None:
                raise ValueError(
                    "For 4D hooked layers [B*D,C,H,W], you must provide batch_size and depth_size."
                )

            BD, C, H, W = self.activations.shape
            if BD != batch_size * depth_size:
                raise ValueError(
                    f"Expected B*D = {batch_size * depth_size}, got {BD}."
                )

            # [B*D,C,H,W] -> [B,D,C,H,W] -> [B,C,D,H,W]
            acts = self.activations.view(batch_size, depth_size, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
            grads = self.gradients.view(batch_size, depth_size, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        else:
            raise ValueError(
                f"Unsupported activation/gradient dims. "
                f"Got activations {tuple(self.activations.shape)} and gradients {tuple(self.gradients.shape)}"
            )

        # IMPORTANT: use grads / acts, not self.gradients / self.activations
        weights = grads.mean(dim=(3, 4), keepdim=True)   # [B,C,D,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,D,H,W]
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=upsample_size, mode="trilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()  # [D,H,W]

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        self.activations = None
        self.gradients = None

        return cam.astype(np.float32)


# -----------------------------
# Model loading
# -----------------------------
def build_model(model_name: str, checkpoint_path: str, device):
    if model_name == "UNet3DFrawley":
        net = UNet3DFrawley(in_channels=1, out_channels=1)
    elif model_name == "UNet2DEnc3DDec":
        net = UNet2DEnc3DDec(in_channels=1, out_channels=1)
    elif model_name == "UNet3D":
        net = UNet3D(in_channels=1, out_channels=1)
    elif model_name == "UNet3D_Aniso":
        net = UNet3D_Aniso(in_channels=1, out_channels=1)
    elif model_name == "CSAM_UNet2p5D":
        net = CSAM_UNet2p5D(
            in_channels=1,
            out_channels=1,
            num_layers=5,
            base_num=32,
            semantic=True,
            positional=True,
            slice_att=True,
        )
    elif model_name == "UNet2p5D_SlidingWindow":
        net = UNet2p5D_SlidingWindow(
            k=7,
            out_channels=1,
            num_layers=3,
            base_num=32,
            pad_mode="replicate",
        )
    elif model_name == "SwinUNETR3D":
        net = SwinUNETR3D(in_channels=1, n_classes=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


# -----------------------------
# Saving
# -----------------------------
def save_slice_outputs(
    out_dir,
    base_name,
    slice_index,
    input_slice01,
    gt_slice01,
    prob_slice01,
    pred_slice01,
    cam_slice01,
    alpha=0.35,
):
    ensure_dir(out_dir)

    input_u8 = np.uint8(np.clip(input_slice01 * 255.0, 0, 255))
    gt_u8 = np.uint8(np.clip(gt_slice01 * 255.0, 0, 255))
    prob_u8 = np.uint8(np.clip(prob_slice01 * 255.0, 0, 255))
    pred_u8 = np.uint8(np.clip(pred_slice01 * 255.0, 0, 255))
    cam_u8 = np.uint8(np.clip(cam_slice01 * 255.0, 0, 255))
    cam_overlay_u8 = overlay_cam_on_gray(input_slice01, cam_slice01, alpha=alpha)

    input_bgr = cv2.cvtColor(input_u8, cv2.COLOR_GRAY2BGR)
    gt_bgr = cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred_u8, cv2.COLOR_GRAY2BGR)
    prob_color = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    cam_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(out_dir, f"{base_name}_slice{slice_index:02d}_input.png"), input_u8)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_slice{slice_index:02d}_gt.png"), gt_u8)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_slice{slice_index:02d}_pred_prob.png"), prob_u8)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_slice{slice_index:02d}_pred_mask.png"), pred_u8)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_slice{slice_index:02d}_cam.png"), cam_u8)
    cv2.imwrite(
        os.path.join(out_dir, f"{base_name}_slice{slice_index:02d}_cam_overlay.png"),
        cv2.cvtColor(cam_overlay_u8, cv2.COLOR_RGB2BGR),
    )

    summary = np.hstack([
        input_bgr,
        gt_bgr,
        prob_color,
        pred_bgr,
        cam_color,
        cv2.cvtColor(cam_overlay_u8, cv2.COLOR_RGB2BGR),
    ])
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_slice{slice_index:02d}_summary.png"), summary)


def save_7x7_grid(
    out_dir,
    base_name,
    input_vol,
    gt_vol,
    pred_vol,
    cam_vol,
    alpha=0.35,
    show_pred_contour=True,
    show_gt_contour=False,
):
    ensure_dir(out_dir)

    D = input_vol.shape[0]
    fig, axes = plt.subplots(7, 7, figsize=(21, 21))
    axes = axes.flatten()

    for z in range(49):
        ax = axes[z]
        ax.axis("off")

        if z >= D:
            blank = np.zeros((256, 256, 3), dtype=np.uint8)
            ax.imshow(blank)
            ax.set_title(f"{z}", fontsize=8)
            continue

        input_slice01 = normalize_2d(input_vol[z])
        cam_slice01 = cam_vol[z]
        pred_slice01 = pred_vol[z]
        gt_slice01 = gt_vol[z]

        overlay = overlay_cam_on_gray(input_slice01, cam_slice01, alpha=alpha)

        if show_gt_contour:
            overlay = draw_mask_contour_rgb(overlay, gt_slice01, color=(255, 255, 0), thickness=1)
        if show_pred_contour:
            overlay = draw_mask_contour_rgb(overlay, pred_slice01, color=(0, 255, 0), thickness=1)

        ax.imshow(overlay)
        ax.set_title(f"{z}", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{base_name}_7x7.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Processing
# -----------------------------
def process_eye_layer(
    model,
    eye_id,
    sample,
    target_layer_name,
    out_dir,
    threshold=0.5,
    alpha=0.35,
    device=None,
):
    imgs = sample["image"].unsqueeze(0).to(dtype=torch.float32, device=device)
    gts = sample["mask"].unsqueeze(0).to(dtype=torch.float32, device=device)

    target_layer = resolve_module_by_name(model, target_layer_name)

    with GradCAM3D(model, target_layer) as cam3d:
        logits = model(imgs)

        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()  # [D,H,W]
        gt_bin = (gts[0, 0] > 0.5).detach().cpu().numpy().astype(np.float32) # ground truth binary mask
        input_vol = imgs[0, 0].detach().cpu().numpy()
        pred_vol = (probs > threshold).astype(np.float32)

            # -------- one 7x7 grid per requested target slice --------
        for k in SLICE_INDEX:
            if not (0 <= k < probs.shape[0]):
                print(f"[WARN] eye_id={eye_id}, target slice={k} out of range")
                continue

            target_mask = (probs[k] > threshold).astype(np.float32)
            if target_mask.sum() == 0:
                target_mask = np.ones_like(target_mask, dtype=np.float32)

            target_mask_t = torch.from_numpy(target_mask).to(device=imgs.device, dtype=torch.float32)
            slice_target = SliceTarget3D(category=0, slice_index=k, mask=target_mask_t)
            slice_score = slice_target(logits)

            with GradCAM3D(model, target_layer) as cam3d_grid:
                # forward again so hooks attach cleanly for this target
                logits_grid = model(imgs)
                slice_score = slice_target(logits_grid)

                cam_vol_for_target_slice = cam3d_grid(
                    target_scalar=slice_score,
                    upsample_size=tuple(logits_grid.shape[2:])
                )

            slice_mean = cam_vol_for_target_slice.mean(axis=(1, 2))
            slice_max = cam_vol_for_target_slice.max(axis=(1, 2))
            print(f"[DEBUG] eye={eye_id}, layer={target_layer_name}, target={k}")
            print("slice_mean:", np.round(slice_mean, 3))
            print("slice_max :", np.round(slice_max, 3))

            grid_dir = os.path.join(out_dir, eye_id, target_layer_name, "grid")
            save_7x7_grid(
                out_dir=grid_dir,
                base_name=f"{eye_id}_{target_layer_name}_targetslice{k:02d}",
                input_vol=input_vol,
                gt_vol=gt_bin,
                pred_vol=pred_vol,
                cam_vol=cam_vol_for_target_slice,
                alpha=alpha,
                show_pred_contour=False,
                show_gt_contour=False,
            )

    # -------- slice-specific CAMs for requested slices --------
    for k in SLICE_INDEX:
        if not (0 <= k < probs.shape[0]):
            print(f"[WARN] eye_id={eye_id}, slice={k} out of range")
            continue

        target_layer = resolve_module_by_name(model, target_layer_name)

        with GradCAM3D(model, target_layer) as cam3d:
            logits = model(imgs)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            gt_bin = (gts[0, 0] > 0.5).detach().cpu().numpy().astype(np.float32)
            input_vol = imgs[0, 0].detach().cpu().numpy()

            pred_mask_slice = (probs[k] > threshold).astype(np.float32)

            target_mask = pred_mask_slice
            if target_mask.sum() == 0:
                target_mask = np.ones_like(pred_mask_slice, dtype=np.float32)

            target_mask_t = torch.from_numpy(target_mask).to(device=imgs.device, dtype=torch.float32)
            target = SliceTarget3D(category=0, slice_index=k, mask=target_mask_t)
            score = target(logits)

            cam_vol = cam3d(
                target_scalar=score,
                upsample_size=tuple(logits.shape[2:])
            )

        input_slice01 = normalize_2d(input_vol[k])
        gt_slice01 = gt_bin[k].astype(np.float32)
        prob_slice01 = probs[k].astype(np.float32)
        pred_slice01 = pred_mask_slice.astype(np.float32)
        cam_slice01 = cam_vol[k].astype(np.float32)

        slice_dir = os.path.join(out_dir, eye_id, target_layer_name, "slices")
        save_slice_outputs(
            out_dir=slice_dir,
            base_name=f"{eye_id}_{target_layer_name}",
            slice_index=k,
            input_slice01=input_slice01,
            gt_slice01=gt_slice01,
            prob_slice01=prob_slice01,
            pred_slice01=pred_slice01,
            cam_slice01=cam_slice01,
            alpha=alpha,
        )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="3D Grad-CAM for OCT volume segmentation")
    parser.add_argument("--base_dir", type=str, default="./")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="./gradcam_3d_outputs")
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_img_dir = os.path.join(args.base_dir, "data_no_anomalies/test/image/")
    test_mask_dir = os.path.join(args.base_dir, "data_no_anomalies/test/mask/")
    dataset = D3Dataset(test_img_dir, test_mask_dir, scale=1, transform=False)

    model = build_model(args.model, args.checkpoint, device)

    for eye_id in EYE_ID:
        if eye_id not in dataset.eye_ids:
            print(f"[WARN] eye_id {eye_id} not found, skipping")
            continue

        idx = dataset.eye_ids.index(eye_id)
        sample = dataset[idx]

        for target_layer_name in TARGET_LAYERS:
            print(f"[INFO] eye_id={eye_id}, layer={target_layer_name}")
            process_eye_layer(
                model=model,
                eye_id=eye_id,
                sample=sample,
                target_layer_name=target_layer_name,
                out_dir=args.out_dir,
                threshold=args.threshold,
                alpha=args.alpha,
                device=device,
            )

    print(f"[DONE] All outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()


"""
python gradCAM_3D.py   --base_dir ./   --checkpoint elm-results/UNet3DFrawley_Feb-20-2026_1329_model/checkpoints/UNet3DFrawley_Feb-20-2026_1329_model_best_epoch_94.pth   --model UNet3DFrawley   --out_dir elm-results/UNet3DFrawley_Feb-20-2026_1329_model/gradcam_outputs

python gradCAM_3D.py   --base_dir ./   --checkpoint elm-results/UNet2DEnc3DDec_Feb-20-2026_2001_model/checkpoints/UNet2DEnc3DDec_Feb-20-2026_2001_model_best_epoch_98.pth   --model UNet2DEnc3DDec   --out_dir elm-results/UNet2DEnc3DDec_Feb-20-2026_2001_model/gradcam_outputs

python gradCAM_3D.py   --base_dir ./   --checkpoint elm-results/UNet2DEnc3DDec_Feb-20-2026_2001_model/checkpoints/UNet2DEnc3DDec_Feb-20-2026_2001_model_best_epoch_98.pth   --model UNet2DEnc3DDec   --out_dir elm-results/UNet2DEnc3DDec_Feb-20-2026_2001_model/gradcam_outputs

python gradCAM_3D.py   --base_dir ./   --checkpoint elm-results/CSAM_UNet2p5D_Feb-25-2026_1104_model/checkpoints/CSAM_UNet2p5D_Feb-25-2026_1104_model_best_epoch_97.pth   --model CSAM_UNet2p5D   --out_dir elm-results/CSAM_UNet2p5D_Feb-25-2026_1104_model/gradcam_outputs

python gradCAM_3D.py   --base_dir ./   --checkpoint elm-results/CSAM_UNet2p5D_Feb-25-2026_1113_model/checkpoints/CSAM_UNet2p5D_Feb-25-2026_1113_model_best_epoch_95.pth   --model CSAM_UNet2p5D   --out_dir elm-results/CSAM_UNet2p5D_Feb-25-2026_1113_model/gradcam_outputs

python gradCAM_3D.py   --base_dir ./   --checkpoint elm-results/SwinUNETR3D_Mar-19-2026_1246_model/checkpoints/SwinUNETR3D_Mar-19-2026_1246_model_best_epoch_91.pth   --model SwinUNETR3D   --out_dir elm-results/SwinUNETR3D_Mar-19-2026_1246_model/gradcam_outputs
"""
