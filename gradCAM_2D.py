import os
import argparse
import numpy as np
import cv2
from PIL import Image
from elm.dataset import make_2d_transforms

import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from elm.model import (
    U_Net, AttU_Net, LinkNetImprove, U2NETP, R2U_Net,
    DeepLabv3_plus, FCN, SegNet, SwinEncoderUNet2D
)

EYE_ID = ["919", "945", "990"]
SLICE_INDEX = [24, 25, 48]

# -----------------------------
# Utilities
# -----------------------------
def load_image_as_tensor(image_path, img_size=256, single_channel=False, device="cuda"):
    """
    Matches BasicDataset preprocessing for inference/Grad-CAM.

    Returns:
      input_tensor:    [1, C, H, W] normalized tensor for model input
      rgb_for_overlay: [H, W, 3] float in [0,1], resized to img_size
      raw_gray:        [H, W] float in [0,1], resized to img_size
    """
    transform = make_2d_transforms(train=False, out_size=(img_size, img_size))

    img = Image.open(image_path)

    if single_channel:
        img = img.convert("L")
        img_np = np.array(img)
        rgb_np = np.stack([img_np, img_np, img_np], axis=-1)
    else:
        img = img.convert("RGB")
        img_np = np.array(img)
        rgb_np = img_np

    gray_np = np.array(img.convert("L"))

    # model input
    out = transform(image=img_np, mask=np.zeros(img_np.shape[:2], dtype=np.uint8))
    input_tensor = out["image"].unsqueeze(0).to(device=device, dtype=torch.float32)

    # resize visualization images to match CAM size
    rgb_for_overlay = cv2.resize(
        rgb_np, (img_size, img_size), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32) / 255.0

    raw_gray = cv2.resize(
        gray_np, (img_size, img_size), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32) / 255.0

    return input_tensor, rgb_for_overlay, raw_gray


class SemanticSegmentationTarget:
    """
    Target for segmentation CAM.
    Uses either:
      - the predicted foreground mask (> threshold), or
      - the full foreground logit map if predicted mask is empty
    """
    def __init__(self, category=0, mask=None):
        self.category = category
        self.mask = mask

    def __call__(self, model_output):
        # model_output: [B, C, H, W]
        output = model_output[0, self.category]
        if self.mask is None:
            return output.mean()
        return (output * self.mask).sum()


def build_model(model_name, checkpoint_path, device):
    """
    Match your training-time constructor style.
    """
    if model_name == "SwinEncoderUNet2D":
        model = SwinEncoderUNet2D(
            n_channels=3,
            n_classes=1,
            backbone="swin_tiny_patch4_window7_224",
            pretrained=True,
        )
    elif model_name == "U_Net":
        model = U_Net(n_channels=3, n_classes=1)
    elif model_name == "AttU_Net":
        model = AttU_Net(n_channels=3, n_classes=1)
    elif model_name == "LinkNetImprove":
        model = LinkNetImprove(n_channels=3, n_classes=1)
    elif model_name == "U2NETP":
        model = U2NETP(in_ch=3, out_ch=1)
    elif model_name == "R2U_Net":
        model = R2U_Net(n_channels=3, n_classes=1, t=2)
    elif model_name == "DeepLabv3_plus":
        model = DeepLabv3_plus(n_channels=3, n_classes=1, os=16, pretrained=True, _print=False)
    elif model_name == "FCN":
        model = FCN(n_channels=3, n_classes=1)
    elif model_name == "SegNet":
        model = SegNet(n_channels=3, n_classes=1)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_default_target_layer(model, model_name):
    """
    Sensible defaults.
    For SwinEncoderUNet2D, use a late decoder conv for a spatially meaningful CAM.
    This avoids dealing with raw Swin NHWC internals.
    """
    if model_name == "SwinEncoderUNet2D":
        # Good first choice: late decoder layer, closer to segmentation output
        # Adjust if your class structure differs slightly.
        return model.dec2.conv2.block[0]

    if model_name == "U_Net":
        # Common fallback for U-Net-like encoder/decoder models
        for attr in ["Conv5", "Up5", "Up_conv5", "Conv4"]:
            if hasattr(model, attr):
                layer = getattr(model, attr)
                if isinstance(layer, nn.Module):
                    return layer

    if model_name == "AttU_Net":
        for attr in ["Up_conv5", "Up5", "Conv5"]:
            if hasattr(model, attr):
                layer = getattr(model, attr)
                if isinstance(layer, nn.Module):
                    return layer

    if model_name == "R2U_Net":
        for attr in ["Up_RRCNN5", "RRCNN5", "Up5"]:
            if hasattr(model, attr):
                layer = getattr(model, attr)
                if isinstance(layer, nn.Module):
                    return layer

    if model_name == "DeepLabv3_plus":
        # Try ASPP / decoder-ish structures if present
        for attr in ["aspp", "decoder", "layer4"]:
            if hasattr(model, attr):
                layer = getattr(model, attr)
                if isinstance(layer, nn.Module):
                    return layer

    if model_name == "FCN":
        for attr in ["classifier", "layer4"]:
            if hasattr(model, attr):
                layer = getattr(model, attr)
                if isinstance(layer, nn.Module):
                    return layer

    if model_name == "SegNet":
        return model.conv21d

    if model_name == "LinkNetImprove":
        for attr in ["decoder4", "decoder3", "encoder4"]:
            if hasattr(model, attr):
                layer = getattr(model, attr)
                if isinstance(layer, nn.Module):
                    return layer

    raise ValueError(
        f"Could not infer a default target layer for {model_name}. "
        "Print model.named_modules() and choose one manually."
    )


def save_outputs(
    out_dir,
    base_name,
    rgb_img,
    pred_prob,
    pred_mask,
    cam_overlay,
):
    os.makedirs(out_dir, exist_ok=True)

    pred_prob_u8 = np.uint8(np.clip(pred_prob * 255.0, 0, 255))
    pred_mask_u8 = np.uint8(pred_mask * 255)
    cam_overlay_u8 = np.uint8(np.clip(cam_overlay, 0, 255))

    cv2.imwrite(os.path.join(out_dir, f"{base_name}_pred_prob.png"), pred_prob_u8)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_pred_mask.png"), pred_mask_u8)
    cv2.imwrite(
        os.path.join(out_dir, f"{base_name}_gradcam_overlay.png"),
        cv2.cvtColor(cam_overlay_u8, cv2.COLOR_RGB2BGR),
    )

    # optional 3-panel summary
    rgb_u8 = np.uint8(np.clip(rgb_img * 255.0, 0, 255))
    rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    pred_prob_color = cv2.applyColorMap(pred_prob_u8, cv2.COLORMAP_JET)
    pred_mask_color = cv2.cvtColor(pred_mask_u8, cv2.COLOR_GRAY2BGR)

    top = np.hstack([rgb_bgr, pred_prob_color, pred_mask_color])
    bottom = np.hstack([
        rgb_bgr,
        cv2.cvtColor(cam_overlay_u8, cv2.COLOR_RGB2BGR),
        rgb_bgr
    ])
    summary = np.vstack([top, bottom])

    cv2.imwrite(os.path.join(out_dir, f"{base_name}_summary.png"), summary)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for OCT ELM segmentation")
    parser.add_argument("--eye-id", type=str, nargs="+", default=EYE_ID, help="Eye IDs, e.g. 919 945 990")
    parser.add_argument("--slice-index", type=int, nargs="+", default=SLICE_INDEX, help="Slice indices, e.g. 24 25 48")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--model-name", type=str, default="SwinEncoderUNet2D",
                        choices=[
                            "SwinEncoderUNet2D", "U_Net", "AttU_Net", "LinkNetImprove",
                            "U2NETP", "R2U_Net", "DeepLabv3_plus", "FCN", "SegNet"
                        ])
    parser.add_argument("--out-dir", type=str, default="./gradcam_outputs")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    model = build_model(args.model_name, args.checkpoint, device)
    target_layer = get_default_target_layer(model, args.model_name)

    for eye_id in args.eye_id:
        for slice_idx in args.slice_index:
            image_path = os.path.join("data_no_anomalies/test/image", f"{eye_id}-{slice_idx}.png")

            if not os.path.exists(image_path):
                print(f"[MISSING] {image_path}")
                continue

            input_tensor, rgb_img, gray_img = load_image_as_tensor(
                image_path,
                img_size=args.img_size,
                single_channel=False,
                device=device
            )

            # Forward once to get segmentation target mask
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            pred_mask = (probs > args.threshold).astype(np.float32)

            # If prediction is empty, fall back to full map target
            target_mask = pred_mask
            if target_mask.sum() == 0:
                target_mask = np.ones_like(pred_mask, dtype=np.float32)

            target_mask_t = torch.from_numpy(target_mask).to(device=device, dtype=torch.float32)
            targets = [SemanticSegmentationTarget(category=0, mask=target_mask_t)]

            with GradCAM(model=model, target_layers=[target_layer]) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            rgb_u8 = np.uint8(np.clip(rgb_img * 255.0, 0, 255))
            cv2.imwrite(
                os.path.join(args.out_dir, f"debug_input_{eye_id}-{slice_idx}.png"),
                cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR),
            )

            base_name = f"{eye_id}-{slice_idx}"
            save_outputs(
                out_dir=args.out_dir,
                base_name=f"{base_name}_{args.model_name}",
                rgb_img=rgb_img,
                pred_prob=probs,
                pred_mask=pred_mask,
                cam_overlay=cam_overlay,
            )

            print(f"Saved Grad-CAM outputs for {eye_id}-{slice_idx} to: {args.out_dir}")


if __name__ == "__main__":
    main()

"""
python gradCAM_2D.py \
  --checkpoint elm-results/SwinEncoderUNet2D_Mar-16-2026_1515_model/checkpoints/SwinEncoderUNet2D_Mar-16-2026_1515_model_best_epoch_20.pth \
  --model-name SwinEncoderUNet2D \
  --out-dir elm-results/SwinEncoderUNet2D_Mar-16-2026_1515_model/gradcam_outputs

python gradCAM_2D.py \
  --image data_no_anomalies/test/image/919-24.png \
  --checkpoint elm-results/SegNet_Feb-16-2026_1745_model/checkpoints/SegNet_Feb-16-2026_1745_model_best_epoch_39.pth \
  --model-name SegNet \
  --out-dir elm-results/SegNet_Feb-16-2026_1745_model/gradcam_outputs

  python gradCAM_2D.py \
  --checkpoint elm-results/SegNet_Feb-16-2026_1745_model/checkpoints/SegNet_Feb-16-2026_1745_model_best_epoch_39.pth \
  --model-name SegNet \
  --out-dir elm-results/SegNet_Feb-16-2026_1745_model/gradcam_outputs
"""
