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
from dice_loss import dice_coeff
from model import U_Net,AttU_Net, LinkNetImprove, U2NETP,R2U_Net,DeepLabv3_plus,FCN,SegNet

#model = get_efficientunet_b3(n_classes=1, concat_input=True, pretrained=True)
#model = AttU_Net(n_channels=3, n_classes=1)
model = U_Net(n_channels=3, n_classes=1)
#model = LinkNetImprove(n_channels=3, n_classes=1)
#model = U2NETP(n_channels=3,n_classes=1)
#model = R2U_Net(n_channels=3, n_classes=1,t=2)
#model = DeepLabv3_plus(n_channels=3, n_classes=1, os=16, pretrained=True, _print=True)
#model = FCN(n_channels=3, n_classes=1)
# model = SegNet(n_channels=3, n_classes=1)

if torch.cuda.is_available():
    model.cuda()

MODEL_NAME = 'ELM_Jan-07-2026_1156.pth'
model.load_state_dict(torch.load('checkpoint/' + MODEL_NAME))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

image_dir = './data/test/image/'
mask_dir  = './data/test/mask/'   # <-- ground-truth masks here (same filenames)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

# create output directory
if not os.path.exists(MODEL_NAME):
    os.makedirs(MODEL_NAME)

if not os.path.exists(MODEL_NAME + '/test_outputs/'):
    os.makedirs(MODEL_NAME + '/test_outputs/')

# ---- Dice accumulators (global / total Dice) ----
smooth = 1
total_intersection = 0.0
total_pred_sum = 0.0
total_gt_sum = 0.0

dice_list = []

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

        # ---- Per-image Dice ----
        d = dice_coeff(torch.from_numpy(pred), torch.from_numpy(gt))
        dice_list.append(d)
        print(f"  Dice: {d:.6f}")

        # ---- Global Dice accumulators ----
        # For binary masks: intersection=sum(pred & gt), sums = sum(pred), sum(gt)
        inter = np.logical_and(pred == 1, gt == 1).sum(dtype=np.float64)
        p_sum = (pred == 1).sum(dtype=np.float64)
        g_sum = (gt == 1).sum(dtype=np.float64)

        total_intersection += inter
        total_pred_sum += p_sum
        total_gt_sum += g_sum

# ---- Report ----
if len(dice_list) == 0:
    print("No Dice computed (no GT masks found or all skipped).")
else:
    mean_dice = float(np.mean(dice_list))
    global_dice = (2.0 * total_intersection + smooth) / (total_pred_sum + total_gt_sum + smooth)

    print("\n====================")
    print(f"Images evaluated: {len(dice_list)}")
    print(f"Mean Dice (per-image average): {mean_dice:.6f}")
    print(f"Total Dice (global over all pixels): {global_dice:.6f}")
    print("====================\n")
