import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dice_loss import dice_coeff, dice_loss, dice_per_slice_mean


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader) * loader.batch_size
    total_imgs = 0
    tot = 0
    criterion = dice_loss

    with tqdm(total=n_val, desc='Validation round', disable = not sys.stdout.isatty(), unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item() * imgs.size(0)
                total_imgs += imgs.size(0)
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                total_imgs += imgs.size(0)
                tot += dice_coeff(pred, true_masks).item() * imgs.size(0) # sum over batch
            pbar.update()

    net.train()
    return tot / total_imgs

def eval_net_windows(net, loader, device, threshold: float = 0.5) -> float:
    """
    Window-level evaluation.

    Expects each batch from loader:
      batch['image']: [B, K, 1, 7, H, W]
      batch['mask'] : [B, K, 1, 7, H, W]

    After flattening K:
      imgs, true_masks: [N, 1, 7, H, W]

    Model must output logits: [N, 1, 7, H, W]

    Returns:
      mean Dice across validation batches (mean over N and D).
    """
    net.eval()
    n_val = len(loader)
    tot = 0.0
    total_samples = 0.0

    with tqdm(total=n_val, desc='Validation (windows)', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image'].to(device=device, dtype=torch.float32)      # [B,K,1,7,H,W]
            true_masks = batch['mask'].to(device=device, dtype=torch.float32) # [B,K,1,7,H,W]

            B, K = imgs.shape[0], imgs.shape[1]
            H, W = imgs.shape[-2], imgs.shape[-1]
            total_samples += B

            imgs = imgs.view(B * K, 1, 7, H, W)                 # [N,1,7,H,W]
            true_masks = true_masks.view(B * K, 1, 7, H, W)      # [N,1,7,H,W]

            logits = net(imgs)                                   # [N,1,7,H,W]
            if logits.shape != true_masks.shape:
                raise ValueError(
                    f"Shape mismatch: logits {tuple(logits.shape)} vs target {tuple(true_masks.shape)}"
                )

            prob = torch.sigmoid(logits)
            pred = (prob > threshold).float()

            tot += dice_per_slice_mean(pred, true_masks).item() * B  # sum over batch
            pbar.update()

    net.train()
    return tot / max(total_samples, 1)
