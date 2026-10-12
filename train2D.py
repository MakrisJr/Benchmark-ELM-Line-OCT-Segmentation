'''
This code was written by:
Anthos Makris

And based on the code by:
Dr. Vivek Kumar Singh
Department of Computer Science
Newcastle University, United Kingdom
Date: 24/August/2021

Also, thanks to "https://github.com/milesial/" for utilzing some of their codes.

'''
import argparse
import gc
import logging
import os
from pyexpat import model
import sys
import copy
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torchvision import transforms
from elm.eval import eval_net
from elm.model import U_Net,AttU_Net,LinkNetImprove,U2NETP,R2U_Net,DeepLabv3_plus,FCN,SegNet, SwinEncoderUNet2D, RETFoundSeg2D
from elm.retfound import DECODERS, apply_freeze_strategy, param_groups_llrd
from elm.fm_encoders import ENCODER_SOURCES, _resolve_checkpoint, encoder_arch
from elm import sizing
from elm.transformation import ELM_transform
from tensorboardX import SummaryWriter
from elm.dataset import BasicDataset, make_2d_transforms
from torch.utils.data import DataLoader, random_split
from elm.dice_loss import dice_coeff, dice_loss, cldice_loss, apply_valid_mask, masked_bce_with_logits
import torch.nn.functional as F
from efficientunet import *
import matplotlib.pyplot as plt
import csv

n_classes =1
n_channels = 3

def train_net(net,
              device,
              epochs=15,
              batch_size=4,
              lr=0.0001,
              save_cp=True,
              img_scale=1,
              args=None):

    if args is None:
        raise ValueError("Arguments 'args' cannot be None")

    model_name = args.model_name
    base_dir = args.base_dir
    data_root = os.path.join(base_dir, args.dataset_dir)
    dir_checkpoint = os.path.join(args.experiment_dir, 'checkpoints')

    transform_train, transform_val, ragged_val = build_transforms(args, net)

    # Only the training split shrinks. Validation and test stay at full size or the
    # curve measures evaluation noise on top of the effect it is trying to isolate.
    train_dataset = BasicDataset(
        root_dir=data_root,
        split="train",
        fold=args.fold,
        scale=img_scale,
        transform=transform_train,
        single_channel=False,
        subset_frac=getattr(args, 'train_frac', 1.0),
        subset_seed=getattr(args, 'train_frac_seed', 0),
    )

    val_dataset = BasicDataset(
        root_dir=data_root,
        split="val",
        fold=args.fold,
        scale=img_scale,
        transform=transform_val,
        single_channel=False
    )

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # The per-image padding policy (§3.3) gives every validation slice its own
    # canvas, so those batches are ragged and can only be stacked one at a time.
    val_loader = DataLoader(
        val_dataset,
        batch_size=1 if ragged_val else batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    writer = SummaryWriter(
        logdir=os.path.join(args.experiment_dir, 'logs'),
        comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}_{model_name}_fold_{args.fold}'
    )

    print('TensorBoard logs writing to:', writer.logdir)
    global_step = 0

    logging.info(f'''Starting training:
        Fold:            {args.fold}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Experiment dir:  {args.experiment_dir}
    ''')

    is_vit_ft = net.__class__.__name__ == "RETFoundSeg2D"
    if is_vit_ft:
        logging.info(apply_freeze_strategy(net, args.freeze_strategy, args.partial_from))
        # Layer-wise LR decay: the pretrained ViT-L blocks train far slower than the
        # freshly initialised decoder, which is what keeps fine-tuning on ~3.4k slices
        # from washing out the foundation-model features.
        optimizer = torch.optim.AdamW(
            param_groups_llrd(
                net, lr=lr,
                weight_decay=args.weight_decay,
                layer_decay=args.layer_decay,
                verbose=True,
            ),
            betas=(0.9, 0.999),
        )
    elif getattr(args, 'model', None) == "SwinEncoderUNet2D" or net.__class__.__name__ == "SwinEncoderUNet2D":
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(
            net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-9
        )

    # ViT fine-tuning wants warmup + cosine rather than plateau-based decay: the
    # first few hundred steps on a 300M-parameter pretrained encoder are exactly where
    # an un-warmed LR destroys the representation.
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    scheduler = None if is_vit_ft else optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=10
    )

    criterion_dice = dice_loss

    best_score = float('-inf')
    best_epoch = -1
    best_model_wts = copy.deepcopy(net.state_dict())
    epochs_since_best = 0

    accum = max(int(getattr(args, 'grad_accum', 1)), 1)
    # bf16 has the same exponent range as fp32, so ViT fine-tuning needs no loss
    # scaler; fp16 would. Restricted to CUDA because bf16 autocast on CPU is slow
    # and unnecessary.
    amp_enabled = bool(getattr(args, 'amp', False)) and device.type == 'cuda'

    for epoch in range(epochs):
        if is_vit_ft:
            set_warmup_cosine_lr(optimizer, base_lrs, epoch, epochs, args.warmup_epochs)
        net.train()
        epoch_loss = 0.0
        # Tracked so the per-fold train/val Dice gap is visible: effective n here is
        # 99 eyes, not 4851 slices, and a 300M-parameter ViT-L will memorise ~69
        # training eyes if nobody is watching (plan §8.1).
        train_dice_sum, train_dice_n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        with tqdm(total=n_train, desc=f'Fold {args.fold} | Epoch {epoch + 1}/{epochs}', disable= not sys.stdout.isatty(), unit='img') as pbar:
            for ibatch, batch in enumerate(train_loader):
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)
                valid = batch['valid'].to(device=device) if 'valid' in batch else None

                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                    masks_pred = net(imgs)
                masks_pred = masks_pred.float()
                out_new = torch.sigmoid(masks_pred)
                # Padding is not retina: it must not enter either loss term, or the
                # loss silently depends on how much padding a sample needed, which
                # varies systematically between the two size cohorts.
                masked_prob, masked_gt = apply_valid_mask(out_new, true_masks, valid)

                loss = (args.bce_weight * masked_bce_with_logits(masks_pred, true_masks, valid)
                        + args.dice_weight * criterion_dice(masked_prob, masked_gt))
                if args.cldice_weight > 0:
                    loss = loss + args.cldice_weight * cldice_loss(masked_prob, masked_gt, iters=args.cldice_iters)
                epoch_loss += float(loss.item())

                with torch.no_grad():
                    train_dice_sum += float(dice_coeff((masked_prob > 0.5).float(), masked_gt).item()) * imgs.shape[0]
                    train_dice_n += imgs.shape[0]

                (loss / accum).backward()
                if (ibatch + 1) % accum == 0 or (ibatch + 1) == len(train_loader):
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])
                global_step += 1

                if epoch == 0 and ibatch == 1:
                    print_gpu_mem(device)

        writer.add_scalar('Loss/train_epoch', epoch_loss / max(n_train, 1), epoch)

        # Validation once per epoch is cleaner for CV
        val_score = eval_net(net, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_score)

        train_dice = train_dice_sum / max(train_dice_n, 1)
        gap = train_dice - val_score
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Dice/val', val_score, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/train_val_gap', gap, epoch)

        logging.info(
            f'Fold {args.fold} | Epoch {epoch}: val Dice = {val_score:.6f}, '
            f'train Dice = {train_dice:.6f}, gap = {gap:+.4f}'
        )
        if gap > args.gap_warn_threshold:
            logging.warning(
                f'Fold {args.fold} | Epoch {epoch}: train/val Dice gap {gap:.3f} exceeds '
                f'{args.gap_warn_threshold:.3f}. On ~69 training eyes this is memorisation; '
                f'prefer --freeze-strategy partial over piling on regularisation (plan §8.1).'
            )

        if val_score > best_score:
            best_epoch = epoch
            best_score = val_score
            best_model_wts = copy.deepcopy(net.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if args.early_stop_patience > 0 and epochs_since_best >= args.early_stop_patience:
                logging.info(
                    f'Fold {args.fold}: early stop at epoch {epoch}; no val improvement '
                    f'for {epochs_since_best} epochs (best {best_score:.6f} at {best_epoch}).'
                )
                break

    if save_cp:
        os.makedirs(dir_checkpoint, exist_ok=True)
        net.load_state_dict(best_model_wts)
        ckpt_path = os.path.join(dir_checkpoint, f'{model_name}_fold_{args.fold}_best_epoch_{best_epoch}.pth')
        torch.save(net.state_dict(), ckpt_path)
        logging.info(f'Checkpoint saved to {ckpt_path} at epoch {best_epoch}')

    writer.close()

    # Release this fold's working set before returning. Refcounting would reclaim
    # most of it anyway, but the peak that matters is the overlap between one fold's
    # teardown and the next fold's construction -- and torch.save has just staged a
    # ~1.2GB host copy of the checkpoint, which would otherwise still be resident
    # when build_model allocates the next ViT-L and torch.loads its encoder weights.
    del best_model_wts, optimizer, scheduler
    del train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return best_score, best_epoch

def build_transforms(args, net=None):
    """Return (train_transform, val_transform, ragged_val) for the sizing policy.

    ``square`` is the benchmark's fixed 256x256 resize, kept so the foundation-model
    arms can be compared like-for-like against the ten architectures already
    published on this dataset. ``native`` and ``sam`` are plan §3.3: they exist
    because a fixed square resize on a dataset with a 4x aspect-ratio spread turns
    cohort membership into an image artefact.
    """
    policy = getattr(args, 'sizing', 'square')
    if policy == 'square':
        input_size = getattr(args, 'input_size', 256)
        return (
            make_2d_transforms(train=True, out_size=(input_size, input_size)),
            make_2d_transforms(train=False, out_size=(input_size, input_size)),
            False,
        )

    # ViT arms constrain the input to their patch size; the convolutional models have
    # no patch_size but do have five pooling stages, so 32 is the safe divisor for
    # them. Getting this wrong shows up as an off-by-one skip-connection shape deep
    # in a decoder, which is a miserable thing to debug on a compute node.
    patch = getattr(net, 'patch_size', None) or getattr(args, 'size_divisor', None) or 32
    crop = tuple(args.crop_size) if getattr(args, 'crop_size', None) else sizing.default_crop(patch)
    # MIRAGE's B-scan adapter expects values in [0, 1], not ImageNet-standardised ones.
    # The encoder declares its own requirement so this cannot drift out of sync; the
    # mistake would be invisible in every shape and merely degrade the features.
    norm = getattr(getattr(net, 'encoder', None), 'INPUT_RANGE', 'imagenet')
    if norm != 'imagenet':
        logging.info(f'encoder requires {norm!r} input normalisation')
    if policy == 'sam':
        return (
            sizing.SamResizePadTransform(size=args.sam_size, augment=True, normalize=norm),
            sizing.SamResizePadTransform(size=args.sam_size, augment=False, normalize=norm),
            False,
        )
    if policy == 'native':
        logging.info(f'native sizing: patch {patch}, training crop {crop[0]}x{crop[1]}')
        return (
            sizing.NativeCropTransform(crop_size=crop, divisor=patch, normalize=norm),
            sizing.NativePadTransform(divisor=patch, normalize=norm),
            True,
        )
    raise ValueError(f"unknown --sizing '{policy}'")


def run_config(args, net, fold, experiment_dir):
    """Everything needed to reproduce this run, written next to the checkpoints.

    Plan §9.5: config, git SHA, seed, package versions and the *resolved* checkpoint
    path. "RETFound" is not a reproducible statement on its own -- which file was
    actually loaded is.
    """
    import platform
    import subprocess

    def git_sha():
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return None

    resolved = None
    if args.model == 'RETFoundSeg2D' and args.encoder_weights not in ('none', 'imagenet'):
        try:
            resolved = _resolve_checkpoint(args.encoder_weights)
        except Exception as exc:
            resolved = f'<unresolved: {exc}>'

    import timm
    return {
        'args': {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
        'fold': fold,
        'experiment_dir': experiment_dir,
        'git_sha': git_sha(),
        'seed': args.seed,
        'encoder_checkpoint': resolved,
        'trainable_params': sum(p.numel() for p in net.parameters() if p.requires_grad),
        'total_params': sum(p.numel() for p in net.parameters()),
        'versions': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'timm': timm.__version__,
            'numpy': np.__version__,
        },
    }


def set_seed(seed: int):
    """Seed every source of randomness the training loop touches."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser(
        description='2D ELM line segmentation from OCT images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model',
        type=str,
        default='SegNet',
        choices=[
            'SegNet',
            'U_Net',
            'AttU_Net',
            'LinkNetImprove',
            'U2NETP',
            'R2U_Net',
            'DeepLabv3_plus',
            'FCN',
            'SwinEncoderUNet2D',
            'RETFoundSeg2D',
        ],
        help='Model architecture to train'
    )
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default='',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-d', '--base-dir', dest='base_dir', type=str, default='./',
                        help='Base files directory')
    parser.add_argument('--dataset-dir', dest='dataset_dir', type=str, default='data_no_anomalies',
                        help='Dataset directory name under base-dir (e.g. data_no_anomalies, data_no_anomalous_slices)')
    parser.add_argument('--run-name', dest='run_name', type=str, default=None,
                        help='Name of the elm-results/ output dir for this run. Defaults to '
                             '<model>_<timestamp>_model. The default timestamp only has minute '
                             'resolution, so pass an explicit name when launching concurrent runs '
                             'of the same model, both to avoid a collision and so a caller can know '
                             'the output path up front instead of discovering it afterwards.')

    parser.add_argument('--fold', type=int, default=0,
                        help='Validation fold index (0..4)')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of CV folds')
    # BooleanOptionalAction so --no-run-all-folds actually exists: with a plain
    # store_true defaulting to True there was no way to run a single fold, which the
    # LR sweep (plan §8) needs.
    parser.add_argument('--run-all-folds', action=argparse.BooleanOptionalAction, default=True,
                        help='Run all folds sequentially. --no-run-all-folds runs only --fold.')

    parser.add_argument('--bce-weight', type=float, default=0.5,
                        help='Weight of the BCE term in the training loss')
    parser.add_argument('--dice-weight', type=float, default=0.5,
                        help='Weight of the Dice term in the training loss')
    parser.add_argument('--cldice-weight', type=float, default=0.0,
                        help='Weight of the clDice term in the training loss (0 = disabled, opt-in)')
    parser.add_argument('--cldice-iters', type=int, default=3,
                        help='Number of soft-skeletonization iterations for the clDice term')

    # --- RETFoundSeg2D / foundation-model fine-tuning ---
    parser.add_argument('--input-size', dest='input_size', type=int, default=256,
                        help='Square size images are resized to for training. 256 keeps '
                             'parity with every other model in the benchmark; raising it '
                             'to 512 quadruples a ViT patch grid (16x16 -> 32x32), which '
                             'is the main lever against thin-structure blurring. Must '
                             'match --input_size at evaluation time.')
    parser.add_argument('--encoder-weights', dest='encoder_weights', type=str, default='retfound_oct',
                        choices=sorted(ENCODER_SOURCES),
                        help='Pretrained encoder for RETFoundSeg2D. retfound_oct is the '
                             'headline arm; retfound_cfp (same lab, same objective, wrong '
                             'modality), mae_in1k (same architecture and MAE objective, '
                             'natural images), imagenet (supervised rather than SSL) and '
                             'none (identical architecture, random init) are the Claim A '
                             'controls. mirage_* and medsam differ in several respects at '
                             'once and belong only in the Claim B table.')
    parser.add_argument('--decoder', type=str, default='dpt', choices=list(DECODERS) + ['linear'],
                        help="Dense-prediction decoder on top of the ViT. 'dpt' is the "
                             "multi-scale fusion decoder; 'convnext' with --freeze-strategy "
                             "frozen is the MIRAGE benchmark protocol, the only configuration "
                             "directly comparable to their published RETFound row; "
                             "'linear_probe' is the floor. 'linear' is a deprecated alias for "
                             "'linear_probe'.")
    parser.add_argument('--no-stem', dest='use_stem', action='store_false', default=True,
                        help='Drop the full-resolution convolutional stem from the DPT '
                             'decoder and upsample bilinearly from the 1/2 stage instead. '
                             'Separate from --decoder so the ablation can attribute the '
                             'difference to the stem alone.')
    parser.add_argument('--freeze-strategy', dest='freeze_strategy', type=str, default='full',
                        choices=['full', 'partial', 'frozen'],
                        help="Which encoder parameters train. 'full' (expected primary "
                             "configuration, given the Topcon->Spectralis domain shift); "
                             "'partial' freezes the patch embed and the first "
                             "--partial-from blocks; 'frozen' trains the decoder only and "
                             "gives the numbers comparable to MIRAGE's published row.")
    parser.add_argument('--partial-from', dest='partial_from', type=int, default=8,
                        help='With --freeze-strategy partial, the first trainable block index')
    parser.add_argument('--freeze-encoder', dest='freeze_encoder', action='store_true',
                        help='Deprecated alias for --freeze-strategy frozen.')

    # --- input sizing (plan §3.3) ---
    parser.add_argument('--sizing', type=str, default='square', choices=['square', 'native', 'sam'],
                        help="Input sizing policy. 'square' is the benchmark's fixed "
                             '--input-size resize, kept for like-for-like comparison with the '
                             'ten published architectures. \'native\' trains on native-scale '
                             'crops and evaluates with per-image padding, which is the only '
                             'policy that does not make cohort membership an image artefact on '
                             'a dataset with a 4x aspect-ratio spread. \'sam\' is canonical '
                             'SAM resize-longest-side-and-pad, for the MedSAM arm.')
    parser.add_argument('--crop-size', dest='crop_size', type=int, nargs=2, default=None,
                        metavar=('H', 'W'),
                        help='Training crop for --sizing native; each axis must be a multiple '
                             "of the encoder patch size. Defaults to 384x384 for every "
                             "encoder -- nnU-Net's fingerprint-derived patch size for this "
                             'dataset, which pads the two size cohorts almost equally (81.9%% '
                             'vs 80.5%% real pixels). Override only for the resolution '
                             'ablation; see elm/sizing.py for why a landscape crop is worse.')
    parser.add_argument('--sam-size', dest='sam_size', type=int, default=1024,
                        help='Canvas size for --sizing sam')
    parser.add_argument('--size-divisor', dest='size_divisor', type=int, default=None,
                        help='Override the padding/crop divisor for --sizing native. '
                             'Defaults to the encoder patch size for ViT models and 32 '
                             'for the convolutional ones.')

    # --- optimisation (plan §8) ---
    parser.add_argument('--grad-accum', dest='grad_accum', type=int, default=1,
                        help='Gradient accumulation steps. Set batch x grad-accum to the '
                             'effective batch of 16 the protocol fixes, so results do not '
                             'depend on how much memory a given GPU happened to have.')
    parser.add_argument('--amp', action='store_true',
                        help='bf16 autocast. Same exponent range as fp32, so no loss scaler.')
    parser.add_argument('--early-stop-patience', dest='early_stop_patience', type=int, default=0,
                        help='Stop after this many epochs without a val-Dice improvement '
                             '(0 disables). The protocol uses 8 with 40 epochs.')
    parser.add_argument('--gap-warn-threshold', dest='gap_warn_threshold', type=float, default=0.05,
                        help='Log a warning when the per-fold train/val Dice gap exceeds this. '
                             'Effective n is 99 eyes, not 4851 slices.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for torch/numpy/random. Report the spread over three seeds '
                             'for the headline arms.')
    parser.add_argument('--train-frac', dest='train_frac', type=float, default=1.0,
                        help='Train on this fraction of the training PATIENTS (default 1.0 = all). '
                             'For learning curves: subsets are nested across fractions at a fixed '
                             '--train-frac-seed, so 0.25 is a subset of 0.5 is a subset of 1.0. '
                             'Validation and test are never subsetted.')
    parser.add_argument('--train-frac-seed', dest='train_frac_seed', type=int, default=0,
                        help='Seed for which patients --train-frac keeps. Vary this to check a '
                             'learning-curve point is not an artefact of one particular draw.')
    parser.add_argument('--layer-decay', dest='layer_decay', type=float, default=0.75,
                        help="Layer-wise LR decay for the ViT encoder. 0.75 is MAE's ViT-L "
                             "setting; RETFound's repo default of 0.65 was tuned for ViT-B "
                             "and effectively freezes the stem across 24 blocks.")
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=0.05,
                        help='AdamW weight decay for the ViT fine-tuning path')
    parser.add_argument('--drop-path', dest='drop_path', type=float, default=0.2,
                        help="Stochastic depth rate for the ViT encoder. 0.2 is MAE's ViT-L "
                             "recipe and RETFound's own --drop_path setting.")
    parser.add_argument('--warmup-epochs', dest='warmup_epochs', type=int, default=5,
                        help='Linear LR warmup epochs before cosine decay (ViT path only)')

    return parser.parse_args()


def set_warmup_cosine_lr(optimizer, base_lrs, epoch, total_epochs, warmup_epochs):
    """Linear warmup then cosine decay, applied per param group.

    Each group keeps its own layer-wise-decayed base LR; this only scales them.
    """
    if warmup_epochs > 0 and epoch < warmup_epochs:
        factor = (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        factor = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    for g, base in zip(optimizer.param_groups, base_lrs):
        g['lr'] = base * factor
    return factor


def mb(x): return x / 1024**2

def host_rss_mb():
    """Resident host memory of this process, in MB, or None off Linux.

    Logged at every fold boundary because host RAM -- not GPU -- is what killed the
    first phase-2 sweep: each fold rebuilds a 304M-parameter ViT-L and torch.loads a
    ~1.2GB encoder checkpoint, and a ratchet across folds is invisible until the OOM
    killer arrives with no traceback.
    """
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def log_host_rss(tag):
    rss = host_rss_mb()
    if rss is not None:
        logging.info(f'host RSS {tag}: {rss:.0f} MB')


def print_gpu_mem(device=None, prefix=''):
    if not torch.cuda.is_available():
        print(prefix + 'No CUDA device')
        return
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)        # memory currently allocated by tensors
    reserved  = torch.cuda.memory_reserved(device)         # memory managed by the caching allocator
    max_alloc  = torch.cuda.max_memory_allocated(device)   # peak allocated by tensors
    print(f"{prefix}GPU {device} allocated: {mb(allocated):.1f} MB, "
          f"reserved: {mb(reserved):.1f} MB, peak_alloc: {mb(max_alloc):.1f} MB")
    # optional: a readable summary
    print(torch.cuda.memory_summary(device=device, abbreviated=True))

def normalize_decoder(name: str) -> str:
    """Map the retired 'linear' label onto 'linear_probe'.

    The old name implied it was "the MIRAGE benchmark protocol". It was not: MIRAGE
    benchmarked a ConvNeXt decoder on a frozen encoder, and used a 1x1-conv probe
    only in their pretraining ablations. Renamed so the results table cannot repeat
    that claim by accident.
    """
    if name == 'linear':
        logging.warning("--decoder linear is deprecated; using 'linear_probe'")
        return 'linear_probe'
    return name


def build_model(model_name: str = 'SegNet', args=None):
    if model_name == 'SegNet':
        return SegNet(n_channels=3, n_classes=1)
    if model_name == 'U_Net':
        return U_Net(n_channels=3, n_classes=1)
    if model_name == 'AttU_Net':
        return AttU_Net(n_channels=3, n_classes=1)
    if model_name == 'LinkNetImprove':
        return LinkNetImprove(n_channels=3, n_classes=1)
    if model_name == 'U2NETP':
        return U2NETP(n_channels=3, n_classes=1)
    if model_name == 'R2U_Net':
        return R2U_Net(n_channels=3, n_classes=1, t=2)
    if model_name == 'DeepLabv3_plus':
        return DeepLabv3_plus(n_channels=3, n_classes=1, os=16, pretrained=True, _print=True)
    if model_name == 'FCN':
        return FCN(n_channels=3, n_classes=1)
    if model_name == 'SwinEncoderUNet2D':
        return SwinEncoderUNet2D(
            n_channels=3,
            n_classes=1,
            backbone="swin_tiny_patch4_window7_224",
            pretrained=True,
        )
    if model_name == 'RETFoundSeg2D':
        encoder_weights = getattr(args, 'encoder_weights', 'retfound_oct')
        # img_size only sizes the initial position-embedding grid; the ViT arms are
        # built with dynamic_img_size, so any patch-multiple rectangle works later.
        if getattr(args, 'sizing', 'square') == 'square':
            img_size = getattr(args, 'input_size', 256)
        elif args.sizing == 'sam':
            img_size = args.sam_size
        else:
            patch = 32 if encoder_arch(encoder_weights) == 'mirage' else 16
            img_size = tuple(args.crop_size) if args.crop_size else sizing.default_crop(patch)
        return RETFoundSeg2D(
            n_channels=3,
            n_classes=1,
            img_size=img_size,
            encoder_weights=encoder_weights,
            decoder=normalize_decoder(getattr(args, 'decoder', 'dpt')),
            use_stem=getattr(args, 'use_stem', True),
            freeze_encoder=(getattr(args, 'freeze_strategy', 'full') == 'frozen'),
            drop_path_rate=getattr(args, 'drop_path', 0.2),
        )

    raise ValueError(f"Unknown model '{model_name}'")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    args.decoder = normalize_decoder(args.decoder)
    if args.freeze_encoder and args.freeze_strategy == 'full':
        args.freeze_strategy = 'frozen'
    if args.encoder_weights == 'medsam' and args.sizing == 'square':
        logging.warning(
            'MedSAM was pretrained under resize-longest-side-to-1024-and-pad; '
            'pass --sizing sam for its primary configuration (--sizing native is the ablation).'
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    timestamp = time.strftime("%b-%d-%Y_%H%M")

    def run_single_fold(fold_idx):
        # Reseeded per fold so a fold is reproducible in isolation, not only as part
        # of a full 5-fold sweep.
        set_seed(args.seed + fold_idx)
        net = build_model(args.model, args)
        net.to(device=device)

        model_name = args.run_name or f'{args.model}_{timestamp}_model'
        experiment_dir = os.path.join(args.base_dir, 'elm-results', model_name, f'fold_{fold_idx}')
        os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)

        # Record what was built, so evaluation can reconstruct the exact architecture
        # instead of relying on the caller passing matching flags months later.
        with open(os.path.join(experiment_dir, 'model_config.json'), 'w') as f:
            json.dump({
                'model': args.model,
                'input_size': args.input_size,
                'encoder_weights': args.encoder_weights,
                'decoder': args.decoder,
                'use_stem': args.use_stem,
                'freeze_strategy': args.freeze_strategy,
                'freeze_encoder': args.freeze_strategy == 'frozen',
                'sizing': args.sizing,
                'crop_size': list(args.crop_size) if args.crop_size else None,
                'sam_size': args.sam_size,
                'size_divisor': args.size_divisor,
                'dataset_dir': args.dataset_dir,
            }, f, indent=2)

        with open(os.path.join(experiment_dir, 'run_config.json'), 'w') as f:
            json.dump(run_config(args, net, fold_idx, experiment_dir), f, indent=2, default=str)

        run_args = copy.deepcopy(args)
        run_args.fold = fold_idx
        run_args.model_name = model_name
        run_args.experiment_dir = experiment_dir

        logging.info(f'Experiment dir: {experiment_dir}')

        if run_args.load:
            net.load_state_dict(torch.load(run_args.load, map_location=device))
            logging.info(f'Model loaded from {run_args.load}')

        best_score, best_epoch = train_net(
            net=net,
            epochs=run_args.epochs,
            batch_size=run_args.batchsize,
            lr=run_args.lr,
            device=device,
            img_scale=run_args.scale,
            args=run_args
        )

        # The next fold builds its own encoder from scratch, so nothing here is
        # reused. Dropping it explicitly keeps two ViT-Ls from ever being resident
        # at once.
        del net
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        log_host_rss(f'after fold {fold_idx}')

        return best_score, best_epoch, experiment_dir

    try:
        if args.run_all_folds:
            cv_results = []
            for fold_idx in range(args.num_folds):
                logging.info(f'===== Starting fold {fold_idx}/{args.num_folds - 1} =====')
                log_host_rss(f'before fold {fold_idx}')
                best_score, best_epoch, exp_dir = run_single_fold(fold_idx)
                cv_results.append({
                    "fold": fold_idx,
                    "best_dice": best_score,
                    "best_epoch": best_epoch,
                    "experiment_dir": exp_dir,
                })

            results_csv = os.path.join(args.base_dir, 'elm-results', f'cv_results_{args.run_name or timestamp}.csv')
            os.makedirs(os.path.dirname(results_csv), exist_ok=True)

            with open(results_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["fold", "best_dice", "best_epoch", "experiment_dir"])
                writer.writeheader()
                writer.writerows(cv_results)

            scores = [r["best_dice"] for r in cv_results]
            logging.info(f'CV results saved to {results_csv}')
            logging.info(f'Mean Dice: {np.mean(scores):.6f}')
            logging.info(f'Std Dice:  {np.std(scores):.6f}')

        else:
            best_score, best_epoch, exp_dir = run_single_fold(args.fold)
            logging.info(f'Fold {args.fold} best Dice: {best_score:.6f} at epoch {best_epoch}')

    except KeyboardInterrupt:
        logging.info('Training interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
