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
import logging
import os
from pyexpat import model
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torchvision import transforms
from eval import eval_net
from model import U_Net,AttU_Net,LinkNetImprove,U2NETP,R2U_Net,DeepLabv3_plus,FCN,SegNet, SwinEncoderUNet2D
from transformation import ELM_transform
from tensorboardX import SummaryWriter
from dataset import BasicDataset, make_2d_transforms
from torch.utils.data import DataLoader, random_split
from dice_loss import dice_loss
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
    data_root = os.path.join(base_dir, "data_no_anomalies")
    dir_checkpoint = os.path.join(args.experiment_dir, 'checkpoints')

    transform_train = make_2d_transforms(train=True, out_size=(256, 256))
    transform_val = make_2d_transforms(train=False, out_size=(256, 256))

    train_dataset = BasicDataset(
        root_dir=data_root,
        split="train",
        fold=args.fold,
        scale=img_scale,
        transform=transform_train,
        single_channel=False
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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

    if getattr(args, 'model', None) == "SwinEncoderUNet2D" or net.__class__.__name__ == "SwinEncoderUNet2D":
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(
            net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-9
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=10
    )

    criterion_dice = dice_loss
    criterion = nn.BCEWithLogitsLoss().to(device)

    best_score = float('-inf')
    best_epoch = -1
    best_model_wts = copy.deepcopy(net.state_dict())

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0

        with tqdm(total=n_train, desc=f'Fold {args.fold} | Epoch {epoch + 1}/{epochs}', disable= not sys.stdout.isatty(), unit='img') as pbar:
            for ibatch, batch in enumerate(train_loader):
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                true_masks = batch['mask'].to(device=device, dtype=torch.float32)

                masks_pred = net(imgs)
                out_new = torch.sigmoid(masks_pred)

                loss = 0.5 * criterion(masks_pred, true_masks) + 0.5 * criterion_dice(out_new, true_masks)
                epoch_loss += float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])
                global_step += 1

                if epoch == 0 and ibatch == 1:
                    print_gpu_mem(device)

        writer.add_scalar('Loss/train_epoch', epoch_loss / max(n_train, 1), epoch)

        # Validation once per epoch is cleaner for CV
        val_score = eval_net(net, val_loader, device)
        scheduler.step(val_score)

        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Dice/val', val_score, epoch)

        logging.info(f'Fold {args.fold} | Epoch {epoch}: Validation Dice = {val_score:.6f}')

        if val_score > best_score:
            best_epoch = epoch
            best_score = val_score
            best_model_wts = copy.deepcopy(net.state_dict())

    if save_cp:
        os.makedirs(dir_checkpoint, exist_ok=True)
        net.load_state_dict(best_model_wts)
        ckpt_path = os.path.join(dir_checkpoint, f'{model_name}_fold_{args.fold}_best_epoch_{best_epoch}.pth')
        torch.save(net.state_dict(), ckpt_path)
        logging.info(f'Checkpoint saved to {ckpt_path} at epoch {best_epoch}')

    writer.close()
    return best_score, best_epoch

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

    parser.add_argument('--fold', type=int, default=0,
                        help='Validation fold index (0..4)')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--run-all-folds', action='store_true', default=True,
                        help='Run all folds sequentially')

    return parser.parse_args()


def mb(x): return x / 1024**2

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

def build_model(model_name: str = 'SegNet'):
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

    raise ValueError(f"Unknown model '{model_name}'")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    timestamp = time.strftime("%b-%d-%Y_%H%M")

    def run_single_fold(fold_idx):
        net = build_model(args.model)
        net.to(device=device)

        model_name = f'{args.model}_{timestamp}_model'
        experiment_dir = os.path.join(args.base_dir, 'elm-results', model_name, f'fold_{fold_idx}')
        os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)

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
        return best_score, best_epoch, experiment_dir

    try:
        if args.run_all_folds:
            cv_results = []
            for fold_idx in range(args.num_folds):
                logging.info(f'===== Starting fold {fold_idx}/{args.num_folds - 1} =====')
                best_score, best_epoch, exp_dir = run_single_fold(fold_idx)
                cv_results.append({
                    "fold": fold_idx,
                    "best_dice": best_score,
                    "best_epoch": best_epoch,
                    "experiment_dir": exp_dir,
                })

            results_csv = os.path.join(args.base_dir, 'elm-results', f'cv_results_{timestamp}.csv')
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