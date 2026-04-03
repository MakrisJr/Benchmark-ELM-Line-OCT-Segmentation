"""
3D ELM line segmentation training with 5-fold cross validation.

This combines:
- the fold orchestration style from train.py
- the 3D model and optimizer handling from the earlier new-train.py
- the metadata-based split logic used across the newer dataset code
"""

import argparse
import copy
import csv
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from elm.dataset import D3Dataset
from elm.dice_loss import dice_loss
from elm.eval import eval_net
from elm.model import (
    CSAM_UNet2p5D,
    SwinUNETR3D,
    UNet2DEnc3DDec,
    UNet2p5D_SlidingWindow,
    UNet3D,
    UNet3D_Aniso,
    UNet3D_Aniso2,
    UNet3DFrawley,
)


def mb(x):
    return x / 1024**2


def print_gpu_mem(device=None, prefix=""):
    if not torch.cuda.is_available():
        print(prefix + "No CUDA device")
        return
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_alloc = torch.cuda.max_memory_allocated(device)
    print(
        f"{prefix}GPU {device} allocated: {mb(allocated):.1f} MB, "
        f"reserved: {mb(reserved):.1f} MB, peak_alloc: {mb(max_alloc):.1f} MB"
    )
    print(torch.cuda.memory_summary(device=device, abbreviated=True))


def build_model(args):
    if args.model == "UNet3D":
        return UNet3D(in_channels=1, out_channels=1)
    if args.model == "UNet3D_Aniso":
        return UNet3D_Aniso(in_channels=1, out_channels=1)
    if args.model == "UNet3D_Aniso2":
        return UNet3D_Aniso2(in_channels=1, out_channels=1)
    if args.model == "UNet3DFrawley":
        return UNet3DFrawley(in_channels=1, out_channels=1)
    if args.model == "UNet2DEnc3DDec":
        return UNet2DEnc3DDec(in_channels=1, out_channels=1)
    if args.model == "CSAM_UNet2p5D":
        return CSAM_UNet2p5D(
            in_channels=1,
            out_channels=1,
            num_layers=3,
            base_num=32,
            semantic=True,
            positional=True,
            slice_att=True,
        )
    if args.model == "UNet2p5D_SlidingWindow":
        return UNet2p5D_SlidingWindow(
            k=args.window_k,
            out_channels=1,
            num_layers=3,
            base_num=32,
            pad_mode="replicate",
        )
    if args.model == "SwinUNETR3D":
        pretrained_path = args.pretrained_path
        if pretrained_path and not os.path.exists(pretrained_path):
            logging.warning(
                f"Pretrained path {pretrained_path} was not found. "
                "Continuing without pretrained encoder weights."
            )
            pretrained_path = None
        return SwinUNETR3D(
            in_channels=1,
            n_classes=1,
            pretrained_path=pretrained_path,
        )
    raise ValueError(f"Unknown 3D model '{args.model}'")


def build_optimizer(net, lr):
    if net.__class__.__name__.startswith("SwinUNETR"):
        encoder_params = []
        decoder_params = []
        for name, param in net.model.named_parameters():
            if name.startswith("swinViT"):
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        logging.info(
            f"SwinUNETR parameter groups: encoder={len(encoder_params)} decoder={len(decoder_params)}"
        )
        return torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": 1e-5},
                {"params": decoder_params, "lr": 1e-4},
            ],
            weight_decay=1e-5,
        )

    return optim.Adam(
        net.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-9,
    )


def make_dataloaders(args):
    data_root = os.path.join(args.base_dir, "data_no_anomalies")
    train_dataset = D3Dataset(
        root_dir=data_root,
        split="train",
        fold=args.fold,
        scale=args.scale,
        transform=True,
    )
    val_dataset = D3Dataset(
        root_dir=data_root,
        split="val",
        fold=args.fold,
        scale=args.scale,
        transform=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def train_net(net, device, args):
    train_dataset, val_dataset, train_loader, val_loader = make_dataloaders(args)
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    writer = SummaryWriter(
        logdir=os.path.join(args.experiment_dir, "logs"),
        comment=f"LR_{args.lr}_BS_{args.batchsize}_{args.model}_fold_{args.fold}",
    )
    logging.info(f"TensorBoard logs writing to: {writer.logdir}")

    logging.info(
        f"""Starting training:
        Fold:            {args.fold}
        Epochs:          {args.epochs}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Images scaling:  {args.scale}
        Experiment dir:  {args.experiment_dir}
    """
    )

    optimizer = build_optimizer(net, args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=10
    )
    criterion_bce = nn.BCEWithLogitsLoss().to(device=device)

    csv_path = os.path.join(args.experiment_dir, "training_log.csv")
    with open(csv_path, mode="w", newline="") as csv_file:
        writer_csv = csv.writer(csv_file)
        writer_csv.writerow(["epoch", "train_loss", "val_dice", "learning_rate"])

    global_step = 0
    best_score = float("-inf")
    best_epoch = -1
    best_model_wts = copy.deepcopy(net.state_dict())

    for epoch in range(args.epochs):
        net.train()
        epoch_loss = 0.0

        with tqdm(
            total=n_train,
            desc=f"Fold {args.fold} | Epoch {epoch + 1}/{args.epochs}",
            disable=not sys.stdout.isatty(),
            unit="vol",
        ) as pbar:
            for ibatch, batch in enumerate(train_loader):
                imgs = batch["image"].to(device=device, dtype=torch.float32)
                true_masks = batch["mask"].to(device=device, dtype=torch.float32)

                masks_pred = net(imgs)
                prob = torch.sigmoid(masks_pred)
                loss = 0.5 * criterion_bce(masks_pred, true_masks) + 0.5 * dice_loss(
                    prob, true_masks
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                writer.add_scalar("Loss/train_batch", loss.item(), global_step)
                pbar.set_postfix(**{"loss (batch)": loss.item()})
                pbar.update(imgs.shape[0])
                global_step += 1

                if epoch == 0 and ibatch == 1:
                    print_gpu_mem(device)

        epoch_loss_avg = epoch_loss / max(1, len(train_loader))
        val_score = eval_net(net, val_loader, device)
        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train_epoch", epoch_loss_avg, epoch)
        writer.add_scalar("Dice/val_epoch", val_score, epoch)
        writer.add_scalar("learning_rate_epoch", current_lr, epoch)

        if "imgs" in locals():
            B, C, D, H, W = imgs.shape
            imgs_to_write = imgs.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            true_masks_to_write = true_masks.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)
            masks_pred_to_write = masks_pred.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)
            writer.add_images("images", imgs_to_write, epoch)
            writer.add_images("masks/true", true_masks_to_write, epoch)
            writer.add_images(
                "masks/pred",
                (torch.sigmoid(masks_pred_to_write) > 0.5).float(),
                epoch,
            )

        if (epoch + 1) % 5 == 0:
            for tag, value in net.named_parameters():
                tag = tag.replace(".", "/")
                writer.add_histogram("weights/" + tag, value.data.cpu().numpy(), epoch)

        if val_score > best_score:
            best_score = float(val_score)
            best_epoch = epoch
            best_model_wts = copy.deepcopy(net.state_dict())

        with open(csv_path, mode="a", newline="") as csv_file:
            writer_csv = csv.writer(csv_file)
            writer_csv.writerow([epoch, epoch_loss_avg, float(val_score), current_lr])

        logging.info(
            f"Fold {args.fold} | Epoch {epoch}/{args.epochs - 1} | "
            f"TrainLoss={epoch_loss_avg:.6f} | ValDice={float(val_score):.6f} | "
            f"LR={current_lr:.2e}"
        )

    os.makedirs(os.path.join(args.experiment_dir, "checkpoints"), exist_ok=True)
    net.load_state_dict(best_model_wts)
    ckpt_path = os.path.join(
        args.experiment_dir,
        "checkpoints",
        f"{args.model_name}_fold_{args.fold}_best_epoch_{best_epoch}.pth",
    )
    torch.save(net.state_dict(), ckpt_path)
    logging.info(f"Checkpoint saved to {ckpt_path}")

    writer.close()
    return best_score, best_epoch


def get_args():
    parser = argparse.ArgumentParser(
        description="3D ELM line segmentation from OCT volumes with 5-fold CV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SwinUNETR3D",
        choices=[
            "UNet3D",
            "UNet3D_Aniso",
            "UNet3D_Aniso2",
            "UNet3DFrawley",
            "UNet2DEnc3DDec",
            "CSAM_UNet2p5D",
            "UNet2p5D_SlidingWindow",
            "SwinUNETR3D",
        ],
        help="3D model architecture to train",
    )
    parser.add_argument(
        "-e", "--epochs", metavar="E", type=int, default=100, dest="epochs"
    )
    parser.add_argument(
        "-b", "--batch-size", metavar="B", type=int, default=2, dest="batchsize"
    )
    parser.add_argument(
        "-l", "--learning-rate", metavar="LR", type=float, default=0.0002, dest="lr"
    )
    parser.add_argument(
        "-f",
        "--load",
        dest="load",
        type=str,
        default="",
        help="Load model weights from a .pth file before training",
    )
    parser.add_argument(
        "-s",
        "--scale",
        dest="scale",
        type=float,
        default=1.0,
        help="Downscaling factor applied before the final resize",
    )
    parser.add_argument(
        "-d",
        "--base-dir",
        dest="base_dir",
        type=str,
        default="./",
        help="Base files directory",
    )
    parser.add_argument(
        "--fold", type=int, default=0, help="Validation fold index (0..4)"
    )
    parser.add_argument(
        "--num-folds", type=int, default=5, help="Number of CV folds"
    )
    parser.add_argument(
        "--run-all-folds",
        dest="run_all_folds",
        action="store_true",
        help="Run all folds sequentially",
    )
    parser.add_argument(
        "--single-fold",
        dest="run_all_folds",
        action="store_false",
        help="Train only the fold given by --fold",
    )
    parser.set_defaults(run_all_folds=True)
    parser.add_argument(
        "--num-workers", type=int, default=2, help="DataLoader worker count"
    )
    parser.add_argument(
        "--window-k",
        type=int,
        default=7,
        help="Sliding-window depth for UNet2p5D_SlidingWindow",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="./checkpoint/model_swinvit_UNETR.pt",
        help="Optional pretrained encoder weights for SwinUNETR3D",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    timestamp = time.strftime("%b-%d-%Y_%H%M")
    model_name = f"{args.model}_{timestamp}_model"
    model_root = os.path.join(args.base_dir, "elm-results", model_name)
    os.makedirs(model_root, exist_ok=True)

    def run_single_fold(fold_idx):
        net = build_model(args)
        net.to(device=device)

        experiment_dir = os.path.join(model_root, f"fold_{fold_idx}")
        os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)

        run_args = copy.deepcopy(args)
        run_args.fold = fold_idx
        run_args.model_name = model_name
        run_args.experiment_dir = experiment_dir

        logging.info(f"Experiment dir: {experiment_dir}")

        if run_args.load:
            net.load_state_dict(torch.load(run_args.load, map_location=device))
            logging.info(f"Model loaded from {run_args.load}")

        best_score, best_epoch = train_net(net=net, device=device, args=run_args)
        return best_score, best_epoch, experiment_dir

    try:
        if args.run_all_folds:
            cv_results = []
            for fold_idx in range(args.num_folds):
                logging.info(f"===== Starting fold {fold_idx}/{args.num_folds - 1} =====")
                best_score, best_epoch, exp_dir = run_single_fold(fold_idx)
                cv_results.append(
                    {
                        "fold": fold_idx,
                        "best_dice": best_score,
                        "best_epoch": best_epoch,
                        "experiment_dir": exp_dir,
                    }
                )

            results_csv = os.path.join(model_root, "cv_results.csv")
            with open(results_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["fold", "best_dice", "best_epoch", "experiment_dir"],
                )
                writer.writeheader()
                writer.writerows(cv_results)

            scores = [row["best_dice"] for row in cv_results]
            logging.info(f"CV results saved to {results_csv}")
            logging.info(f"Mean Dice: {np.mean(scores):.6f}")
            logging.info(f"Std Dice:  {np.std(scores):.6f}")
        else:
            best_score, best_epoch, exp_dir = run_single_fold(args.fold)
            logging.info(
                f"Fold {args.fold} best Dice: {best_score:.6f} at epoch {best_epoch} "
                f"({exp_dir})"
            )

    except KeyboardInterrupt:
        logging.info("Training interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
