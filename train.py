'''
This code is written by:

Dr. Vivek Kumar Singh
Department of Computer Science
Newcastle University, United Kingdom
Date: 24/August/2021

Also, thanks to "https://github.com/milesial/" for utilzing some of their codes.

'''
import argparse
import logging
import os
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
from model import U_Net,AttU_Net,LinkNetImprove,U2NETP,R2U_Net,DeepLabv3_plus,FCN,SegNet
from transformation import ELM_transform
from tensorboardX import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from dice_loss import Dice_Loss
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
              val_percent=0.1,
              save_cp=True,
              img_scale=1,
              args=None):
    
    if args is None:
        raise ValueError("Arguments 'args' cannot be None")
    
    model_name = args.model_name
    base_dir = args.base_dir

    train_dir_img = os.path.join(base_dir, "data/train/image/")
    train_dir_mask = os.path.join(base_dir, "data/train/mask/")
    val_dir_img = os.path.join(base_dir, "data/val/image/")
    val_dir_mask = os.path.join(base_dir, "data/val/mask/")
    dir_checkpoint = os.path.join(base_dir, 'elm-results/', model_name + '/checkpoints/')
    
    transform = ELM_transform()
    train_dataset = BasicDataset(train_dir_img, train_dir_mask, img_scale, transform = transform['train'])
    val_dataset= BasicDataset(val_dir_img, val_dir_mask, img_scale, transform = transform['val'])

    n_train=len(train_dataset)
    n_val=len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)


    writer = SummaryWriter(logdir=os.path.join(args.experiment_dir, 'logs'),
                            comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}_{model_name}')
                           
    print('TensorBoard logs writing to:', writer.logdir)
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Experiment dir: {args.experiment_dir}
    ''')

# !!---------- Defined the optimizer --------------------------!!

    optimizer = optim.Adam(net.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=10)

# ------------------ Loss function ------------------!!
    criterion_dice = Dice_Loss
    criterion = nn.BCEWithLogitsLoss().cuda()
    
# !!-------------- Training and validation loop ------------------!!
    best_score = float('-inf')
    best_epoch = -1
    best_model_wts = copy.deepcopy(net.state_dict())
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for ibatch, batch in enumerate(train_loader):
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = batch['mask'].to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                out_new =  torch.sigmoid(masks_pred)

                loss = 0.5*criterion(masks_pred, true_masks) + 0.5*criterion_dice(out_new, true_masks)
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
                
                # -------------- Validation round ------------------
                if global_step % ((n_val+n_train) // (2 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/val', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/val', val_score, global_step)
                    
                    if val_score > best_score:
                        best_epoch = epoch
                        best_score = val_score
                        best_model_wts = copy.deepcopy(net.state_dict())

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        
        writer.add_scalar('Loss/train_epoch', epoch_loss / n_train, epoch)

    if save_cp:
        os.makedirs(dir_checkpoint, exist_ok=True)
        net.load_state_dict(best_model_wts)
        ckpt_path = os.path.join(dir_checkpoint, f'{model_name}_best_epoch_{best_epoch}.pth')
        torch.save(net.state_dict(), ckpt_path)
        logging.info(f'Checkpoint saved to {ckpt_path} at epoch {best_epoch}')

    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='2D ELM line segmentation from OCT images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=15.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-d', '--base-dir', dest='base_dir', type=str, default='./',
                        help='Base files directory')

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



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')   

# -------------- Load the model -----------
    #net = get_efficientunet_b3(n_classes=1, concat_input=True, pretrained=True)
    #net = LinkNetImprove(n_channels=3, n_classes=1)
    #net = AttU_Net(n_channels=3, n_classes=1)
    # net = U_Net(n_channels=3, n_classes=1)
    net = R2U_Net(n_channels=3, n_classes=1,t=2)
    #net = DeepLabv3_plus(n_channels=3, n_classes=1, os=16, pretrained=True, _print=True)
    #net = FCN(n_channels=3, n_classes=1)
    # net = SegNet(n_channels=3, n_classes=1)

    MODEL_NAME = f'{net.__class__.__name__}_{time.strftime("%b-%d-%Y_%H%M")}_model'
    experiment_dir = os.path.join(args.base_dir, 'elm-results/', MODEL_NAME)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    logging.info(f'Experiment dir: {experiment_dir}')

    args.model_name = MODEL_NAME
    args.experiment_dir = experiment_dir

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    net.to(device=device)
    try:
        train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100.0,
            args=args
        )
    except KeyboardInterrupt:
        interrupted_path = os.path.join(args.experiment_dir, 'checkpoints', f'INTERRUPTED_{args.model_name}.pth')
        torch.save(net.state_dict(), interrupted_path)
        logging.info(f'Saved interrupt checkpoint: {interrupted_path}')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
