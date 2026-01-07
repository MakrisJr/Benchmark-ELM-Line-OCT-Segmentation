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
from model import U_Net,AttU_Net,LinkNetImprove,U2NETP,R2U_Net,DeepLabv3_plus,FCN,SegNet, UNet2, UNet2D_attention, UNet3D, UNet3D_Aniso, UNet3D_Aniso2, UNet3DFrawley, MGUNet_2, UNet2DEnc3DDec
from transformation import ELM_transform, ELM_transform_gray
from tensorboardX import SummaryWriter
from dataset import BasicDataset, D3Dataset
from torch.utils.data import DataLoader, random_split
from dice_loss import Dice_Loss
import torch.nn.functional as F
from efficientunet import *
import matplotlib.pyplot as plt
import csv



def train_net(net,
              device,
              epochs=15,
              batch_size=4,
              lr=0.0001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1,
              args=None):
    
    model_name = args.model_name
    base_dir = args.base_dir
    train_dir_img = os.path.join(base_dir, 'data/train/image/')
    train_dir_mask = os.path.join(base_dir, 'data/train/mask/')
    val_dir_img = os.path.join(base_dir, 'data/val/image/')
    val_dir_mask = os.path.join(base_dir, 'data/val/mask/')
    dir_checkpoint = os.path.join(base_dir, 'elm-results/', model_name, 'checkpoints/')
    
    transform_train = True
    transform_val = False
    train_dataset = D3Dataset(train_dir_img, train_dir_mask, img_scale, transform = transform_train)
    val_dataset= D3Dataset(val_dir_img, val_dir_mask, img_scale, transform = transform_val)

    n_train=len(train_dataset)
    n_val=len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


    writer = SummaryWriter(logdir=os.path.join(args.experiment_dir, 'logs'), comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', factor = 0.2, patience=10)

# ------------------ Loss function ------------------!!
    criterion_dice = Dice_Loss
    criterion = nn.BCEWithLogitsLoss().cuda()
    
# !!-------------- Training and validation loop ------------------!!
    best_acc=0
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for ibatch, batch in enumerate(train_loader):
                imgs = batch['image']
                true_masks = batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # print(f"Input image size:", imgs.size())

                masks_pred = net(imgs)
                out_new =  F.sigmoid(masks_pred)

                loss = 0.5*criterion(masks_pred, true_masks) + 0.5*criterion_dice(out_new, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                if epoch == 0 and ibatch == 1:
                    print_gpu_mem(device)
                pbar.update(imgs.shape[0])
                global_step += 1
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
                        # logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/val', val_score, global_step)
                    
                    if val_score > best_acc:
                        best_acc = val_score
                        best_model_wts = copy.deepcopy(net.state_dict())
                        best_epoch = epoch

                    if imgs.dim() == 5:
                        B,C,D,H,W = imgs.shape
                        # permute to (B, D, C, H, W) then flatten B and D into the batch dim
                        imgs_to_write = imgs.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
                        true_masks_to_write = true_masks.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)
                        masks_pred_to_write = masks_pred.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)
                    else:
                        imgs_to_write = imgs

                    writer.add_images('images', imgs_to_write, global_step)

                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks_to_write, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred_to_write) > 0.5, global_step)

    if save_cp:
        try:
            os.makedirs(os.path.join(dir_checkpoint), exist_ok=True)
            logging.info('Created checkpoint directory')
        except OSError:
            print("Error: did not save checkpoint")
            pass
        net.load_state_dict(best_model_wts)
        torch.save(net.state_dict(), '{}/checkpoints/{}_best_epoch_{}.pth'.format(args.experiment_dir, model_name, best_epoch))
        logging.info(f'Checkpoint {best_epoch} saved !')

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
    parser.add_argument('-d', '--base_dir', dest='base_dir', type=str, default='./',
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

    n_classes =1 # number of output classes
    n_channels = 1 # number of input channels (1 for grayscale, 3 for RGB)


# -------------- Load the model -----------
    #net = get_efficientunet_b3(n_classes=1, concat_input=True, pretrained=True)
    #net = LinkNetImprove(n_channels=3, n_classes=1)
    #net = AttU_Net(n_channels=3, n_classes=1)
    #net = U_Net(n_channels=3, n_classes=1)
    #net = R2U_Net(n_channels=3, n_classes=1,t=2)
    #net = DeepLabv3_plus(n_channels=3, n_classes=1, os=16, pretrained=True, _print=True)
    #net = FCN(n_channels=3, n_classes=1)
    # net = SegNet(n_channels=3, n_classes=1)
    # net = UNet2(1,1)
    # net = UNet3D(1,1)
    # net = UNet3D_Aniso(1,1)
    # net = UNet3DFrawley(1,1)
    # net = MGUNet_2(in_channels=1, out_channels=1, feature_scale=4, is_deconv=True, is_batchnorm=True)
    # net = UNet3D_Aniso2(1,1)
    # net = UNet2D_attention(in_channels=1, out_channels=1)
    net = UNet2DEnc3DDec(in_channels=1, out_channels=1)

    MODEL_NAME = f'{net.__class__.__name__}_{time.strftime("%b-%d-%Y_%H%M")}_model'

    experiment_dir = os.path.join(args.base_dir, 'elm-results/', MODEL_NAME)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not os.path.exists(os.path.join(experiment_dir, 'checkpoints')):
        os.makedirs(os.path.join(experiment_dir, 'checkpoints'))
    logging.info(f'Experiment dir : {experiment_dir}')

    # add model name to args
    args.model_name = MODEL_NAME
    args.experiment_dir = experiment_dir


    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  args=args)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(args.experiment_dir, 'checkpoints','INTERRUPTED_' + args.model_name + '.model'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
