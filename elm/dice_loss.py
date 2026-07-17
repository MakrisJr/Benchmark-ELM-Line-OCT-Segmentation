import torch
from torch.autograd import Function
import torch.nn.functional as F

'''
def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return  s / (i + 1)
'''
def dice_coeff(pred, gt):

    p = pred.contiguous().view(-1)
    g = gt.contiguous().view(-1)
    intersection = (p * g).sum()

    p_sum = torch.sum(p * p)
    g_sum = torch.sum(g * g)
    
    return (2. * intersection + 1) / (p_sum + g_sum + 1) 



def dice_loss(pred, gt): # dims are B x 1 x D x H x W

    p = pred.contiguous().view(-1)
    g = gt.contiguous().view(-1)
    intersection = (p * g).sum()

    p_sum = torch.sum(p * p)
    g_sum = torch.sum(g * g)
    
    return 1 - ((2. * intersection + 1) / (p_sum + g_sum + 1) )  

def soft_dice_loss_per_slice(pred, gt): 
    """
    input dimensions: B x 1 x D x H x W
    """
    pred = pred.float()
    gt = gt.float()
    dims = (1,3,4)  # dimensions to sum over
    intersection = (pred * gt).sum(dim=dims)
    denominator = pred.sum(dim=dims) + gt.sum(dim=dims)
    dice = (2. * intersection + 1) / (denominator + 1)  # shape: B x D
    mean_dice = dice.mean()  # average over batch and slices
    return 1 - mean_dice # output is scalar

def dice_per_slice_mean(pred, gt):
    """
    input dimensions: B x 1 x D x H x W
    """
    pred = pred.float()
    gt = gt.float()
    dims = (1,3,4)  # dimensions to sum over
    intersection = (pred * gt).sum(dim=dims)
    denominator = pred.sum(dim=dims) + gt.sum(dim=dims)
    dice = (2. * intersection + 1) / (denominator + 1)  # shape: B x D
    mean_dice = dice.mean()  # average over batch and slices
    return mean_dice # output is scalar


def soft_erode(img):
    # img: B x C x H x W (2D) or B x C x D x H x W (3D)
    if img.dim() == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif img.dim() == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    else:
        raise ValueError(f'soft_erode expects a 4D or 5D tensor, got {img.dim()}D')

def soft_dilate(img):
    if img.dim() == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif img.dim() == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
    else:
        raise ValueError(f'soft_dilate expects a 4D or 5D tensor, got {img.dim()}D')

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iters):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

def cldice_loss(pred, gt, iters=3, smooth=1.0):
    """
    pred: sigmoid probabilities (not logits), gt: binary mask.
    Both B x C x H x W (2D) or B x C x D x H x W (3D).
    """
    skel_pred = soft_skel(pred, iters)
    skel_gt = soft_skel(gt, iters)
    tprec = (torch.sum(skel_pred * gt) + smooth) / (torch.sum(skel_pred) + smooth)
    tsens = (torch.sum(skel_gt * pred) + smooth) / (torch.sum(skel_gt) + smooth)
    return 1 - 2.0 * (tprec * tsens) / (tprec + tsens)
