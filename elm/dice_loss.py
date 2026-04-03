import torch
from torch.autograd import Function

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
