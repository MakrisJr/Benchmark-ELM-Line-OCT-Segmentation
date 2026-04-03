from PIL import Image, ImageOps
from torchvision.transforms import functional as F
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
The code was adapted from
https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
"""

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
    
class ToTensor(object):
    def __call__(self, img, mask):
        return F.to_tensor(img), F.to_tensor(mask)

class ToPILImage(object):
    def __call__(self, img, mask):
        return F.to_pil_image(img), F.to_pil_image(mask)

# class RandomHorizontallyFlip(object):
#     def __call__(self, img, mask):
#         if random.random() < 0.5:
#             return img.transpose(Image.FLIP_LEFT_RIGHT),  mask.transpose(Image.FLIP_LEFT_RIGHT)
#         return img, mask

class Equalization(object):
    def __call__(self, img, mask):
        return ImageOps.equalize(img), mask
    
class GammaAdjustment(object):
    def __init__(self, gamma = 1.3):
        self.gamma = gamma
    def __call__(self, img, mask):
        return F.adjust_gamma(img = img, gamma = self.gamma), mask
        
class ContrastAdjustment(object):
    def __init__(self, contrast_factor = 2):
        self.contrast_factor = contrast_factor
    def __call__(self, img,mask):
        return F.adjust_contrast(img = img, contrast_factor=self.contrast_factor), mask
    
class GaussianNoise(object):
    def __call__(self, img, mask):
        return img + torch.randn_like(img), mask 
    
class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

class RandomLightVar(object):
    def __call__(self, img,mask):
        return (img+random.random()*64-32).astype('uint8'), mask

class RandomLightRevert(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return 255-img, mask
        else:
            return img, mask
        
class Normalization(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self, img,mask):
        return F.normalize(img, self.mean, self.std, self.inplace), mask
    
class Grayscale(object):
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img,mask):

        return F.to_grayscale(img, num_output_channels=self.num_output_channels), mask
        
class Resize(object):
    def __init__(self, size, interpolation = Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img,mask):
        return F.resize(img, self.size, self.interpolation), F.resize(mask, self.size, self.interpolation)    

class ReverseSlices:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_volume, mask_volume):
        if random.random() < self.p:
            return img_volume[::-1].copy(), mask_volume[::-1].copy()
        return img_volume, mask_volume

def ELM_transform(normalize = True):
    transform = {
    'train': Compose([
        #RandomHorizontallyFlip(),
        #RandomRotate(5),
        #ContrastAdjustment(1.2),
        Resize((256,256)),
        ToTensor(),
        Normalization([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) 
    ]),
    'val': Compose([Resize((256,256)),
        ToTensor(),
        Normalization([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    'test': Compose([Resize((256,256)),
        ToTensor(),
        Normalization([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    }
    return transform


# def ELM_transform_gray(normalize = True):
#     # Training data mean: 54.76707466898142, std: 57.55662815500704 
#     # or from 0 to 1: mean: 0.2147, std: 0.2256
#     transform = {
#     'train': Compose([
#         #RandomHorizontallyFlip(),
#         #RandomRotate(5),
#         #ContrastAdjustment(1.2),
#         Resize((256,256)),
#         ToTensor(),
#         Normalization([0.2147],[0.2256])
#     ]),
#     'val': Compose([Resize((256,256)),
#         ToTensor(),
#         Normalization([0.2147],[0.2256])
#     ]),
#     'test': Compose([Resize((256,256)),
#         ToTensor(),
#         Normalization([0.2147],[0.2256])
#     ])
#     }
#     return transform



def apply_volume_transform(volume, mask_volume, transform_2d):
    """
    Apply a 2D Albumentations transform to all slices in a volume with consistent parameters.
    volume: np.array (D, H, W)
    mask_volume: np.array (D, H, W)
    transform_2d: A.ReplayCompose for deterministic transforms
    """
    # Generate deterministic parameters on first slice
    replay = transform_2d(image=volume[0], mask=mask_volume[0])['replay']
    
    transformed_slices = []
    transformed_masks = []
    for img_slice, mask_slice in zip(volume, mask_volume):
        augmented = A.ReplayCompose.replay(replay, image=img_slice, mask=mask_slice)
        transformed_slices.append(augmented['image'])
        transformed_masks.append(augmented['mask'])
    
    return np.stack(transformed_slices), np.stack(transformed_masks)

def ELM_transform_gray(normalize=True, mean=[0.2147], std=[0.2256]):
    """
    Returns train/val/test transforms for full 3D OCT volumes.
    Geometric transforms (rotation, shift) are 3D-consistent.
    Photometric transforms (brightness, contrast, noise) are applied slice-wise.
    """
    
    # Geometric transforms applied consistently across slices
    train_geom = A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            rotate=(-5, 5),
            p=0.7,
            mode=0
        )
    ])
    
    # Photometric transforms applied slice-wise
    train_photometric = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
    ])
    
    # Resize, ToTensor, Normalize (per slice)
    base_transforms = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    transform = {
        'train': lambda volume, mask_volume: (
            np.stack([
                base_transforms(image=train_photometric(image=slice_, mask=mask_slice)['image'])['image']
                for slice_, mask_slice in zip(*apply_volume_transform(volume, mask_volume, train_geom))
            ]),
            np.stack([
                base_transforms(image=train_photometric(image=slice_, mask=mask_slice)['mask'])['mask']
                for slice_, mask_slice in zip(*apply_volume_transform(volume, mask_volume, train_geom))
            ])
        ),
        'val': lambda volume, mask_volume: (
            np.stack([
                base_transforms(image=slice_)['image'] for slice_ in volume
            ]),
            np.stack([
                base_transforms(image=mask_slice)['mask'] for mask_slice in mask_volume
            ])
        ),
        'test': lambda volume, mask_volume: (
            np.stack([
                base_transforms(image=slice_)['image'] for slice_ in volume
            ]),
            np.stack([
                base_transforms(image=mask_slice)['mask'] for mask_slice in mask_volume
            ])
        )
    }
    
    return transform