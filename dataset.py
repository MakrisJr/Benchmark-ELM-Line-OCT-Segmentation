from os.path import splitext
import os
import numpy as np
from glob import glob
import torch
from torchvision import transforms
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random



class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1,transform = None, single_channel = False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.single_channel = single_channel
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.im_ids = [splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        self.mask_ids = [splitext(file)[0] for file in os.listdir(masks_dir)
                    if not file.startswith('.')]

        # sort the lists
        self.im_ids.sort()
        self.mask_ids.sort()

        for im_id,mask_id in zip(self.im_ids,self.mask_ids):
            assert im_id == mask_id, \
                f'Images and masks {im_id} should be the same ID'
        
        logging.info(f'Creating dataset with {len(self.im_ids)} examples')
        self.transform=transform

    def __len__(self):
        return len(self.im_ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans
    

    def __getitem__(self, i):
        im_idx = self.im_ids[i]
      
        mask_idx = self.mask_ids[i]
  
        mask_file = glob(self.masks_dir + mask_idx + '.*')
        img_file = glob(self.imgs_dir + im_idx + '.*')

        # print(f"Loading image {img_file} and mask {mask_file}")
        # print(f"dimensions: image {Image.open(img_file[0]).size}, mask {Image.open(mask_file[0]).size}")

        # assert len(mask_file) == 1, \
            # f'Either no mask or multiple masks found for the ID {mask_idx}: {mask_file}'
        # assert len(img_file) == 1, \
            # f'Either no image or multiple images found for the ID {im_idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {im_idx} should be the same size, but are {img.size} and {mask.size}'
        
        mask = mask.convert('1')
        if self.single_channel:
            # Return a single-channel grayscale image (mode 'L')
            img = img.convert(mode='L')
        else:
            # Default behavior: return 3-channel RGB image
            img = img.convert(mode='RGB')

        if self.transform:
            img,mask=self.transform(img,mask)
      
        return {'image': img,'mask': mask}
    
"""
The 3D dataset stacks the 49 slices of the OCT into a 3D volume with dimensions 49 x H x W 
Image format is EYE_ID-SLICE_ID.png, where SLICE_ID is from 0 to 48. Slices are greyscale (single channel).

"""
# class D3Dataset(Dataset): 
#     def __init__(self, imgs_dir, masks_dir, scale=1,transform = None):
#         self.imgs_dir = imgs_dir
#         self.masks_dir = masks_dir
#         self.scale = scale
#         assert 0 < scale <= 1, 'Scale must be between 0 and 1'

#         # Get unique eye IDs by removing the slice part
#         all_files = [file for file in os.listdir(imgs_dir) if not file.startswith('.')]
#         self.eye_ids = sorted(set('-'.join(file.split('-')[:-1]) for file in all_files))

#         logging.info(f'Creating 3D dataset with {len(self.eye_ids)} examples')
#         self.transform=transform
#     def __len__(self):
#         return len(self.eye_ids)
#     @classmethod
#     def preprocess(cls, pil_img, scale):
#         w, h = pil_img.size
#         newW, newH = int(scale * w), int(scale * h)
#         assert newW > 0 and newH > 0, 'Scale is too small'
#         pil_img = pil_img.resize((newW, newH))

#         img_nd = np.array(pil_img)

#         if len(img_nd.shape) == 2:
#             img_nd = np.expand_dims(img_nd, axis=2)

#         # HWC to CHW
#         img_trans = img_nd.transpose((2, 0, 1))
#         if img_trans.max() > 1:
#             img_trans = img_trans / 255

#         return img_trans
#     def __getitem__(self, i):
#         eye_id = self.eye_ids[i]
#         img_slices = []
#         mask_slices = []
#         for slice_idx in range(49):  # Assuming 49 slices per eye
#             img_file = os.path.join(self.imgs_dir, f"{eye_id}-{slice_idx}.png")
#             mask_file = os.path.join(self.masks_dir, f"{eye_id}-{slice_idx}.png")
#             try:
#                 img = Image.open(img_file).convert(mode='L')  # Grayscale
#                 mask = Image.open(mask_file).convert('1')  # Binary mask

#                 if self.transform:
#                     img, mask = self.transform(img, mask)

#             except FileNotFoundError:
#                 # logging.warning(f"Missing slice {slice_idx} for eye {eye_id}. Using last available slice.")
#                 img = img_slices[-1]  # Use last available slice
#                 mask = mask_slices[-1]  # Use last available slice


#             img_slices.append(np.array(img)) # each slice is 1 x H x W
#             mask_slices.append(np.array(mask))

#         # Combine slices to create 3D volume
#         img_volume = np.concatenate(img_slices, axis=0)  # 49 x H x W 
#         mask_volume = np.concatenate(mask_slices, axis=0)  # 49 x H x W
#         # print(f"img dimsions for eye {eye_id}:", img_volume.shape)

#         return {'image': torch.tensor(img_volume, dtype=torch.float32).unsqueeze(0), # 1 x 49 x H x W
#                 'mask': torch.tensor(mask_volume, dtype=torch.float32).unsqueeze(0)} # 1 x 49 x H x W

class D3Dataset(Dataset): 
    def __init__(self, imgs_dir, masks_dir, scale=1, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        # Get unique eye IDs by removing the slice part
        all_files = [file for file in os.listdir(imgs_dir) if not file.startswith('.')]
        self.eye_ids = sorted(set('-'.join(file.split('-')[:-1]) for file in all_files))

        logging.info(f'Creating 3D dataset with {len(self.eye_ids)} examples')
        self.transform = transform

        # Define base augmentations
        self.resize = A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)  # masks stay nearest
        ])

        self.normalize = A.Compose([
            A.Normalize(mean=[0.2147], std=[0.2256])
        ])


        self.to_tensor = ToTensorV2()

    def apply_volume_transform(self, img_volume, mask_volume):
        """
        Apply 3D-consistent geometric and photometric transforms.
        """

        # reverse slice order augmentation: 
        if random.random() < 0.5:
            img_volume = img_volume[::-1].copy()
            mask_volume = mask_volume[::-1].copy()
        

        # One ReplayCompose for geometric and photometric transforms, to apply on all slices
        vol_transform = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),

            A.Affine(
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                scale=(0.90, 1.10),
                rotate=(-20, 20),
                p=0.7,
                interpolation=0,
                mask_interpolation=0,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0
            ),

            # Photometrics — these MUST be inside ReplayCompose
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),

            A.GaussNoise(
                std_range=(0.01, 0.03),
                mean_range=(0.0, 0.0),
                per_channel=False,
                p=0.3,
            )
        ])

        # Generate replay from the first slice
        replay = vol_transform(
            image=img_volume[0],
            mask=mask_volume[0]
        )['replay']

        transformed_slices = []
        transformed_masks = []

        # Apply EXACT same params to all slices
        for img_slice, mask_slice in zip(img_volume, mask_volume):
            out = A.ReplayCompose.replay(
                replay,
                image=img_slice,
                mask=mask_slice
            )
            transformed_slices.append(out['image'])
            transformed_masks.append(out['mask'])

        return np.stack(transformed_slices), np.stack(transformed_masks)

    def __getitem__(self, i):
        eye_id = self.eye_ids[i]
        img_slices = []
        mask_slices = []

        # load all 49 slices
        for slice_idx in range(49):
            img_file = f"{self.imgs_dir}/{eye_id}-{slice_idx}.png"
            mask_file = f"{self.masks_dir}/{eye_id}-{slice_idx}.png"
            try:
                img = Image.open(img_file).convert('L')
                mask = Image.open(mask_file).convert('L')
            except FileNotFoundError:
                # repeat last slice if current is missing
                img = img_slices[-1]
                mask = mask_slices[-1]

            img_slices.append(np.array(img))
            mask_slices.append(np.array(mask))

        # Form 3D volume D x H x W
        img_volume = np.stack(img_slices)
        mask_volume = np.stack(mask_slices)

        # Apply transforms
        if self.transform == True:
            img_volume, mask_volume = self.apply_volume_transform(img_volume, mask_volume)

        # Resize
        resized_imgs = []
        resized_masks = []

        for img_slice, mask_slice in zip(img_volume, mask_volume):
            out = self.resize(image=img_slice, mask=mask_slice)
            resized_imgs.append(out["image"])
            resized_masks.append(out["mask"])

        img_volume = np.stack(resized_imgs)
        mask_volume = np.stack(resized_masks)



        # apply median filter:
        # img_volume = np.stack([
        #     cv2.medianBlur(slice, ksize=3) for slice in img_volume
        # ])

        # Normalize (images only)
        # img_volume = np.stack(norm_imgs)
        img_volume = img_volume.astype(np.float32) / 255.0
        img_volume = (img_volume - 0.2147) / 0.2256


        # Convert to torch tensor with channel dimension
        return {
            'image': torch.tensor(img_volume, dtype=torch.float32).unsqueeze(0),
            'mask': torch.tensor(mask_volume / 255.0, dtype=torch.float32).unsqueeze(0) # also convert to binary
        }

    def __len__(self):
        return len(self.eye_ids)
