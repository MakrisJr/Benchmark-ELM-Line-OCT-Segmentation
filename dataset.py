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
from pathlib import Path
import pandas as pd


def make_2d_transforms(train: bool, out_size=(256, 256)):
    """
    Returns Albumentations Compose object with 2D augmentations for training or validation.
    - train=True: augmentation + resize + normalize + to tensor
    - train=False: resize + normalize + to tensor (no augmentation)
    """
    resize = A.Resize(height=out_size[0], width=out_size[1], interpolation=cv2.INTER_LINEAR)

    aug = A.Compose([
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
    ], p=1.0)

    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = ToTensorV2()
    if train:
        return A.Compose([aug, resize, normalize, to_tensor])
    else:
        return A.Compose([resize, normalize, to_tensor])

class BasicDataset_old(Dataset):
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

        # keep only the IDs that are present in both images and masks
        self.im_ids = list(set(self.im_ids) & set(self.mask_ids))
        self.im_ids.sort()
        self.mask_ids = self.im_ids.copy()

        # for im_id,mask_id in zip(self.im_ids,self.mask_ids):
        #     # print(f"Image ID: {im_id}, Mask ID: {mask_id}")

        #     assert im_id == mask_id, \
        #         f'Images and masks {im_id} should be the same ID'
        
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
        
        mask = mask.convert('L')
        if self.single_channel:
            # Return a single-channel grayscale image (mode 'L')
            img = img.convert(mode='L')
        else:
            # Default behavior: return 3-channel RGB image
            img = img.convert(mode='RGB')

        img = np.array(img)
        mask = np.array(mask).astype(np.uint8)

        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out['image'], out['mask']

        if torch.is_tensor(mask):
            mask = (mask>0).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.from_numpy((mask>0).astype(np.float32)).unsqueeze(0)
        return {'image': img,'mask': mask}


class BasicDataset(Dataset):
    def __init__(
        self,
        root_dir="data_no_anomalies",
        split=None,            # None, "train", or "val"
        fold=None,             # integer fold index, required if split is train/val
        scale=1.0,
        transform=None,
        single_channel=False,
        image_dir="all/image",
        mask_dir="all/mask",
        metadata_filename="metadata.csv",
        image_ext=".png",      # change if your files are .jpg, .tif, etc.
        mask_ext=".png",
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_dir
        self.mask_dir = self.root_dir / mask_dir
        self.metadata_path = self.root_dir / metadata_filename

        self.scale = scale
        self.transform = transform
        self.single_channel = single_channel
        self.image_ext = image_ext
        self.mask_ext = mask_ext

        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Read metadata
        df = pd.read_csv(self.metadata_path, dtype={"patient_id": str})

        required_cols = {"patient_id", "fold"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"metadata.csv is missing required columns: {missing}")

        # Normalize patient IDs like 004, 012, etc.
        df["patient_id"] = df["patient_id"].astype(str).str.strip().str.zfill(3)

        # Apply split filtering
        if split is not None:
            if split not in {"train", "val"}:
                raise ValueError("split must be one of: None, 'train', 'val'")
            if fold is None:
                raise ValueError("fold must be provided when split is 'train' or 'val'")

            if split == "train":
                df = df[df["fold"] != fold].copy()
            elif split == "val":
                df = df[df["fold"] == fold].copy()

        # Keep only patients that have both image and mask files
        valid_rows = []
        for _, row in df.iterrows():
            pid = row["patient_id"]

            # Dataset convention: per-slice 2D images and masks
            # e.g. 572-0.png .. 572-48.png (or similar)
            img_paths = sorted(self.image_dir.glob(f"{pid}-*{self.image_ext}"))
            if not img_paths:
                logging.warning(
                    f"Skipping {pid}: no images found matching {pid}-*{self.image_ext} in {self.image_dir}"
                )
                continue

            n_added = 0
            for img_path in img_paths:
                mask_path = self.mask_dir / img_path.name
                if not mask_path.exists():
                    logging.warning(f"Skipping {img_path.name}: mask not found at {mask_path}")
                    continue

                valid_rows.append(
                    {
                        "patient_id": pid,
                        "fold": int(row["fold"]),
                        "image_path": img_path,
                        "mask_path": mask_path,
                    }
                )
                n_added += 1

            if n_added == 0:
                logging.warning(f"Skipping {pid}: no valid (image, mask) slice pairs found")

        self.meta = pd.DataFrame(valid_rows).reset_index(drop=True)

        logging.info(
            f"Creating dataset with {len(self.meta)} examples."
            f"(root={self.root_dir}, split={split}, fold={fold})"
        )

    def __len__(self):
        return len(self.meta)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"

        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC -> CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255.0

        return img_trans

    def __getitem__(self, i):
        row = self.meta.iloc[i]

        patient_id = row["patient_id"]
        img_path = row["image_path"]
        mask_path = row["mask_path"]

        # Slice index is encoded in filename like "841-0.png"
        image_name = Path(img_path).name
        stem = Path(img_path).stem
        parts = stem.split("-", 1)
        slice_idx = int(parts[1])


        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if img.size != mask.size:
            raise ValueError(
                f"Image and mask {patient_id} should be the same size, "
                f"but are {img.size} and {mask.size}"
            )

        mask = mask.convert("L")
        if self.single_channel:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        if self.scale != 1.0:
            w, h = img.size
            newW, newH = int(self.scale * w), int(self.scale * h)
            if newW <= 0 or newH <= 0:
                raise ValueError("Scale is too small")

            img = img.resize((newW, newH))
            mask = mask.resize((newW, newH), resample=Image.NEAREST)

        img = np.array(img)
        mask = np.array(mask).astype(np.uint8)

        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        else:
            # Manual conversion if no transform is provided
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            img = img.transpose((2, 0, 1)).astype(np.float32)
            if img.max() > 1:
                img = img / 255.0
            img = torch.from_numpy(img)

        if torch.is_tensor(mask):
            mask = (mask > 0).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)

        return {
            "patient_id": patient_id,
            "fold": int(row["fold"]),
            "image_name": image_name,
            "slice_idx": slice_idx,
            "image": img,
            "mask": mask,
        }
    
"""
The 3D dataset stacks the 49 slices of the OCT into a 3D volume with dimensions 49 x H x W 
Image format is EYE_ID-SLICE_ID.png, where SLICE_ID is from 0 to 48. Slices are greyscale (single channel).

"""
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
            'image': torch.tensor(img_volume, dtype=torch.float32).unsqueeze(0), # dims are [1 x 49 x 256 x 256]
            'mask': torch.tensor(mask_volume / 255.0, dtype=torch.float32).unsqueeze(0) # also convert to binary
        }

    def __len__(self):
        return len(self.eye_ids)


class D3WindowDataset(Dataset):
    """
    Returns K windows per volume.
    Each window is 7 slices: image -> [K, 1, 7, H, W], mask -> [K, 1, 7, H, W]
    """

    def __init__(self, imgs_dir, masks_dir, transform=False, window_depth=7, K=1, return_all_windows=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.window_depth = window_depth
        self.K = K
        self.D = 49  # total slices per volume
        assert self.K >= 1, "K must be at least 1"
        self.W = self.D - self.window_depth + 1 # possible window start positions (43 for depth=7)
        self.return_all_windows = return_all_windows # ignores K and returns all windows

        all_files = [file for file in os.listdir(imgs_dir) if not file.startswith('.')]
        self.eye_ids = sorted(set('-'.join(file.split('-')[:-1]) for file in all_files))
        logging.info(f'Creating windowed 3D dataset with {len(self.eye_ids)} volumes, {self.window_depth}-slice windows, K={self.K} windows per epoch')

        self.resize = A.Compose([
            A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)
        ])

        # training dataset normalization params
        self.mean = 0.2147
        self.std = 0.2256

    def __len__(self):
        return len(self.eye_ids)
    
    def apply_volume_transform(self, img_volume, mask_volume):
        """
        Apply 3D-consistent geometric and photometric transforms across all slices.
        """

        # reverse slice order augmentation:
        if random.random() < 0.5:
            img_volume = img_volume[::-1].copy()
            mask_volume = mask_volume[::-1].copy()

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

        replay = vol_transform(image=img_volume[0], mask=mask_volume[0])['replay']

        transformed_slices = []
        transformed_masks = []
        for img_slice, mask_slice in zip(img_volume, mask_volume):
            out = A.ReplayCompose.replay(replay, image=img_slice, mask=mask_slice)
            transformed_slices.append(out['image'])
            transformed_masks.append(out['mask'])

        return np.stack(transformed_slices), np.stack(transformed_masks)
    
    def _load_volume(self, eye_id):
        """
        Load all 49 slices for the given eye_id into a 3D volume.
        """
        img_slices = []
        mask_slices = []

        for slice_idx in range(49):
            img_file = f"{self.imgs_dir}/{eye_id}-{slice_idx}.png"
            mask_file = f"{self.masks_dir}/{eye_id}-{slice_idx}.png"
            try:
                # use 'L' mode for grayscale
                img = Image.open(img_file).convert('L')
                mask = Image.open(mask_file).convert('L')
            except FileNotFoundError: # if slice is missing, repeat last slice
                img = img_slices[-1]
                mask = mask_slices[-1]

            img_slices.append(np.array(img))
            mask_slices.append(np.array(mask))

        img_volume = np.stack(img_slices)
        mask_volume = np.stack(mask_slices)
        return img_volume, mask_volume # [49 x H x W], [49 x H x W]
    
    def _resize_volume(self, img_volume, mask_volume):
        resized_imgs = []
        resized_masks = []

        for img_slice, mask_slice in zip(img_volume, mask_volume):
            out = self.resize(image=img_slice, mask=mask_slice)
            resized_imgs.append(out["image"])
            resized_masks.append(out["mask"])

        img_volume = np.stack(resized_imgs)
        mask_volume = np.stack(resized_masks)
        return img_volume, mask_volume # [49 x 256 x 256], [49 x 256 x 256]
    
    def _normalize_volume(self, img_volume):
        img_volume = img_volume.astype(np.float32) / 255.0
        img_volume = (img_volume - self.mean) / self.std
        return img_volume # [49 x 256 x 256]
    
    def _sample_starts(self):
        """
        Sample K starting indices for windows of size window_depth within D=49 slices.
        """
        if self.K <= self.W:
            return random.sample(range(self.W), self.K) # K samples without replacement.
        else:
            return [random.randint(0, self.W - 1) for _ in range(self.K)] # with replacement.

    def __getitem__(self, i):
        eye_id = self.eye_ids[i]
        img_volume, mask_volume = self._load_volume(eye_id) # [49 x H x W], [49 x H x W]

        # Apply transforms
        if self.transform:
            img_volume, mask_volume = self.apply_volume_transform(img_volume, mask_volume)

        # Resize (regardless of augmentations)
        img_volume, mask_volume = self._resize_volume(img_volume, mask_volume) # [49 x 256 x 256], [49 x 256 x 256]
        
        # Normalize
        img_volume = self._normalize_volume(img_volume) # [49 x 256 x 256]

        # Binarize masks
        mask_volume = (mask_volume > 127).astype(np.uint8) # [49 x 256 x 256]

        # Sample K windows or return all windows
        if self.return_all_windows:
            starts = list(range(self.W)) # all possible starting indices 0..42
        else:
            starts = self._sample_starts() # K starting indices
        img_windows = []
        mask_windows = []

        for start in starts:
            end = start + self.window_depth
            img_window = img_volume[start:end] # [window_depth x 256 x 256]
            mask_window = mask_volume[start:end] # [window_depth x 256 x 256]

            img_windows.append(img_window)
            mask_windows.append(mask_window)

        # Stack windows into tensors
        x = np.stack(img_windows, axis=0) # [K x window_depth x 256 x 256]
        y = np.stack(mask_windows, axis=0) # [K x window_depth x 256 x 256]

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1) # [K x 1 x window_depth x 256 x 256]
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # [K x 1 x window_depth x 256 x 256]

        return {
            'image': x, # [K x 1 x window_depth x 256 x 256]
            'mask': y,   # [K x 1 x window_depth x 256 x 256]
            'eye_id': eye_id
        }
