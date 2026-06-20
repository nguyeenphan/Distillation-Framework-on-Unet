"""
Transforms
──────────
Standard spatial & colour augmentations applied to image + mask together.
These are *separate* from StyCona (which operates at image-level before transforms).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    ])


def get_val_transform(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
    ])
