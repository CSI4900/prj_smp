# =====================================================
# File Name:    cityscapes.py
# Project Name: Object Segmentation
# Description:  Buildup of Cityscapes dataset
#
# Usage:        1. Import this file into train.py
#               2. Download Cityscapes dataset from
#               https://www.cityscapes-dataset.com/
#
# Contributors: 
# - Zechen Zhou     zzhou186@uottawa.ca
# - Shun Hei Yiu    syiu017@uottawa.ca
# =====================================================
import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset


class Cityscapes(BaseDataset):
    """Cityscapes Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder (e.g., leftImg8bit folder)
        masks_dir (str): path to segmentation masks folder (e.g., gtFine folder)
        classes (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline 
            (e.g., flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g., normalization, shape manipulation, etc.)
    """

    # Check the class definitions of cityscapes dataset
    # https://www.cityscapes-dataset.com/dataset-overview/#class-definitions
    CLASSES = ['road', 'sidewalk', 'parking', 'rail track', 'person', 'rider', 'car', 'truck', 'bus', 
               'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence', 
               'guard rail', 'bridge', 'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'vegetation', 
               'terrain', 'sky', 'ground', 'dynamic', 'static']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id)
                           for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id)
                          for image_id in self.ids]

        # Convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # Extract certain classes from mask (e.g., cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
