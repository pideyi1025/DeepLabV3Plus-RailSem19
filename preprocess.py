# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset as base
from torchvision import transforms
from tqdm import tqdm
import albumentations as album

class main_dataset(base):


    CLASSES = ['road',
            'sidewalk',
            'construction',
            'tram-track',
            'fence',
            'pole',
            'traffic-light',
            'traffic-sign',
            'vegetation',
            'terrain',
            'sky',
            'human',
            'rail-track',
            'car',
            'truck',
            'trackbed',
            'on-rails',
            'rail-raised',
            'rail-embedded']

    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            image_count,
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = sorted(os.listdir(images_dir))
        self.ids = self.ids[0:image_count]
        self.mask_ids = sorted(os.listdir(masks_dir))
        self.mask_ids = self.mask_ids[0:image_count]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        #read data
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i],0)
        
        #crop and reshape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks =[(mask == v) for v in self.class_values]
        mask = np.stack(masks,axis = -1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
        
    def __len__(self):
        return len(self.ids)

#######VISUAL###########
class vis_dataset(base):
    CLASSES = ['road',
            'sidewalk',
            'construction',
            'tram-track',
            'fence',
            'pole',
            'traffic-light',
            'traffic-sign',
            'vegetation',
            'terrain',
            'sky',
            'human',
            'rail-track',
            'car',
            'truck',
            'trackbed',
            'on-rails',
            'rail-raised',
            'rail-embedded']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = sorted(os.listdir(images_dir))
        #self.ids = self.ids[0:1000]
        self.mask_ids = sorted(os.listdir(masks_dir))
        #self.mask_ids = self.mask_ids[0:1000]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
    
    def __getitem__(self, i):
        
        #read data
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i],0)
        #crop and reshape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks =[(mask == v) for v in self.class_values]
        mask = np.stack(masks,axis = -1).astype(np.float32)

        return image, mask
        
    def __len__(self):
        return len(self.ids)


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(15,5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if name == 'predicted_mask_railtrack' or name == 'predicted_mask_railraised':
            x_start = 0
            x_end = 1920
            y_start = 420
            y_end = 1500
            image = image[y_start:y_end,x_start:x_end]
        plt.imshow(image, interpolation='nearest')
    plt.show()
