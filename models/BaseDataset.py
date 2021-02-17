#Run as Python 3.7!
from ModelUtils import make_dataset

import random, pickle
from PIL import Image
from abc import abstractmethod

from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

import os

class BaseDataset(datasets.ImageFolder):
    """BaseDataset is an abstract class for defining other datasets.
        Must subclass and implement the __transform function
    """

    def __init__(self, root_dir, centers_dir, transform=None, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        super().__init__(root_dir, **kwargs)
        
        self.center_points = pickle.load(open(centers_dir, "rb"))
        
        self.samples = make_dataset(self.root, self.class_to_idx, self.center_points)
                    
        self.root_dir = root_dir
        
        self.final_loader = False
        
        self.transform = transform

    @abstractmethod
    def custom_transform(self, image, center):
        raise NotImplementedError("Cannot call custom_transform on abstract BaseDataset object. Must invoke from subclass") 

    def __center_for_img(self, img_name):
        """
        Get the focal point of an img based on the image name
        """
        img_name = img_name.split(os.sep)
        
        if "val" in img_name:
            i = img_name.index("val")
        else:
            i = img_name.index("train")

        img_key = "/".join(img_name[i:])

        center = self.center_points[img_key]
        if isinstance(center, tuple):
            center = [center]

        if self.final_loader:
            """
            If loading as part of final validation, we need a specific focal point... 
            """
            idx = self.final_loader
            center = center[idx]
        else:
            """
            ...otherwise just pick a random one
            """
            center = random.choice(center)
            
        return center
            
    def __getitem__(self, idx):     
        img_name, target = self.samples[idx]
        
        with open(img_name, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
            
        center = self.__center_for_img(img_name)
           
        image = self.custom_transform(image, center)
        
        return (image, target)
