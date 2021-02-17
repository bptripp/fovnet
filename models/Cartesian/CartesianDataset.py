#Run as Python 3.7!
import random, sys
sys.path.insert(0, '../../fovnet/')
from remap import RGCMap

sys.path.insert(0, '../')
from ModelUtils import get_top_corner
from ModelUtils import get_image_range
from ModelUtils import get_padding

from BaseDataset import BaseDataset

import numpy as np
from PIL import Image

from torchvision import transforms

import cv2
from skimage.filters import gaussian

class Distort:
    """
    Cartesian Distort an image via RGCM Radial Pixel Positions. See remap.py
    """
    def __init__(self, image_size=(256,256)):
        """
        Parameters:
            Image size: The size of the image being warped. Assumed 256x256 from
            standard Imagenet models
        """
        rgcm = RGCMap(image_size, 70, .3, angle_steps=512, right=True)
        self.map = rgcm.radial_pixel_positions
        
        dim = int(self.map[-1]) #Size of target image
        x, y = np.mgrid[0:dim:1, 0:dim:1]
        x = x - dim/2.
        y = y - dim/2.
        d = (x*x + y*y)**0.5 
        self.theta = np.arctan2(y, x)

        f = np.vectorize(lambda x: self.map[int(x)] if int(x)<len(self.map) else self.map[-1])
        
        #Radius map
        self.r = f(d)
        
    def __call__(self, img, mask=None):
        height, width, _ = img.shape
        
        #Map
        map_x = self.r*np.cos(self.theta) + (width/2.)
        map_y = self.r*np.sin(self.theta) + (height/2.)
        
        ##Normalize Range
        max_x = map_x.max()
        max_y = map_y.max()
        min_x = map_x.min()
        min_y = map_y.min()
        
        map_x = np.array([(width)*(i-min_x)/(max_x - min_x) for i in map_x])
        map_y = np.array([(height)*(i-min_y)/(max_y - min_y) for i in map_y])
        
        #Correct dtype
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
                
        #Map
        img = gaussian(img, .5)
        img = cv2.remap(img, map_y, map_x, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return img
    

class CartesianDataset(BaseDataset):
    """Cartesian Retinal Downsampling at one of three focal points"""

    def __init__(self, root_dir, centers_dir, transform=None, crop_size=256, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the images.
            centers_dir (string): location of focal points as pickle file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        super(CartesianDataset, self).__init__(
            root_dir=root_dir, 
            centers_dir=centers_dir, 
            transform=transform, 
            **kwargs
        )

        self.crop_size = crop_size
        
        self.cartesian = Distort()

        
    def custom_transform(self, image, center):
        #Random Horizontal flip
        if random.random() < 0.5:
            hFlip = True
            width = image.size[0]
            center = (center[0], width - center[1] - 1)
            image = transforms.functional.hflip(image)
        
        image_range = get_image_range(center)
            
        padding = get_padding(image, image_range)
                
        upper_left = get_top_corner(center, padding, self.crop_size)
            
        ### Transforms
        #Pad
        image = transforms.Pad(padding, padding_mode="edge")(image)
        #Crop for foveation
        image = transforms.functional.crop(image,upper_left[0], upper_left[1], self.crop_size, self.crop_size)
        #Foveate
        image = np.array(image)
        if len(image.shape)==2:
            image = np.stack([image,image,image],axis=2)
                    
        warped_cartesian = self.cartesian(image)
        image = Image.fromarray(np.uint8(warped_cartesian*255))
        #Downsample
        image = transforms.Resize((32,32), Image.BOX)(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image