#Run as Python 3.7!
import os, random, sys
sys.path.insert(0, '../')
from ModelUtils import get_top_corner
from ModelUtils import get_image_range
from ModelUtils import get_padding

from BaseDataset import BaseDataset

from PIL import Image

from torchvision import transforms

class MultiResDataset(BaseDataset):
    """MultiRes Dataset: Collated images downsampled at various resolutions"""

    def __init__(self, root_dir, centers_dir, transform=None, **kwargs):
         """
        Args:
            root_dir (string): Directory with all the images.
            centers_dir (string): location of focal points as pickle file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        super(MultiResDataset, self).__init__(
            root_dir=root_dir, 
            centers_dir=centers_dir, 
            transform=transform, 
            **kwargs
        )

    
    def custom_transform(self, image, center):
        #Random Horizontal flip
        if random.random() < 0.5:
            hFlip = True
            width = image.size[0]
            center = (center[0], width - center[1] - 1)
            image = transforms.functional.hflip(image)
        
        collated_image = Image.new('RGB', (16*4, 16))
        offset = 0
        
        for resolution in [256, 128, 64, 32]:
            image_range = get_image_range(center, size=resolution)

            padding = get_padding(image, image_range)

            upper_left = get_top_corner(center, padding, resolution)

            ### Transforms
            #Pad
            image_sample = transforms.Pad(padding, padding_mode="edge")(image)
            #Crop
            image_sample = transforms.functional.crop(image_sample,upper_left[0], upper_left[1], resolution, resolution)
            #Downsample
            image_sample = transforms.Resize((16,16), Image.BOX)(image_sample)
            #Collate
            collated_image.paste(image_sample, (offset,0))
            offset += 16
        
        if self.transform:
            collated_image = self.transform(collated_image)
        
        return collated_image
