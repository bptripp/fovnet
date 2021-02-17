#Run as Python 3.7!
import sys
sys.path.insert(0, '../')
from BaseDataset import BaseDataset

from PIL import Image

from torchvision import transforms

class BaselineDataset(BaseDataset):
    """Fovnet V1 Dataset: Polar retinal downsampling on one focal point"""

    def __init__(self, root_dir, centers_dir, transform=None, **kwargs):
        """
        Args:
            root_dir (string): Directory with all the images.
            centers_dir (string): location of focal points as pickle file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        super().__init__(
            root_dir=root_dir, 
            centers_dir=centers_dir, 
            transform=transform, 
            **kwargs
        )
        
    def custom_transform(self, image, center):
        #Downsample
        image = transforms.Resize((32,32), Image.BOX)(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image
