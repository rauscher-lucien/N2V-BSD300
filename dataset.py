import os
import numpy as np
import torch

from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image

from utils import *


class DatasetLoadJPG(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        """
        Initializes the dataset with the path to a folder containing grayscale JPEG images,
        and an optional transform to be applied to each image.

        Parameters:
        - root_folder_path: Path to the root folder containing grayscale JPEG image files.
        - transform: Optional transform to be applied to each image.
        """
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.file_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.root_folder_path)
                           for f in filenames if f.lower().endswith('.jpg')]

        # Preemptively load all images into memory
        self.images = []
        for file_path in self.file_paths:
            image = np.array(Image.open(file_path).convert('L'))  # Convert image to grayscale ('L' mode)
            if image.ndim == 2:
                image = image[np.newaxis, ...]  # Add channel dimension (C, H, W)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        # Apply the transform if specified
        if self.transform:
            image = self.transform(image)

        return image








