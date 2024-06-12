import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


import numpy as np

from PIL import Image


class TrainingDataset(Dataset):
    def __init__(self, training_dir) -> None:
        super().__init__()
        self.training_dir = training_dir
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5)
        ])
        
    def __len__(self):
        return len(os.listdir(self.training_dir))
    
    def __getitem__(self, index):
        file_name = os.path.join(self.training_dir, self.training_dir[index])
        img = np.array(Image.open(file_name))/255
        
        if img.ndim == 2:
            img = img.reshape(1, -1)
        elif img.ndim == 3:
            img = img.transpose(2, 0, 1)
        else:
            raise ValueError('Incorrect number of dimensions')
        img = torch.from_numpy(img)
        img = self.transforms(img)
        return img
        
        
        
        
    
        