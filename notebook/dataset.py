from torch.utils.data import TensorDataset, DataLoader,Dataset
import os
import cv2
import pandas as pd
import numpy as np
from utils import make_mask

class CloudDataset(Dataset):
    def __init__(self, path:str, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None):
        self.df = df
        if datatype != 'test':
            self.data_folder = f"{path}/train"
        else:
            self.data_folder = f"{path}/test"
        self.img_ids = img_ids
#         self.transforms = transforms
#         self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         augmented = self.transforms(image=img, mask=mask)
#         img = augmented['image']
#         mask = augmented['mask']
        # if self.preprocessing:
        #     preprocessed = self.preprocessing(image=img, mask=mask)
        #     img = preprocessed['image']
        #     mask = preprocessed['mask']
        img = img.transpose(2,0,1).astype('float32')
        mask = img.transpose(2,0,1).astype('float32')
        
        return img, mask

    def __len__(self):
        return len(self.img_ids)