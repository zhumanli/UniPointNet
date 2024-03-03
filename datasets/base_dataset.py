import importlib
import os

import pytorch_lightning as pl
import torch
import torch.utils.data
from datasets.mydataset import mydataset
from torchvision import transforms


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_root, image_size, batch_size, num_workers=6):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.data_root = data_root



        self.train_dataset = mydataset(data_dir=os.path.join(self.data_root, self.dataset, 'train'), flag='train')
        self.val_dataset = mydataset(data_dir=os.path.join(self.data_root, self.dataset, 'val'), flag='val')
        self.test_dataset = mydataset(data_dir=os.path.join(self.data_root, self.dataset, 'test'), flag='test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=False, drop_last=False)
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, shuffle=False, drop_last=False)