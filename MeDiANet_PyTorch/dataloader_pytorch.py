import os 
import torch
from torch.utils.data import Dataset
import numpy as np


dataset_folder = '/data/dipayan/MedMNIST_Dataset/NewDataset/newdataset'


class mdnet_dataset(Dataset):
    def __init__(self, MODE):
        self.train_folder = os.path.join(dataset_folder, "train")
        self.val_folder = os.path.join(dataset_folder, "val")
        self.test_folder = os.path.join(dataset_folder, "test")

        self.MODE = MODE # Used to set train, val, test mode

        self.train_filenames = []
        self.val_filenames = []
        self.test_filenames = []

        # Getting train, validation and test filenames:
        for file in os.listdir(self.train_folder):
            relative_file_path = os.path.join(self.train_folder, file)
            self.train_filenames.append(relative_file_path)

        for file in os.listdir(self.val_folder):
            relative_file_path = os.path.join(self.val_folder, file)
            self.val_filenames.append(relative_file_path)

        for file in os.listdir(self.test_folder):
            relative_file_path = os.path.join(self.test_folder, file)
            self.test_filenames.append(relative_file_path)

    
    def __getitem__(self, index):
        if self.MODE == 'train':
            data = self.train_filenames[index]

            with np.load(data) as npz_file: 
                img = npz_file['a']
                label = npz_file['b']
            img = img.astype('float16')
            img = torch.from_numpy(img).permute(2, 0, 1)
            label = label.astype('int64')
            if len(img.shape) == 2:  # (height, width) Expand dims if the data is grayscale (single channel)
                img = np.expand_dims(img, axis=-1)
            
            return img, label

        if self.MODE == 'val':
            data = self.val_filenames[index]

            with np.load(data) as npz_file: 
                img = npz_file['a']
                label = npz_file['b']
            img = img.astype('float16')
            img = torch.from_numpy(img).permute(2, 0, 1)
            label = label.astype('int64')
            if len(img.shape) == 2:  
                img = np.expand_dims(img, axis=-1)
            
            return img, label

        if self.MODE == 'test':
            data = self.test_filenames[index]

            with np.load(data) as npz_file: 
                img = npz_file['a']
                label = npz_file['b']
            img = img.astype('float16')
            img = torch.from_numpy(img).permute(2, 0, 1)
            label = label.astype('int64')
            if len(img.shape) == 2:  
                img = np.expand_dims(img, axis=-1)
            
            return img, label
        

    def __len__(self):
        if self.MODE == 'train':
            return len(self.train_filenames)

        if self.MODE == 'val':
            return len(self.val_filenames)

        if self.MODE == 'test':
            return len(self.test_filenames)
        