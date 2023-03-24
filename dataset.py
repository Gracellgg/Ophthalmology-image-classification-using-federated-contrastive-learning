# this is the dataset class for the cataract dataset
# it is used to load the dataset and return the data in batches
# it also contains the code for data augmentation

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd

# randomly split the csv file equally into client_num clients, return a list of csv files
def split_csv_to_clients(data_dir, client_num):
    # split the csv file into client_num clients
    csv = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    client_csv = []
    for i in range(client_num):
        client_csv.append(pd.DataFrame(columns=csv.columns))
    #print(pd.DataFrame(csv.iloc[0]).transpose())

    for i in range(len(csv)):
        client_csv[i % client_num] = pd.concat([client_csv[i % client_num], pd.DataFrame(csv.iloc[i]).transpose()]).reset_index(drop=True)
        #client_csv[i % client_num] = client_csv[i % client_num].append(csv.iloc[i])
    return client_csv


def get_csv(data_dir):
    if not os.path.exists(os.path.join(data_dir, 'train.csv')):
        data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)
        train_data.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
        valid_data.to_csv(os.path.join(data_dir, 'valid.csv'), index=False)

class MyDataset(Dataset):
    def __init__(self, data_dir, img_size, csv_file=None, train=True, transform=None, return_dir=False):
        """
        data_dir: path to the data directory
        transform: optional transform to be applied on a sample
        file structure:
        data_dir
        ├── data.csv
        └── data
            ├──1.png
            ├──2.png
            └── ...
        """
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        self.return_dir = return_dir

        if train:
            self.csv = csv_file
        else:
            self.csv = pd.read_csv(os.path.join(self.data_dir, 'valid.csv'))

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read img_dir in the data_dir column of the csv file
        img_dir = self.csv['data_dir'][idx]


        #img_dir = self.csv.iloc[idx, 0]

        data = cv2.imread(os.path.join(self.data_dir, 'data_copy', img_dir), cv2.IMREAD_GRAYSCALE)

        # resize to 128*128
        data = cv2.resize(data, (self.img_size, self.img_size))

        #label = self.csv.iloc[idx, 1]
        label = self.csv['label'][idx]

        if self.transform:
            data = self.transform(data)

        if self.return_dir:
            return data, label, img_dir



        return data, label
