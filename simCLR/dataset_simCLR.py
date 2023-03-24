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
from torchvision.transforms import RandomApply, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, \
    Normalize, Grayscale
from PIL import Image


# randomly split the csv file equally into client_num clients, return a list of csv files
def split_csv_to_clients(data_dir, client_num, dataset='as-oct'):
    # split the csv file into client_num clients
    csv = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    client_csv = []
    for i in range(client_num):
        client_csv.append(pd.DataFrame(columns=csv.columns))
    if dataset == 'as-oct':
        # print(pd.DataFrame(csv.iloc[0]).transpose())
        for i in range(len(csv)):
            client_csv[i % client_num] = pd.concat(
                [client_csv[i % client_num], pd.DataFrame(csv.iloc[i]).transpose()]).reset_index(drop=True)
            # client_csv[i % client_num] = client_csv[i % client_num].append(csv.iloc[i])
        return client_csv
    elif dataset == 'messidor':
        for i in range(len(csv)):
            client_csv[csv['department'][i]] = pd.concat(
                [client_csv[csv['department'][i]], pd.DataFrame(csv.iloc[i]).transpose()]).reset_index(drop=True)
        return client_csv


def get_csv(data_dir):
    if not os.path.exists(os.path.join(data_dir, 'train.csv')):
        data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
        train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)
        train_data.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
        valid_data.to_csv(os.path.join(data_dir, 'valid.csv'), index=False)

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


class ASOCT_Dataset(Dataset):
    def __init__(self, data_dir, img_size, csv_file=None, train=True, transform=None, return_dir=False):

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

        data = cv2.imread(os.path.join(self.data_dir, 'data', img_dir), cv2.IMREAD_GRAYSCALE)
        # resize to 128*128
        data = cv2.resize(data, (self.img_size, self.img_size))

        label = self.csv['label'][idx]

        transform = transforms.Compose([
            RandomApply([RandomResizedCrop(size=(128, 128), scale=(0.2, 1.0)), RandomHorizontalFlip()], p=0.5),
            RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])

        # if self.transform:
        #     data = Image.fromarray(data)
        #     data = transform(data)

        if self.return_dir:
            data = Image.fromarray(data)
            return transform(data), label, img_dir

        data = Image.fromarray(data)

        return transform(data), transform(data)


class Messidor_Dataset(Dataset):
    def __init__(self, data_dir, img_size, csv_file=None, train=True, transform=None, return_dir=False,
                 label_type='Retinopathy grade'):

        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        self.return_dir = return_dir
        self.label_type = label_type

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
        img_dir = self.csv['Image name'][idx]

        data = cv2.imread(os.path.join(self.data_dir, 'data', img_dir))
        # resize to 128*128
        data = cv2.resize(data, (self.img_size, self.img_size))



        if self.label_type == 'Retinopathy grade':
            label = self.csv['Retinopathy grade'][idx]
        elif self.label_type == 'Risk of macular edema':
            label = self.csv['Risk of macular edema '][idx]

        # label = self.csv['label'][idx]

        transform = transforms.Compose([
            RandomApply([RandomResizedCrop(size=(128, 128), scale=(0.2, 1.0)), RandomHorizontalFlip()], p=0.5),
            RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])

        # Define the transform function
        transform = transforms.Compose([transforms.RandomResizedCrop(size=128, scale=(0.2, 1.)),
                                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.RandomApply(
                                            [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])

        # if self.transform:
        #     data = Image.fromarray(data)
        #     data = transform(data)

        if self.return_dir:
            data = Image.fromarray(data)
            return transform(data), label, img_dir

        data = Image.fromarray(data)

        return transform(data), transform(data)
