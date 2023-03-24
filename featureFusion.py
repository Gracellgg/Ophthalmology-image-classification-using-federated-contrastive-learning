# this file is to fuse the feature of the FL model and the machine learning model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MyDataset
import torchvision.models as models
from tqdm import tqdm
from dataset import get_csv, split_csv_to_clients
import argparse
import os
import pandas as pd
import numpy as np
from sklearn import svm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--data_dir', type=str, default='D:\\PycharmProjects\\fairness\\pics\\cataract\\all_pic\\')
    parser.add_argument('--model_dir', type=str, default='D:/PycharmProjects/fairness/pics/cataract/all_pic/')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--log_name', type=str, default='log.txt')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_clients', type=int, default=5)
    return parser.parse_args()


def train(model, train_loader, criterion, optimizer, device):
    loss_avg = 0
    accuracy_avg = 0
    model.train()
    for data, labels in tqdm(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()
        loss_avg += loss.item()
        accuracy_avg += accuracy.item()
    return loss_avg / len(train_loader), accuracy_avg / len(train_loader)


def validate(model, valid_loader, criterion, device):
    loss_avg = 0
    accuracy_avg = 0
    model.eval()
    with torch.no_grad():
        for data, labels in tqdm(valid_loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().mean()
            loss_avg += loss.item()
            accuracy_avg += accuracy.item()
    return loss_avg / len(valid_loader), accuracy_avg / len(valid_loader)


def fine_tune():
    config = parse_args()
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyper-parameters
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    data_dir = config.data_dir
    # Image preprocessing modules
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Create model
    model = models.resnet18(pretrained=True)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, config.num_classes)
    model = model.to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Load data
    get_csv(data_dir)
    # split csv file to clients
    client_list = split_csv_to_clients(data_dir, config.num_clients)
    valid_dataset = MyDataset(data_dir, img_size=config.image_size, train=False, transform=transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    best_accuracy = 0
    best_model_dict = None
    total_step = len(client_list)
    best_acc_list = [0] * total_step
    best_model_list = [None] * total_step
    # Train the model
    for epoch in range(num_epochs):
        for step in range(total_step):
            train_dataset = MyDataset(data_dir, img_size=config.image_size, csv_file=client_list[step],
                                      train=True, transform=transform)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, step + 1, total_step, train_loss, train_accuracy * 100))
            valid_loss, valid_accuracy = validate(model, valid_loader, criterion, device)
            print('Epoch [{}/{}], Step [{}/{}], Valid Loss: {:.4f}, Valid Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, step + 1, total_step, valid_loss, valid_accuracy * 100))
            if valid_accuracy > best_accuracy:
                best_acc_list[step] = valid_accuracy
                best_model_list[step] = model.state_dict()

    # save the best model
    for i in range(total_step):
        torch.save(best_model_list[i], config.model_dir + 'model_{}.pth'.format(i))


def feature_fusion():
    config = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client_list = split_csv_to_clients(config.data_dir, config.num_clients)
    total_step = len(client_list)
    criterion = nn.CrossEntropyLoss()
    # Image preprocessing modules
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_dict = {}
    col_num = 0


    # fuse the features of the model and the features in the csv to classify by svm

    for client in range(total_step):
        model = models.resnet18(pretrained=True)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, config.num_classes)
        model.load_state_dict(torch.load(config.model_dir + 'model_{}.pth'.format(client)))
        # remove the last layer of the model
        model.fc = nn.Sequential(*list(model.fc.children())[:-1])
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # get the features of the model using validation set, and combine the features in the csv file
        valid_dataset = MyDataset(config.data_dir, img_size=config.image_size, train=False, transform=transform, return_dir=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False)
        loss_avg = 0
        accuracy_avg = 0
        model.eval()
        with torch.no_grad():
            for data, labels, img_dir in tqdm(valid_loader):
                data = data.to(device)
                outputs = model(data)
                outputs = outputs.cpu().numpy()

                # create a csv file of all clients, frst column is the img_dir, the rest columns are the features of the model
                # add new columns in the csv file that stores the features of the model, one column for one feature number

                for i in range(len(img_dir)):
                    column_name = 'row_{}'.format(col_num)
                    col_num += 1
                    row_list = [img_dir[i]] + outputs[i].tolist()
                    data_dict[column_name] = row_list

    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['img_dir'] + ['feature_{}'.format(i) for i in range(512)])
    df.to_csv('feature_fusion.csv', index=False)













if __name__ == '__main__':
    #fine_tune()
    feature_fusion()









