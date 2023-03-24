# this is the main file for training the federated learning model
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_simCLR import ASOCT_Dataset, Messidor_Dataset
import torchvision.models as models
from tqdm import tqdm
from dataset_simCLR import get_csv, split_csv_to_clients
from torchvision.transforms import RandomApply, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, \
    Normalize, Grayscale

from simclr import SimCLR, ContrastiveLoss

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--datasets', type=str, default='as-oct', choices=['as-oct', 'messidor'])
    # parser.add_argument('--data_dir', type=str, default='D:/PycharmProjects/fairness/pics/Messdior dataset/'
    #                     , choices=['D:/PycharmProjects/fairness/pics/cataract/all_pic/',
    #                                'D:/PycharmProjects/fairness/pics/Messdior dataset/'])
    #parser.add_argument('--model_dir', type=str, default='D:/PycharmProjects/fairness/pics/cataract/all_pic/')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--log_name', type=str, default='log.txt')
    #parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=128)
    #parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    return parser.parse_args()


def train_simclr(simclr, dataloader, optimizer, criterion, device):
    simclr.train()
    loss_avg = 0.0
    for x_i, x_j in tqdm(dataloader):
        x_i, x_j = x_i.to(device), x_j.to(device)
        optimizer.zero_grad()
        z_i, z_j = simclr(x_i), simclr(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()
    return loss_avg / len(dataloader), simclr.state_dict()


def main():
    config = parse_args()
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyper-parameters
    num_epochs = config.num_epochs
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    #data_dir = config.data_dir
    if config.datasets == 'as-oct':
        data_dir = 'D:/PycharmProjects/fairness/pics/cataract/all_pic/'
        num_classes = 3
        num_clients = 5
    elif config.datasets == 'messidor':
        data_dir = 'D:/PycharmProjects/fairness/pics/Messdior dataset/'
        num_classes = 4
        num_clients = 3

    # Image preprocessing modules
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    transform = transforms.Compose([
        RandomApply([RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0)), RandomHorizontalFlip()], p=0.5),
        RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    # get csv file
    get_csv(data_dir)
    # split csv file to clients
    client_list = split_csv_to_clients(data_dir, num_clients, config.datasets)
    # Create model
    base_encoder = models.resnet18(pretrained=True)
    if config.datasets == 'as-oct':
        base_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = base_encoder.fc.in_features
    base_encoder.fc = nn.Linear(num_features, out_features=512)
    model = SimCLR(base_encoder).to(device)
    # Loss and optimizer
    criterion = ContrastiveLoss()
    weight_decay = 1e-6
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model on client
    best_loss = 100000
    best_state_dict = None
    total_step = len(client_list)
    for epoch in range(num_epochs):
        # train the model on client, get the model_dict, and aggregate the model_dict to get the new model_dict
        old_model_dict = model.state_dict()
        # create new model dict: a dict with the same keys as the old model dict, but all values are zeros
        new_model_dict = {}
        for key in old_model_dict.keys():
            new_model_dict[key] = torch.zeros(old_model_dict[key].size()).to(device)
        # train the model on client
        for step in range(total_step):
            if config.datasets == 'as-oct':
                train_dataset = ASOCT_Dataset(data_dir, img_size=config.image_size, csv_file=client_list[step],
                                          train=True, transform=transform)
            elif config.datasets == 'messidor':
                train_dataset = Messidor_Dataset(data_dir, img_size=config.image_size, csv_file=client_list[step],
                                          train=True, transform=transform)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            model.load_state_dict(old_model_dict)
            loss_avg, local_model_dict = train_simclr(model, train_loader, optimizer, criterion, device)
            # loss, accuracy, local_model_dict = train(model, train_loader, criterion, optimizer, device)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, step + 1, total_step, loss_avg))
            # aggregate the model_dict
            for key in new_model_dict.keys():
                new_model_dict[key] += local_model_dict[key]
        print('model aggregation...')
        # update the model
        for key in new_model_dict.keys():
            new_model_dict[key] /= total_step
        model.load_state_dict(new_model_dict)

        best_state_dict = model.state_dict()

        # save the best model
        if (epoch + 1) % 5 == 0:
            torch.save(best_state_dict, 'simCLR_model_small_{}.pt'.format(epoch + 1))


if __name__ == '__main__':
    main()
