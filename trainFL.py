# this is the main file for training the federated learning model
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MyDataset
import torchvision.models as models
from tqdm import tqdm
from dataset import get_csv, split_csv_to_clients

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--data_dir', type=str, default='D:\\PycharmProjects\\fairness\\pics\\cataract\\more_pic\\')
    parser.add_argument('--model_dir', type=str, default='D:/PycharmProjects/fairness/pics/cataract/more_pic/')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--log_name', type=str, default='log.txt')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=64)
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
    return loss_avg / len(train_loader), accuracy_avg / len(train_loader), model.state_dict()

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

def main():
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
    # get csv file
    get_csv(data_dir)
    # split csv file to clients
    client_list = split_csv_to_clients(data_dir, config.num_clients)
    # Create model
    model = models.resnet18(pretrained=True)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, config.num_classes)
    model = model.to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model on client
    best_accuracy = 0
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
            train_dataset = MyDataset(data_dir, img_size=config.image_size, csv_file=client_list[step],
                                      train=True,transform=transform)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            model.load_state_dict(old_model_dict)
            loss, accuracy, local_model_dict = train(model, train_loader, criterion, optimizer, device)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Training Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, step + 1, total_step, loss, accuracy * 100))
            # aggregate the model_dict
            for key in new_model_dict.keys():
                new_model_dict[key] += local_model_dict[key]
        print('model aggregation...')
        # update the model
        for key in new_model_dict.keys():
            new_model_dict[key] /= total_step
        model.load_state_dict(new_model_dict)

        # validate the model
        print('model validation...')
        valid_dataset = MyDataset(data_dir, img_size=config.image_size, transform=transform, train=False)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
        loss, accuracy = validate(model, valid_loader, criterion, device)
        print('Epoch [{}/{}], Loss: {:.4f}, Validation Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, loss, accuracy * 100))


        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state_dict = model.state_dict()

        # save the best model
        if (epoch + 1) % 5 == 0:
            torch.save(best_state_dict, os.path.join(config.model_dir, 'model_{}.pt'.format(epoch + 1)))





if __name__ == '__main__':
    main()










