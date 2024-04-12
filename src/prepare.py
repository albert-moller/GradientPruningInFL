from src.dataset import CIFARDataset, iid_dataloader, non_iid_dataloader, filter_dataset_by_class
from src.lenet import LeNet, weights_init
from src.utils import plot_class_distribution
from torchvision import datasets, transforms
from torch.autograd import grad
import os
import numpy as np
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

def prepare():
    #Set seeds:
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)

    top_5_classes_indices = [0, 9, 6, 1, 8]
    device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)

    batch_size = 50
    num_clients = 5
    cifar_data = CIFARDataset(batch_size=batch_size, num_clients=num_clients, top_5_classes_indices=top_5_classes_indices)
    train_dataset, validation_dataset = cifar_data.get_dataset()

    iid_client_train_loader = iid_dataloader(train_dataset, batch_size=batch_size, num_clients=num_clients)
    plot_class_distribution(iid_client_train_loader, top_5_classes_indices)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    num_images_per_class = 5
    clients_dataset = {}

    for index, client_loader in enumerate(iid_client_train_loader):
        client_dataset = client_loader.dataset
        clients_dataset[index] = filter_dataset_by_class(client_dataset, top_5_classes_indices, num_images_per_class)

    #Intialize model:
    global_model = LeNet().to(device)
    global_model.apply(weights_init)

    return iid_client_train_loader, device, criterion, validation_loader, train_loader, clients_dataset, global_model