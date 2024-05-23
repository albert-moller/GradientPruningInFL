import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from src.idlg_modified import iDLG
from src.metrics import Metrics
import matplotlib.pyplot as plt
from tqdm import tqdm

def prepare_tensor_for_plotting(tensor):
    np_image = tensor.cpu().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    if np_image.min() < 0 or np_image.max() > 1:
        np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
    return np_image

def test_accuracy(model, test_dataloader, device):
    model = model.to(device)
    model.eval()
    num_correct = 0
    total = 0
    with torch.no_grad():
        for (index, (img, label)) in enumerate(test_dataloader):
            img = img.to(device)
            label = label.to(device)
            predict = model(img)
            num_correct += torch.sum(torch.argmax(predict, dim=1) == label).item()
            total += img.shape[0]
    accuracy = num_correct / total
    return accuracy

def train_accuracy(model, train_dataloader, device):
    model = model.to(device)
    model.eval()
    num_correct = 0
    total = 0
    with torch.no_grad():
        for (index, (img, label)) in enumerate(train_dataloader):
            img = img.to(device)
            label = label.to(device)
            predict = model(img)
            num_correct += torch.sum(torch.argmax(predict, dim=1) == label).item()
            total += img.shape[0]
    accuracy = num_correct / total
    return accuracy

def prune_gradients(model, thres, alpha):
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients += list(param.grad.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(gradients, thres)
    for param in model.parameters():
        if param.grad is not None:
            grad_above_thresh = param.grad.data.abs() > threshold
            param.grad.data[grad_above_thresh] *= alpha

def prepare_tensor_for_plotting(tensor):
    np_image = tensor.cpu().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    if np_image.min() < 0 or np_image.max() > 1:
        np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
    return np_image

def train_client(id, global_round_num, client_dataloader, global_model, num_local_epochs, lr, device, criterion, filtered_train_dataset, alpha, idlg = True, prune=False):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

    #label mapping for CIFAR5
    label_mapping = {0: 0, 9: 1, 6: 2, 1: 3, 8: 4}

    #parameters for pruning
    thres = 95

    for epoch in range(num_local_epochs):
        for (index, (img, label)) in enumerate(client_dataloader):
            img, label = img.to(device), torch.tensor([label_mapping[l.item()] for l in label], device=device)
            optimizer.zero_grad()
            predict = local_model(img)
            loss = criterion(predict, label)
            loss.backward()
            # apply gradient pruning optionally
            if prune:
                prune_gradients(local_model, thres=thres, alpha=alpha)
            optimizer.step()        

        if epoch == 0 and idlg is True and global_round_num == 0:

            #Perform gradient inversion attack using iDLG:
            psnr_vals_attempts = []
            ssim_vals_attempts = []

            for _ in range(15):
                #Choose image 18 for reconstruction
                image, label = filtered_train_dataset[18]
                gt_data = image.to(device)
                gt_data = gt_data.view(1, *gt_data.size())
                gt_label = torch.Tensor([label_mapping[label]]).long().to(device)
                gt_label = gt_label.view(1,)
                idlg = iDLG(model = local_model, orig_img=image, gt_data=gt_data, label=gt_label, device=device)
                dummy_data, label_pred, history, losses, ssim_vals, psnr_vals, mse_vals = idlg.attack()
            
                psnr_vals_attempts.append(psnr_vals)
                ssim_vals_attempts.append(ssim_vals)

    return dummy_data, local_model, image, history, psnr_vals_attempts, ssim_vals_attempts, mse_vals

def global_model_average(curr, next, scale):
    if curr == None:
        curr = next
        for key in curr:
            curr[key] = curr[key]*scale
    else:
        for key in curr:
            curr[key] = curr[key] + (next[key]*scale)  
    return curr

def federated_learning_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, device, criterion, test_dataloader, train_dataloader, filtered_train_dataset, alpha, idlg = True, prune=False):

    for round in range(max_rounds):
        clients = np.random.choice(np.arange(5), num_clients_per_round, replace=False)
        global_model.eval()
        global_model = global_model.to(device)
        running_avg = None 

        for index, client in enumerate(clients):
            print(f"round {round}, starting client {(index+1)}/{num_clients_per_round}, id: {client}")
            dummy_data, local_model, image, history, psnr_vals, ssim_vals, mse_vals = train_client(client, round, client_train_loader[client], global_model, num_local_epochs, lr, device=device, criterion=criterion, filtered_train_dataset=filtered_train_dataset[client], idlg=idlg, prune=prune, alpha=alpha)
            running_avg = global_model_average(running_avg, local_model.state_dict(), 1/num_clients_per_round) 

    return dummy_data, image, history, ssim_vals, psnr_vals, mse_vals