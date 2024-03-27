import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from src.idlg import iDLG
import matplotlib.pyplot as plt

def validation(model, test_dataloader, device):
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


def train_client(id, client_dataloader, global_model, num_local_epochs, lr, device, criterion, filtered_train_dataset, prune=False):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

    #parameters for pruning
    thres = 30
    alpha = 0.1

    for epoch in range(num_local_epochs):
        for (index, (img, label)) in enumerate(client_dataloader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            predict = local_model(img)
            loss = criterion(predict, label)
            loss.backward()
            # apply gradient pruning optionally
            if prune:
                prune_gradients(local_model, thres=thres, alpha=alpha)
            optimizer.step()        

        #Perform gradient inversion attack using iDLG:
        img_index = 0
        image, label = filtered_train_dataset[img_index]
        tp = transforms.ToTensor()
        tt = transforms.ToPILImage()
       
        gt_data = image.to(device)
        gt_data = gt_data.view(1, *gt_data.size())

        gt_label = torch.Tensor([label]).long().to(device)
        gt_label = gt_label.view(1,)

        idlg = iDLG(model = local_model, gt_data=gt_data, label=gt_label, device=device)
        dummy_data, history, losses = idlg.attack()
        iteration = 300
        
        print("\n")
        print(f"Client {id}, epoch {epoch}, reconstruction performance using iDLG:")

        plt.figure(figsize=(12,8))
        for i in range(int(iteration/10)):
            plt.subplot(int(iteration/100),10,i+1)
            plt.imshow(history[i])
            plt.title("iter=%d"%(i*10))
            plt.axis('off')
        plt.show()

    return local_model

def global_model_average(curr, next, scale):
    if curr == None:
        curr = next
        for key in curr:
            curr[key] = curr[key]*scale
    else:
        for key in curr:
            curr[key] = curr[key] + (next[key]*scale)  
    return curr

def federated_learning_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, device, criterion, test_dataloader, filtered_train_dataset, prune=False):
    round_train_accuracy = []
    for round in range(max_rounds):
        print(f"Round {round} is starting")
        clients = np.random.choice(np.arange(5), num_clients_per_round, replace=False)
        print(f"Clients for round {round} are: {clients}")
        global_model.eval()
        global_model = global_model.to(device)
        running_avg = None 

        for index, client in enumerate(clients):
            print(f"round {round}, starting client {(index+1)}/{num_clients_per_round}, id: {client}")
            local_model = train_client(client, client_train_loader[client], global_model, num_local_epochs, lr, device=device, criterion=criterion, filtered_train_dataset=filtered_train_dataset, prune=prune)
            running_avg = global_model_average(running_avg, local_model.state_dict(), 1/num_clients_per_round) 

        global_model.load_state_dict(running_avg)
        validation_accuracy = validation(global_model, test_dataloader, device)
        print(f"Round {round}, validation accuracy: {validation_accuracy*100} %")
        round_train_accuracy.append(validation_accuracy)
    
    return round_train_accuracy