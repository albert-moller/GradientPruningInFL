import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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

def train_client(id, client_dataloader, global_model, num_local_epochs, lr, device, criterion):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

    save_dir = f'saved_models/client_{id}'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_local_epochs):
        for (index, (img, label)) in enumerate(client_dataloader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            predict = local_model(img)
            loss = criterion(predict, label)
            loss.backward()
            # Optionally: prune_gradients(local_model)
            optimizer.step()
        
        model_path = os.path.join(save_dir, f'local_model_epoch_{epoch}.pt')
        torch.save(local_model.state_dict(), model_path)

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

def federated_learning_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, device, criterion, test_dataloader):
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
            local_model = train_client(client, client_train_loader[client], global_model, num_local_epochs, lr, device=device, criterion=criterion)
            running_avg = global_model_average(running_avg, local_model.state_dict(), 1/num_clients_per_round) 

        global_model.load_state_dict(running_avg)
        validation_accuracy = validation(global_model, test_dataloader, device)
        print(f"Round {round}, validation accuracy: {validation_accuracy*100} %")
        round_train_accuracy.append(validation_accuracy)
    
    return round_train_accuracy


