import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from src.idlg import iDLG
from src.metrics import Metrics
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_accuracy(model, test_dataloader, device):
    model = model.to(device)
    model.eval()
    num_correct = 0
    total = 0
    label_mapping = {0: 0, 9: 1, 6: 2, 1: 3, 8: 4}
    with torch.no_grad():
        for (index, (img, label)) in enumerate(test_dataloader):
            img = img.to(device)
            label = torch.tensor([label_mapping[l.item()] for l in label], device=device)
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
    label_mapping = {0: 0, 9: 1, 6: 2, 1: 3, 8: 4}
    with torch.no_grad():
        for (index, (img, label)) in enumerate(train_dataloader):
            img = img.to(device)
            label = torch.tensor([label_mapping[l.item()] for l in label], device=device)
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

    #parameters for pruning
    thres = 95

    label_mapping = {0: 0, 9: 1, 6: 2, 1: 3, 8: 4}

    mse, psnr, ssim = None, None, None

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
            reconstructed_imgs = []
            ground_truth_imgs = []
            extracted_labels = []
            ground_truth_labels = []

            for idx in tqdm(range(len(filtered_train_dataset)), desc="Reconstructing training images using iDLG"):
                img_index = idx
                image, label = filtered_train_dataset[img_index]
                ground_truth_imgs.append(image)
                ground_truth_labels.append(label)

                gt_data = image.to(device)
                gt_data = gt_data.view(1, *gt_data.size())

                gt_label = torch.Tensor([label_mapping[label]]).long().to(device)
                gt_label = gt_label.view(1,)

                idlg = iDLG(model = local_model, gt_data=gt_data, label=gt_label, device=device)
                dummy_data, label_predict, history, losses = idlg.attack()
                extracted_labels.append(label_predict)
                reconstructed_imgs.append(history[-1])

            metrics = Metrics(ground_truth_imgs, reconstructed_imgs)
            mse = metrics.compute_mse()
            psnr = metrics.compute_psnr()
            ssim = metrics.compute_ssim()

            ground_truth_imgs_for_plotting = [prepare_tensor_for_plotting(img.squeeze(0)) for img in ground_truth_imgs]
            plt.figure(figsize=(20, 10))

            for i in range(25):
                plt.subplot(5, 10, 2*i + 1)
                plt.imshow(ground_truth_imgs_for_plotting[i])
                plt.title(f"GT {i}")
                plt.axis('off')

                plt.subplot(5, 10, 2*i + 2)
                plt.imshow(reconstructed_imgs[i])
                plt.title(f"Recon {i}")
                plt.axis('off')

            plt.tight_layout()

            if alpha is None:
                save_path = os.path.join(os.getcwd(), "plots", "original")
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, f"client_{id}_iDLG.png")

            else:
                save_path = os.path.join(os.getcwd(), "plots", f"alpha_{alpha}")
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, f"client_{id}_iDLG.png")
                
            plt.savefig(save_path, dpi = 800)
            plt.close()

    return local_model, mse, psnr, ssim

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
    round_train_accuracy = []
    round_test_accuracy = []
    ssim_vals = []
    psnr_vals = []
    mse_vals = []

    for round in range(max_rounds):
        clients = np.random.choice(np.arange(5), num_clients_per_round, replace=False)
        global_model.eval()
        global_model = global_model.to(device)
        running_avg = None 

        client_ssim = []
        client_psnr = []
        client_mse = []

        for index, client in enumerate(clients):
            local_model, mse, psnr, ssim = train_client(client, round, client_train_loader[client], global_model, num_local_epochs, lr, device=device, criterion=criterion, filtered_train_dataset=filtered_train_dataset[client], idlg=idlg, prune=prune, alpha=alpha)
            running_avg = global_model_average(running_avg, local_model.state_dict(), 1/num_clients_per_round) 

            if ssim is not None:
                client_ssim.append(ssim)
            if mse is not None:
                client_mse.append(mse)
            if psnr is not None:
                client_psnr.append(psnr)

        global_model.load_state_dict(running_avg)
        test_accuracy_ = test_accuracy(global_model, test_dataloader, device)
        train_accuracy_ = train_accuracy(global_model, train_dataloader, device)
        round_train_accuracy.append(train_accuracy_)
        round_test_accuracy.append(test_accuracy_)
        ssim_vals.append(np.mean(client_ssim))
        psnr_vals.append(np.mean(client_psnr))
        mse_vals.append(np.mean(client_mse))
    
    return round_train_accuracy, round_test_accuracy, mse_vals, psnr_vals, ssim_vals