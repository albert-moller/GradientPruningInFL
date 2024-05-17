import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms

from src.train_convex_experiment import ModifiediDLG
from src.metrics import Metrics
import matplotlib.pyplot as plt
from tqdm import tqdm

class FederatedLearning:

    def __init__(self, model, clients_dataloaders, num_clients_per_round, num_local_epochs, lr, device, criterion, filtered_train_dataset, train_dataloader, test_dataloader, max_rounds, alpha) -> None:
        
        self.global_model = model
        self.clients_dataloader = clients_dataloaders
        self.num_local_epochs = num_local_epochs
        self.lr = lr
        self.device = device
        self.criterion = criterion
        self.filtered_train_dataset = filtered_train_dataset
        self.alpha = alpha
        self.num_clients_per_round = num_clients_per_round
        self.test_dataloader, self.train_dataloader = train_dataloader, test_dataloader
        self.max_rounds = max_rounds

    def prune_gradients(self, model, thres, alpha):
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients += list(param.grad.cpu().data.abs().numpy().flatten())
        threshold = np.percentile(gradients, thres)
        for param in model.parameters():
            if param.grad is not None:
                grad_above_thresh = param.grad.data.abs() > threshold
                param.grad.data[grad_above_thresh] *= alpha

    def prepare_tensor_for_plotting(self, tensor):
        np_image = tensor.cpu().numpy()
        np_image = np.transpose(np_image, (1, 2, 0))
        if np_image.min() < 0 or np_image.max() > 1:
            np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
        return np_image

    def train_client(self, id, global_round_num, client_dataloader, filtered_dataset, global_model):
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()
        optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr)

        #parameters for pruning
        thres = 95

        label_mapping = {0: 0, 9: 1, 6: 2, 1: 3, 8: 4}
        psnr = None
        ssim = None

        for epoch in range(self.num_local_epochs):
            for (index, (img, label)) in enumerate(client_dataloader):
                img, label = img.to(self.device), torch.tensor([label_mapping[l.item()] for l in label], device=self.device)
                optimizer.zero_grad()
                predict = local_model(img)
                loss = self.criterion(predict, label)
                loss.backward()
                # apply gradient pruning optionally
                if self.alpha is not None:
                    self.prune_gradients(local_model, thres=thres, alpha=self.alpha)
                optimizer.step()        

            if epoch == 0 and global_round_num == 0 and id == 1:

                #Perform gradient inversion attack using iDLG:
                reconstructed_imgs = []
                ground_truth_imgs = []
        
                for idx in tqdm(range(len(filtered_dataset)), desc="Reconstructing training images using iDLG"):
                    
                    #Perform multiple reconstruction due to iDLG solving a non-convex optimization problem
                    results = {}
                    for attempt in range(15):

                        image, label = filtered_dataset[idx]
                        gt_data = image.to(self.device)
                        gt_data = gt_data.view(1, *gt_data.size())
                        gt_label = torch.Tensor([label_mapping[label]]).long().to(self.device)
                        gt_label = gt_label.view(1,)

                        idlg = ModifiediDLG(model = local_model, orig_img=image, gt_data=gt_data,label=gt_label,device=self.device)
                        dummy_data, label_pred, history, losses, final_grad_diff, psnr_vals, ssim_vals = idlg.attack()
                        results[attempt] = (history[-1], final_grad_diff, psnr_vals[-1], ssim_vals[-1])

                    best_attempt = min(results, key=lambda attempt: results[attempt][1])
                    best_reconstructed_image = results[best_attempt][0]
                    reconstructed_imgs.append(best_reconstructed_image)
                    ground_truth_imgs.append(image)

                total_psnr = sum(map(lambda attempt: results[attempt][2], results))
                total_ssim = sum(map(lambda attempt: results[attempt][3], results))
                num_attempts = len(results)

                psnr = total_psnr / num_attempts
                ssim = total_ssim / num_attempts

                print("PSNR: ", psnr)
                print("SSIM: ", ssim)

                # metrics = Metrics(ground_truth_imgs, reconstructed_imgs)
                # psnr = metrics.compute_psnr()
                # ssim = metrics.compute_ssim()

                ground_truth_imgs_for_plotting = [self.prepare_tensor_for_plotting(img.squeeze(0)) for img in ground_truth_imgs]
                
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
                
                if self.alpha is None:
                    save_path = os.path.join(os.getcwd(), "plots", "original")
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, f"client_{id}_iDLG.png")

                else:
                    save_path = os.path.join(os.getcwd(), "plots", f"alpha_{self.alpha}")
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, f"client_{id}_iDLG.png")
                    
                plt.savefig(save_path, dpi = 800)
                plt.close()
        
        return local_model, psnr, ssim
    
    def global_model_average(self, curr, next, scale):
        if curr == None:
            curr = next
            for key in curr:
                curr[key] = curr[key]*scale
        else:
            for key in curr:
                curr[key] = curr[key] + (next[key]*scale)  
        return curr
    
    def test_accuracy(self, model, test_dataloader, device):
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

    def train_accuracy(self, model, train_dataloader, device):
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
    
    def federated_learning_experiment(self):
        round_train_accuracy = []
        round_test_accuracy = []
        ssim_vals = []
        psnr_vals = []

        global_model = self.global_model

        for round in range(self.max_rounds):
            clients = np.random.choice(np.arange(5), self.num_clients_per_round, replace=False)
            global_model.eval()
            global_model = global_model.to(self.device)
            running_avg = None 

            client_ssim = []
            client_psnr = []
     
            for index, client in enumerate(clients):
                local_model, psnr, ssim = self.train_client(client, round, self.clients_dataloader[client], self.filtered_train_dataset[client], global_model)
                running_avg = self.global_model_average(running_avg, local_model.state_dict(), 1/self.num_clients_per_round) 

                if ssim is not None:
                    client_ssim.append(ssim)
                if psnr is not None:
                    client_psnr.append(psnr)

            global_model.load_state_dict(running_avg)
            test_accuracy_ = self.test_accuracy(global_model, self.test_dataloader, self.device)
            train_accuracy_ = self.train_accuracy(global_model, self.train_dataloader, self.device)
            round_train_accuracy.append(train_accuracy_)
            round_test_accuracy.append(test_accuracy_)
            ssim_vals.append(np.mean(client_ssim))
            psnr_vals.append(np.mean(client_psnr))
        
        return round_train_accuracy, round_test_accuracy, psnr_vals, ssim_vals