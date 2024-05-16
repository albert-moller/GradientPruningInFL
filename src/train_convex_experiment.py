import copy
import numpy as np
import torch
from src.idlg_modified import iDLG
from src.metrics import Metrics
from tqdm import tqdm
import matplotlib.pyplot as plt

def prepare_tensor_for_plotting(tensor):
    np_image = tensor.cpu().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    if np_image.min() < 0 or np_image.max() > 1:
        np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
    return np_image

class ModifiediDLG(iDLG):

    def attack(self, seed = None):
        if seed is not None:
            torch.manual_seed(seed)

        gt_onehot_label = self.label_to_onehot(self.label)
        iteration = 300

        #iDLG training image reconstruction:
        self.model.eval()
        predicted = self.model(self.gt_data)
        loss = self.criterion(predicted, gt_onehot_label)
        dy_dx = torch.autograd.grad(loss, self.model.parameters())
        orig_dy_dx = list((_.detach().clone() for _ in dy_dx))
        dummy_data = torch.randn(self.gt_data.size()).to(self.device).requires_grad_(True)
        label_pred = torch.argmin(torch.sum(orig_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
        optimizer = torch.optim.LBFGS([dummy_data, ])
        history = []
        losses = []

        final_grad_diff = None
        psnr_vals = []
        ssim_vals = []

        for iters in range(iteration):
            def closure():
                optimizer.zero_grad()
                dummy_pred = self.model(dummy_data)
                dummy_loss = self.criterion(dummy_pred, label_pred)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, orig_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff
    
            optimizer.step(closure)

            if iters%10==0:
                current_loss = closure()
                losses.append(current_loss)
                history.append(self.tt(dummy_data[0].cpu()))
                final_grad_diff = current_loss

                #Compute metrics
                metrics = Metrics(ground_truth_imgs=[self.orig_img], reconstructed_imgs=[self.tt(dummy_data[0].cpu())])
                ssim = metrics.compute_ssim()
                psnr = metrics.compute_psnr()
                ssim_vals.append(ssim)
                psnr_vals.append(psnr)

        return dummy_data, label_pred, history, losses, final_grad_diff, psnr_vals, ssim_vals
    
class TrainExperiment:
    def __init__(self, global_model, num_local_epochs, lr, criterion, client_dataloader, device, filtered_train_dataset, prune, alpha):
        self.label_mapping = {0: 0, 9: 1, 6: 2, 1: 3, 8: 4}
        self.device = device
        self.local_model = copy.deepcopy(global_model).to(self.device)
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr)
        self.criterion = criterion
        self.client_dataloader = client_dataloader
        self.filtered_train_dataset = filtered_train_dataset
        self.prune = prune
        self.alpha = alpha
        self.num_local_epochs = num_local_epochs
        self.thres = 0.95

    def prune_gradients(self):
        gradients = []
        for param in self.local_model.parameters():
            if param.grad is not None:
                gradients += list(param.grad.cpu().data.abs().numpy().flatten())
        threshold = np.percentile(gradients, self.thres)
        for param in self.local_model.parameters():
            if param.grad is not None:
                grad_above_thresh = param.grad.data.abs() > threshold
                param.grad.data[grad_above_thresh] *= self.alpha

    def train_and_attack(self):
        #Step 1) Train a local model
        for epoch in range(self.num_local_epochs):
            for (index, (img, label)) in enumerate(self.client_dataloader):
                img, label = img.to(self.device), torch.tensor([self.label_mapping[l.item()] for l in label], device=self.device)
                self.optimizer.zero_grad()
                predict = self.local_model(img)
                loss = self.criterion(predict, label)
                loss.backward()
                # apply gradient pruning optionally
                if self.prune:
                    self.prune_gradients()
                self.optimizer.step()    

        #Step 2) Perform iDLG reconstruction on the same image using 10 different randomly initialized dummy data
        
        #Choose image 18 for reconstruction (Silver car)
        image, label = self.filtered_train_dataset[14]
        gt_data = image.to(self.device)
        gt_data = gt_data.view(1, *gt_data.size())
        gt_label = torch.Tensor([self.label_mapping[label]]).long().to(self.device)
        gt_label = gt_label.view(1,)

        #Step 3) Collect gradient differences, psnr values and ssim values

        grad_diffs = []
        psnr_values = []
        ssim_values = []

        for _ in tqdm(range(50), desc="Performing iDLG for different dummy data"):
            idlg = ModifiediDLG(model=self.local_model, gt_data = gt_data, label=gt_label, device=self.device, orig_img=image)
            dummy_data, label_pred, history, losses, final_grad_diff, psnr_vals, ssim_vals = idlg.attack()
            grad_diffs.append(final_grad_diff.detach().cpu().item())
            psnr_values.append(psnr_vals)
            ssim_values.append(ssim_vals)
        
        return grad_diffs, psnr_values, ssim_values


