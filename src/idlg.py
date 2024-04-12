import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

class iDLG:
    def __init__(self, model, gt_data, label, device):
        self.model = model.to(device)
        self.gt_data = gt_data.to(device)
        self.label = label.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.tp = transforms.ToTensor()
        self.tt = transforms.ToPILImage()

    @staticmethod
    def label_to_onehot(target, num_classes=5):
        target = torch.unsqueeze(target, 1)
        onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
        onehot_target.scatter_(1, target, 1)
        return onehot_target

    def attack(self):
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

        return dummy_data, label_pred, history, losses
















        
    