import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from utils import cifar_iid


class CIFARDataset:
    def __init__(self, batch_size, num_clients):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
                transforms.ToTensor()
        ])
        self.train_dataset = datasets.CIFAR10(root = './data', train=True, download=True, transform=self.transform)
        self.validation_dataset = datasets.CIFAR10(root = './data', train=False, download=True, transform=self.transform)
        self.user_groups = cifar_iid(self.train_dataset, num_clients)

    def get_dataset(self):
        return self.train_dataset, self.validation_dataset, self.user_groups
    
class SplitDataset(Dataset):

    def __init__(self, dataset, user_idx):
        self.dataset = dataset
        self.user_idx = user_idx

    def __len__(self):
        return len(self.user_idx)
    
    def __getitem__(self, i):
        img, label = self.dataset[self.user_idx[i]]
        img = torch.tensor(img)
        label = torch.tensor(label)
        return img, label
    


