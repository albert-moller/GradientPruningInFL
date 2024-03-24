import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from utils import cifar_iid
import torchvision

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
    
def get_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def data_to_tensor(data):
    loader = DataLoader(data, batch_size=len(data))
    img, label = next(iter(loader))
    return img, label

def iid_dataloader(data, batch_size, num_clients):
    m = len(data)
    assert m % num_clients == 0
    m_per_client = m // num_clients
    assert m_per_client % batch_size == 0
    client_data = random_split(data, [m_per_client for _ in range(num_clients)])
    client_dataloader = [DataLoader(x, batch_size=batch_size, shuffle=True) for x in client_data]
    return client_dataloader


