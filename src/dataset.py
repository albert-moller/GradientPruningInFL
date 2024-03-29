import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset, random_split
import torchvision
import random

def get_indices(dataset, classes):
    """
    This function retrieves indices of dataset samples that belong to the specified classes.
    """
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in classes:
            indices.append(i)
    return indices

class CIFARDataset:
    def __init__(self, batch_size, num_clients, top_5_classes_indices):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
                transforms.ToTensor()
        ])

        if top_5_classes_indices is None:
            self.train_dataset = datasets.CIFAR10(root = './data', train=True, download=True, transform=self.transform)
            self.validation_dataset = datasets.CIFAR10(root = './data', train=False, download=True, transform=self.transform)
        else:
    
            full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
            full_validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
            
            train_indices = get_indices(full_train_dataset, top_5_classes_indices)
            validation_indices = get_indices(full_validation_dataset, top_5_classes_indices)

            self.train_dataset = Subset(full_train_dataset, train_indices)
            self.validation_dataset = Subset(full_validation_dataset, validation_indices)
            

    def get_dataset(self):
        return self.train_dataset, self.validation_dataset
    
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

def non_iid_dataloader(data, batch_size, num_clients, num_shard_p_client = 5):

    img, label = data_to_tensor(data)
    idx = torch.argsort(label)
    img = img[idx]
    label = label[idx]

    m = len(data)
    num_shards = num_clients * num_shard_p_client
    m_per_shard = m // num_shards
    assert m % m_per_shard == 0

    idx_shards = [torch.arange(m_per_shard*i, m_per_shard*(i+1)) for i in range(num_shards)]
    random.shuffle(idx_shards)

    client_data = []

    for i in range(num_clients):
        shards_per_client = torch.cat([idx_shards[j] for j in range(i*num_shard_p_client, (i+1)*num_shard_p_client)])
        client_dataset = TensorDataset(img[shards_per_client], label[shards_per_client])
        client_data.append(client_dataset)

    client_dataloader = [DataLoader(x, batch_size=batch_size, shuffle=True) for x in client_data]
    return client_dataloader

def filter_dataset_by_class(train_dataset, class_indices, images_per_class):
    class_counts = {class_idx: 0 for class_idx in class_indices}
    filtered_indices = []

    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        if label in class_counts and class_counts[label] < images_per_class:
            filtered_indices.append(idx)
            class_counts[label] += 1
            if all(count == images_per_class for count in class_counts.values()):
                break  # Stop early if we've collected enough images for each class

    return Subset(train_dataset, filtered_indices)


