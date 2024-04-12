import numpy as np
import copy
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset

def plot_class_distribution(client_dataloaders, selected_classes):
    
    num_clients = len(client_dataloaders)
    num_selected_classes = len(selected_classes)
    class_counts = {class_label: np.zeros(num_clients) for class_label in selected_classes}

    for client_id, dataloader in enumerate(client_dataloaders):
        for _, labels in dataloader:
            labels_np = labels.numpy() 
            for class_label in selected_classes:
                class_counts[class_label][client_id] += np.sum(labels_np == class_label)

    clients = [f'Client{idx+1}' for idx in range(num_clients)]
    bottom = np.zeros(num_clients)
    colors = plt.cm.get_cmap('viridis', num_selected_classes)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # for idx, class_label in enumerate(selected_classes):
    #     ax.bar(clients, class_counts[class_label], bottom=bottom,
    #            label=f'Class {class_label}', color=colors(idx))
    #     bottom += class_counts[class_label]

    # ax.set_ylabel('Count')
    # ax.set_title('Class distribution across clients')
    # ax.legend(title="Classes")
    #plt.savefig("./class_distribution.png")

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

