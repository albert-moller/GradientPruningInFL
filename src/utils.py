import numpy as np
import copy
import torch

def cifar_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    remainder = len(dataset) % num_users

    for i in range(num_users):
        if i < remainder:
            extra_item = 1
        else:
            extra_item = 0
        dict_users[i] = set(np.random.choice(all_idxs, num_items + extra_item, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

