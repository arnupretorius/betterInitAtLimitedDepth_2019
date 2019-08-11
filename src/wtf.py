import torch
import torch.nn as nn
import numpy as np
import copy
from torchvision import datasets, transforms

from src.data_loader import load_data
from src.loss_landscape import loss_landscape_step_2d, create_pca_directions, loss_landscape_pca_2d, \
    project_trajectories, tensorlist_to_tensor
from src.net import Net
from src.utils import test
import matplotlib.pyplot as plt


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    n_in = 784
    n_hidden = 100
    n_out = 10
    n_layer = 10

    seed = 42
    init = 1

    n_epoch = 20

    torch.manual_seed(seed)


    dxs = list()
    dys = list()

    for seed in [42,43,44]:
        path = '../models/mnist/seed_{}_init_{}'.format(seed, init)
        d = np.load('{}/pca_directions.npy.npz'.format(path))
        dx, dy = d['arr_0'], d['arr_1']
        dxs.append(dx)
        dys.append(dy)

    import itertools
    for (x, y) in itertools.combinations(dxs, 2):
        dist = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        print(dist)

    print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh")

    for (x, y) in itertools.combinations(dys, 2):
        dist = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        print(dist)


