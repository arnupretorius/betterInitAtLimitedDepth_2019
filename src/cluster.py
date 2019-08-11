import torch
import torch.nn as nn
import numpy as np
import copy
from torchvision import datasets, transforms

from src.cca_core import get_cca_similarity
from src.data_loader import load_data
from src.loss_landscape import loss_landscape_step_2d, create_pca_directions, loss_landscape_pca_2d, \
    project_trajectories, tensorlist_to_tensor
from src.net import Net
from src.utils import test
import matplotlib.pyplot as plt

import tables
import itertools

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    epochs = 20
    n_layers = 11

    # path = '../models/mnist/seed_{}_init_{}'.format(seed, init)

    correlations = np.zeros((n_layers, n_layers))

    seed1 = 44
    init1 = 2

    seed2 = 44
    init2 = 1

    for layer in range(0, n_layers):
        activation_path = '../models/mnist/seed_{}_init_{}/model_{}_act'.format(seed1, init1, epochs)
        f = tables.open_file('{}/layer_{}.h5'.format(activation_path, layer))
        act1 = np.asarray(f.root.data).T
        f.close()

        for other_layer  in range(0, n_layers):

            if layer <= other_layer:
                activation_path = '../models/mnist/seed_{}_init_{}/model_{}_act'.format(seed2, init2, epochs)
                f = tables.open_file('{}/layer_{}.h5'.format(activation_path, other_layer))
                act2 = np.asarray(f.root.data).T
                f.close()
                cca = get_cca_similarity(act1, act2, compute_dirns=False, verbose=False)
                correlations[layer, other_layer] = cca['mean'][0]
                correlations[other_layer, layer] = cca['mean'][0]
    plt.imshow(correlations, vmin=0, vmax=1)
    plt.show()


    #
    # seed = 42
    # init = 1
    #
    # n_epoch = 20
    #
    # torch.manual_seed(seed)
    #
    # epoch = 3
    #
    # metric = np.zeros((6, 6))
    # layer = 0
    # for i, (seed, init) in enumerate(itertools.product([42, 43, 44], [1, 2])):
    #     path = '../models/mnist/seed_{}_init_{}/model_{}_act'.format(seed, init, n_epoch)
    #     f = tables.open_file('{}/layer_{}.h5'.format(path, layer))
    #     act1 = np.asarray(f.root.data).T
    #     f.close()
    #
    #     for j, (seed, init) in enumerate(itertools.product([42, 43, 44], [1, 2])):
    #
    #         if i <= j:
    #             path = '../models/mnist/seed_{}_init_{}/model_{}_act'.format(seed, init, n_epoch)
    #             f = tables.open_file('{}/layer_{}.h5'.format(path, layer))
    #             act2 = np.asarray(f.root.data).T
    #             f.close()
    #
    #             metric[i, j] = 1 - get_cca_similarity(act1, act2, compute_dirns=False, verbose=False)['mean'][0]
    #             metric[j, i] = metric[i ,j]
    #
    # print(metric)
    #
    # plt.imshow(metric, vmin=0, vmax=1)
    # plt.show()