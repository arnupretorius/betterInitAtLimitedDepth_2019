import os
import torch
import torch.nn as nn
import numpy as np
import copy
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# custom imports
from src.data_loader import load_data
from src.utils import load_model, recreate_model, test, hp_dict, get_experiment_dicts, make_save_path
from src.loss_landscape_utils import loss_landscape_step_2d, create_pca_directions, loss_landscape_pca_2d, project_trajectories, tensorlist_to_tensor, get_hp, get_exp_details
from src.net import Net


def get_model_paths(noise_type, noise_level, exp_num, init_index, start_epoch=0, end_epoch=None, step=1, points=None, root_dir = "../results"):
    """Return model paths of specifed experiment"""
    model_dict = {noise_type: {noise_level: {exp_num: {init_index: {}}}}}
    result_path = root_dir
    models = load_model(model_dict, result_path)
    model_paths = models[noise_type][noise_level][exp_num][init_index]

    last_epoch = int(model_paths[-1].split("/")[-1].split(".")[0].split("_")[-1])

    if points != None:
        step = int(last_epoch/points)
        if step == 0:
            step = 1

    if end_epoch != None: 
        model_paths = models[noise_type][noise_level][exp_num][init_index][start_epoch:end_epoch:step]
    else:
        model_paths = models[noise_type][noise_level][exp_num][init_index][start_epoch::step]

    return model_paths


def get_models(model_paths, device):
    """Return the contrusted models of the model paths given"""
    hp = hp_dict()

    n_in = 784
    n_hidden = hp["width"][get_hp(model_paths[0], 3)]
    n_layer = hp["depth"][get_hp(model_paths[0], 2)]
    n_out = 10

    seed = get_hp(model_paths[0], 4)
    init = get_hp(model_paths[0], 10)

    n_epoch = len(model_paths)

    torch.manual_seed(seed)

    # path = '../models/mnist/seed_{}_init_{}'.format(seed, init)

    models = [Net(n_in, n_hidden, n_out, n_layer, act='relu', init_val=2).to(device) for _ in range(n_epoch)]

    for i, model in enumerate(models):
        model.load_state_dict(torch.load(model_paths[i], map_location=device))

    return models

# Only used to plot the landscapes using PCA, not sure of others
def plot_loss_landscape(models, device, init, save_path=None, verbose=True):
    """Plotting the loss landscape and creating a file for replotting"""
    train_loader, test_loader = load_data('mnist', path='../data')
    criterion = nn.CrossEntropyLoss()
    xv, yv, loss, directions, x_coord, y_coord = loss_landscape_pca_2d(models, test_loader, criterion, device, verbose=verbose)

    # fig, ax = plt.subplots()
    # CS = ax.contour(xv, yv, loss, levels=20)
    # ax.clabel(CS, inline=1, fontsize=10)
    # ax.set_title('Loss Landscape (with PCA)')

    # ax.plot(x_coord, y_coord, marker='.')

    if save_path != None:
        print("Saving plot information in {}init_{}.npz\n".format(save_path, init))
        # np.savez('{}.npz'.format(save_path[:-4]), xv=xv, yv=yv, loss=loss, xdir=np.asarray(directions[0]), ydir=np.asarray(directions[1]), x=x_coord, y=y_coord)
        np.savez('{}init_{}.npz'.format(save_path, init), xv=xv, yv=yv, loss=loss, x=x_coord, y=y_coord)

        # No need to save figure now
        # plt.savefig('{}init_{}.png'.format(save_path, init))

    # No need to show figure
    # plt.show()


# Need to be refined to collect the file from the correct directory
def plot_loss_landscape_from_file(filename):
    """Plotting a loss landscape from an .npz file"""
    npzfile = np.load(filename)
    xv, yv, loss, x_coord, y_coord = npzfile['xv'], npzfile['yv'], npzfile['loss'], npzfile['x'], npzfile['y'], 

    fig, ax = plt.subplots()
    CS = ax.contour(xv, yv, loss, levels=20)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Loss Landscape (with PCA)')

    ax.plot(x_coord, y_coord, marker='.')

    plt.show()


# Not used at the moment, function similar to loss_landscape_collect.py
def plot_all():
    """
    Run through all experiments gathered from dir_dict method in utils
    Create and save the plotting information for each experiment in the save path
    
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dir_dict = get_experiment_dicts(root_dir = "../results")

    for i in range(len(dir_dict)):
       noise_type, noise_level, exp_num, init_index = get_exp_details(dir_dict[i])
       
       model_paths = get_model_paths(noise_type, noise_level, exp_num, init_index, points=30)
       
       models = get_models(model_paths, device)
       
       path = make_save_path("./plots/loss_landscapes", model_paths[0].split("/")[-1].split(".")[0])
       
       plot_loss_landscape(models, device, init=init_index, save_path=path, verbose=True)

