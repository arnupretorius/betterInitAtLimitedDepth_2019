import torch
import torch.nn as nn
import numpy as np
import copy
from sklearn.decomposition import PCA

# custom imports
from src.utils import test


def get_random_states(states, device):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    return [torch.randn(w.size()).to(device) for k, w in states.items()]


def normalize_directions_for_states(direction, states, ignore='ignore'):
    assert(len(direction) == len(states))
    for d, (k, w) in zip(direction, states.items()):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                d.copy_(w)  # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w)


def normalize_direction(direction, weights):
    direction.div_(direction.norm())


def create_random_direction(net, device, ignore='biasbn'):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
    states = net.state_dict()  # a dict of parameters, including BN's running mean/var.
    direction = get_random_states(states, device)
    normalize_directions_for_states(direction, states, ignore)
    return direction


def set_states(net, states, directions, step):
    """
        Overwrite the network's state_dict or change it along directions with a step size.
    """
    assert directions is not None
    assert step is not None
    dx = directions[0]
    dy = directions[1]
    changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]

    new_states = copy.deepcopy(states)
    assert (len(new_states) == len(changes))
    for (k, v), d in zip(new_states.items(), changes):
        # d = torch.tensor(d)
        d = d.clone().detach()
        v.add_(d.type(v.type()))
    net.load_state_dict(new_states)


def get_diff_states(states1, states2):
    """ Produce a direction from 'states1' to 'states2'."""
    return [v2 - v1 for (k1, v1), (k2, v2) in zip(states1.items(), states2.items())]


def create_target_direction(net1, net2, device, ignore='biasbn'):
    s1 = net1.state_dict()
    s2 = net2.state_dict()
    direction = get_diff_states(s1, s2)
    normalize_directions_for_states(direction, s1, ignore)
    return direction

# random direction, random direction
def loss_landscape_random_2d(model, loader, criterion, device):
    xmin, xmax, xnum = -1, 1, 21
    ymin, ymax, ynum = -1, 1, 21

    state = copy.deepcopy(model.state_dict())

    directions = create_random_direction(model, device), create_random_direction(model, device)

    x = np.linspace(xmin, xmax, xnum)
    y = np.linspace(ymin, ymax, ynum)

    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

    loss = np.empty((xnum, ynum))
    for i in range(xnum):
        for j in range(ynum):
            set_states(model, state, directions, (xv[i, j], yv[i, j]))
            # set_states(model, state, directions, (0.9, 0.1))
            loss[i, j] = test(model, loader, criterion, device)
            print('Evaluating coord=({},{}), loss={}'.format(xv[i, j], yv[i, j], loss[i, j]))

    return xv, yv, loss


# single step, random direction
def loss_landscape_step_2d(model1, model2, loader, criterion, device):
    xmin, xmax, xnum = -1, 1, 21
    ymin, ymax, ynum = -1, 1, 21

    state = copy.deepcopy(model1.state_dict())

    directions = create_target_direction(model1, model2, device), create_random_direction(model1, device)

    x = np.linspace(xmin, xmax, xnum)
    y = np.linspace(ymin, ymax, ynum)

    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

    loss = np.empty((xnum, ynum))
    for i in range(xnum):
        for j in range(ynum):
            set_states(model1, state, directions, (xv[i, j], yv[i, j]))
            # set_states(model, state, directions, (0.9, 0.1))
            loss[i, j] = test(model1, loader, criterion, device)
            print('Evaluating coord=({},{}), loss={}'.format(xv[i, j], yv[i, j], loss[i, j]))

    return xv, yv, loss


# 2 PCA directions of trajectories
def loss_landscape_pca_2d(nets, loader, criterion, device, verbose=True):
    # xmin, xmax, xnum = -1, 1, 21
    # ymin, ymax, ynum = -1, 1, 21

    directions = create_pca_directions(nets, device)
    x_coord, y_coord = project_trajectories(nets, directions)

    deltax = 0.05 * (np.max(x_coord) - np.min(x_coord))
    deltay = 0.05 * (np.max(y_coord) - np.min(y_coord))
    xmin, xmax, xnum = np.min(x_coord) - deltax, np.max(x_coord) + deltax, 11
    ymin, ymax, ynum = np.min(y_coord) - deltay, np.max(y_coord) + deltay, 11

    state = copy.deepcopy(nets[-1].state_dict())

    x = np.linspace(xmin, xmax, xnum)
    y = np.linspace(ymin, ymax, ynum)

    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

    loss = np.empty((xnum, ynum))
    for i in range(xnum):
        for j in range(ynum):
            set_states(nets[-1], state, directions, (xv[i, j], yv[i, j]))
            # set_states(model, state, directions, (0.9, 0.1))
            loss[i, j], _ = test(nets[-1], loader, criterion, device)
            if verbose: print('Evaluating coord = ({}, {}), loss = {}'.format(xv[i, j], yv[i, j], loss[i, j]))

    return xv, yv, loss, directions, x_coord, y_coord


def create_pca_directions(nets, device):

    matrix = list()
    final_state = nets[-1].state_dict()
    for net in nets[:-1]:
        state = net.state_dict()
        d = get_diff_states(final_state, state)
        ignore_biasbn(d)
        # for i in range(len(d)):
        #     d[i] = d[i].cpu()
        d = tensorlist_to_tensor(d)
        matrix.append(d.cpu().numpy())

    # do PCA on matrix
    pca = PCA(n_components=2)
    pca.fit(np.array(matrix))
    pc1 = np.array(pca.components_[0])
    pc2 = np.array(pca.components_[1])

    x_dir = npvec_to_tensorlist(pc1, final_state, device)
    y_dir = npvec_to_tensorlist(pc2, final_state, device)
    ignore_biasbn(x_dir)
    ignore_biasbn(y_dir)
    return x_dir, y_dir

def tensorlist_to_tensor(weights):
    """ Concatnate a list of tensors into one tensor.

        Args:
            weights: a list of parameter tensors, e.g. net_plotter.get_weights(net).

        Returns:
            concatnated 1D tensor
    """
    return torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.cuda.FloatTensor(w) for w in weights])
    # return torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.FloatTensor(w) for w in weights]) # If device is a 'cpu'

def npvec_to_tensorlist(direction, params, device):
    """ Convert a numpy vector to a list of tensors with the same shape as "params".

        Args:
            direction: a list of numpy vectors, e.g., a direction loaded from h5 file.
            base: a list of parameter tensors from net

        Returns:
            a list of tensors with the same shape as base
    """
    # device = torch.device("cpu")
    s2 = []
    idx = 0
    for (k, w) in params.items():
        s2.append(torch.Tensor(direction[idx:idx + w.numel()]).view(w.size()).to(device))
        idx += w.numel()
    assert(idx == len(direction))
    return s2


def ignore_biasbn(directions):
    """ Set bias and bn parameters in directions to zero """
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)


def project_trajectories(nets, directions):
    final_state = nets[-1].state_dict()

    # for i in range(len(directions[0])):
    #     directions[0][i] = directions[0][i].cpu()
    # for i in range(len(directions[1])):
    #     directions[1][i] = directions[1][i].cpu()

    dx, dy = tensorlist_to_tensor(directions[0]), tensorlist_to_tensor(directions[1])
    x_coord = list()
    y_coord = list()
    for net in nets[:-1]:
        state = net.state_dict()
        d = get_diff_states(final_state, state)
        # ignore_biasbn(d)
        # for i in range(len(d)):
        #     d[i] = d[i].cpu()
        d = tensorlist_to_tensor(d)

        x, y = project_2d(d, dx, dy)
        x_coord.append(x)
        y_coord.append(y)
    return x_coord, y_coord

def project_2d(d, dx, dy):

    return project_1d(d, dx), project_1d(d, dy) # because dx and dy are orthogonal

def project_1d(w, d):
    assert len(w) == len(d)
    scale = torch.dot(w, d) / d.norm()
    return scale.item()

def get_hp(path, index):
    return int(path.split('/')[-1].split(".")[0].split("_")[index])

def get_exp_details(dir_dict):
    for nt in dir_dict:
        noise_type = nt
        # print(noise_type)
        for nl in dir_dict[nt]:
            noise_level = nl
            # print(noise_level)
            for exp in dir_dict[nt][nl]:
                exp_num = exp
                # print(exp_num)
                for init in dir_dict[nt][nl][exp]:
                    init_index = init
                    # print(init)

    return noise_type, noise_level, exp_num, init_index