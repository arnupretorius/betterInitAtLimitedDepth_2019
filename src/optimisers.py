import torch
import torch.optim as optim


def get_optimiser(net_params, name="SGD", learning_rate=0.01, momentum=0.0):
    optimiser = None
    if name not in ["SGD", "Adam", "RMSprop"]:
        raise ValueError("Optimiser not supported.")
    if name == "SGD":
        optimiser = optim.SGD(net_params, lr=learning_rate, momentum=momentum)
    elif name == "Adam":
        optimiser = optim.Adam(net_params, lr=learning_rate)
    elif name == "RMSprop":
        optimiser = optim.RMSprop(net_params, lr=learning_rate, momentum=momentum)
    return optimiser
