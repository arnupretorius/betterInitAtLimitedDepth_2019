import torch
import torch.nn as nn
import torch.nn.functional as F

def shifted_relu(x, alpha):
    return torch.clamp(x+alpha, min=0) - alpha

def get_act(act='relu'):
    act_func = None
    if act == 'relu':
        act_func = F.relu
    elif act == 'shifted_relu':
        act_func = shifted_relu
    elif act == 'sigmoid':
        act_func = torch.sigmoid
    return act_func

