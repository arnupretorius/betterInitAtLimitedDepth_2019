import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# custom imports
from src.acts import get_act
from src.noise_layers import get_noise_layer

class Net(nn.Module):

    def __init__(self, n_in, n_hidden, n_out, n_layer, act='relu',
                 noise_type=None, noise_level=None, init_val=1):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()

        # input_layer
        self.layers.append(nn.Linear(n_in, n_hidden))

        for _ in torch.arange(n_layer):
            self.layers.append(nn.Linear(n_hidden, n_hidden))

        # output_layer
        self.out = nn.Linear(n_hidden, n_out)

        self.n_layer = n_layer
        self.act = get_act(act)

        # noise layers
        if noise_type != None:
            self.noise_layer = get_noise_layer(noise_type)
            self.noise_level = noise_level
        else:
            self.noise_layer = None

        # initialise layers
        torch.nn.init.normal_(self.layers[0].weight, mean=0, std=np.sqrt(init_val / n_in))
        torch.nn.init.constant_(self.layers[0].bias, 0)

        for layer in self.layers[1:]:
            torch.nn.init.normal_(layer.weight, mean=0, std=np.sqrt(init_val / n_hidden))
            torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.normal_(self.out.weight, mean=0, std=np.sqrt(1 / n_hidden))
        torch.nn.init.constant_(self.out.bias, 0)

    def forward(self, x):

        # flatten input
        x = x.view(x.size(0), -1)

        if self.noise_layer != None:
            for layer in self.layers:
                x = self.act(layer(self.noise_layer(x, self.noise_level, self.training)))

        else:
            for layer in self.layers:
                x = self.act(layer(x))

        # output layer
        x = self.out(x)
        return x

    def predict(self, x):
        self.eval()

        # flatten input
        x = x.view(x.size(0), -1)

        if self.noise_layer != None:
            for layer in self.layers:
                x = self.act(layer(self.noise_layer(x, self.noise_level, self.training)))

        else:
            for layer in self.layers:
                x = self.act(layer(x))

        # output layer
        x = self.out(x)
        return F.softmax(x, dim=-1)
