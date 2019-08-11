import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def gauss(x, stddev=1, is_training=True):
    if is_training:
         noise = Variable(x.data.new(x.size()).normal_(1, stddev))
         return x*noise
    return x

def dropout(x, p=0.5, is_training=True):
    if is_training:
         noise = Variable(x.data.new(x.size()).bernoulli_(p))/p
         return x*noise
    return x

def get_noise_layer(noise_type='dropout'):
    
    noise_layer = None
    if noise_type == 'dropout':
        noise_layer = dropout
    elif noise_type == 'gauss':
        noise_layer = gauss
    
    return noise_layer