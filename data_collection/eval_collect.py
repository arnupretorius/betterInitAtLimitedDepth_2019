import os
import sys

import torch.nn as nn
import numpy as np

# custom import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.data_loader import load_data
from src.utils import get_experiment_dicts, test_models, load_model_results


# Load MNIST data
train_loader, test_loader = load_data('mnist', path='../data')

# Load models
result_path = '../results'
model_dicts = get_experiment_dicts(result_path)

# Test models
dump_path='../plotting'
criterion = nn.CrossEntropyLoss()
test_models(model_dicts, criterion, train_loader, test_loader, result_path=result_path, dump_path=dump_path, quiet=True)
