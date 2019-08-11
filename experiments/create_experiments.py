import os
import sys
import torch

# custom imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.utils import create_exp_file


create_exp_file(split=63)