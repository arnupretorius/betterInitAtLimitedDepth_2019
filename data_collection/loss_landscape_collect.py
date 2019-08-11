import os
import sys
import torch
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# custom imports
from src.utils import get_experiment_dicts, make_save_path
from src.loss_landscape_utils import get_exp_details
from src.loss_landscape_visual import get_models, get_model_paths, plot_loss_landscape, plot_loss_landscape_from_file

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

root_dir = "../results"

# Get all experiments in dictionary form
dir_dict = get_experiment_dicts(root_dir = root_dir)

start_time = time.time()

# For each experiment, create a .npz file in "plotting" folder under similar hierarchy for later plotting
for i in range(len(dir_dict)):
	noise_type, noise_level, exp_num, init_index = get_exp_details(dir_dict[i])
       
	model_paths = get_model_paths(noise_type, noise_level, exp_num, init_index, points=50, root_dir = root_dir)

	print("\nModels paths:")
	for model in model_paths:
		print(model)
	print()

	models = get_models(model_paths, device)

	path = make_save_path("../plotting/loss_landscapes", model_paths[0].split("/")[-1].split(".")[0])

	plot_loss_landscape(models, device, init=init_index, save_path=path, verbose=True)

# Print time taken for experiments to run
t = time.time() - start_time
print("\nTime taken: {0:.0f} hours {1:.0f} minutes {2:.0f} seconds".format(t // 3600 % 24, t // 60 % 60, t % 60))

# Test for visualising a graph from .npz file
# plot_loss_landscape_from_file("../plotting/loss_landscapes/none/0/16/init_6.npz")