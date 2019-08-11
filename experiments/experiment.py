import os, sys, torch
import numpy as np

# custom imports
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, '..'))

from src.utils import get_hyperparameters, get_settings, get_experiment, get_number_of_experiments, save_model, get_noise
from src.utils import train, recreate_model
from src.data_loader import load_data, get_data_dimensions
from src.optimisers import get_optimiser
from src.net import Net
from src.utils import model_name, get_train_and_start_epoch

if __name__ == "__main__":
    device, dtype, _ = get_settings()
    hyperparam_indices = [int(i) for i in sys.argv[1].split("_")]
    experiment_indices = [int(i) for i in sys.argv[2].split("_")]
    act = "relu"
    dataset = sys.argv[3]
    epochs = int(sys.argv[4])
    model_states_dir = sys.argv[5]
    plotting_dir = sys.argv[6]
    experiment_name = model_name(hyperparam_indices, experiment_indices)

    # # hyperparameters
    noise_type, noise_level = get_noise(experiment_indices)
    hyperparams = get_hyperparameters(hyperparam_indices, noise_type, noise_level)
    batch_size = hyperparams[1]
    n_hidden = hyperparams[3]
    n_layer = hyperparams[2]
    seed = hyperparams[4]
    learning_rate = hyperparams[5]
    momentum = hyperparams[6]
    op = hyperparams[7]

    # experiment
    _, _, init_val = get_experiment(experiment_indices, n_layer)

    if noise_type == 'none':
        noise_type = None
        noise_level = None

    print("=====================================================================")
    print("NEXT EXPERIMENT:")
    print("     Noise type:", noise_type)
    print("     Noise level:", noise_level)
    print("     Initialisation:", init_val)
    print("     Activation:", act)
    print("     Dataset:", dataset)
    print("     Epochs:", epochs)
    print("     Batch size:", batch_size)
    print("     Width:", n_hidden)
    print("     Depth:", n_layer)
    print("     Optimiser:", op)
    print("     Learning rate:", learning_rate)
    print("     Momentum:", momentum)
    print("=====================================================================")

    model_dir = os.path.join(model_states_dir, dataset)

    noise_level_str = "0" if noise_level is None else str(noise_level)
    noise_type_str = "none" if noise_type is None else noise_type
    hyperparam_index = str(hyperparam_indices[0])
    init_index = str(experiment_indices[-1])

    experiment_results_directory = os.path.abspath(os.path.join(
        plotting_dir, dataset, noise_type_str, noise_level_str, hyperparam_index, init_index
    ))
    os.makedirs(experiment_results_directory, exist_ok=True)

    model_states_dir = os.path.join(model_dir, noise_type_str, noise_level_str, hyperparam_index)
    run_train, start_epoch, model_to_load = get_train_and_start_epoch(experiment_results_directory, model_states_dir, epochs, experiment_name)

    if run_train:
        # Load data
        train_loader, test_loader = load_data(dataset, batch_size=batch_size, path=os.path.join(base_path, '../data'))

        if start_epoch == 0:
            print("results directory is empty, starting test from beginning")

            # create network
            torch.manual_seed(seed)
            n_in, n_out = get_data_dimensions(dataset)
            net = Net(n_in, n_hidden, n_out, n_layer, act=act, noise_type=noise_type,
                    noise_level=noise_level, init_val=init_val).to(device, dtype)

            save_model(net, experiment_name, 0, noise_type, noise_level, model_dir=model_dir)
        else:
            print("starting from epoch {}".format(start_epoch))
            net = recreate_model(model_to_load, dataset=dataset, act=act)

        # optimiser parameters
        optimiser = get_optimiser(net.parameters(), op, learning_rate, momentum)

        # training criterion
        criterion = torch.nn.CrossEntropyLoss()

        # train network
        train(net, train_loader, test_loader, criterion, optimiser, epochs, noise_type, noise_level, save=True, name=experiment_name, model_dir=model_dir, results_dir=experiment_results_directory, start_epoch=start_epoch)

    else:
        print("results are already present, skipping test.")
