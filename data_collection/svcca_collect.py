import shutil, sys, os
import numpy as np

# custom imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.utils import load_model, get_dir_structure, get_dict_structure, load_model, load_data
from src.svcca_utils import get_svcca_correlations_per_layer, dump_svcca_correlations

if __name__ == "__main__":
    root_dir = '../results/mnist'
    # dir_list = get_dir_structure(root_dir)
    # dir_dict_list = get_dict_structure(dir_list)
    dir_dict_list = np.load("{}/temp/test_dict.npz".format(os.path.dirname(os.path.abspath(__file__))))["data"]

    # create data loaders for getting activations
    train_loader, test_loader = load_data('mnist', path='../data')

    # create storage directories
    temp_activations_path = 'temp_activations'
    dumped_correlations_path = '../plotting/svcca'

    # collect svcca correlation data for all experiments on machine
    # get details of the experiment that is to be run
    experiment_details = str(sys.argv[1]).split(" ")
    dict_index = int(experiment_details[0])
    noise_type = experiment_details[1]
    noise_level = experiment_details[2]
    hyperparam_index = experiment_details[3]
    init_index = experiment_details[4]

    print("#######################################################################")
    print("Checking Experiment:")
    print("Noise type: {}".format(noise_type))
    print("Noise level: {}".format(noise_level))
    print("Hyper-parameter index: {}".format(hyperparam_index))
    print("Initialization index: {}".format(init_index))
    print("#######################################################################")


    model_state_paths = load_model(dir_dict_list[dict_index], root_dir)
    model_states = model_state_paths[noise_type][noise_level][hyperparam_index][init_index]

    # dump correlations
    dump_svcca_correlations(model_states, train_loader, n_slices=5,
                            act_path=temp_activations_path, corr_path=dumped_correlations_path)

    shutil.rmtree('temp_activations')

    # # collect svcca correlation data for all experiments on machine
    # for i, dir_dict in enumerate(dir_dict_list):
    #     print('#####################################################')
    #     print("Collecting data for experiment", i+1, 'out of', n_exp)
    #     print('#####################################################')
    #     print('')
    #     keys = dir_list[i].split('/')

    #     print(keys[2:6])

    #     model_state_paths = load_model(dir_dict, root_dir)
    #     model_states = model_state_paths[keys[2]][keys[3]][keys[4]][keys[5]]

    #     print(model_states)
    #     exit()


    #     # dump correlations
    #     dump_svcca_correlations(model_states, train_loader, n_slices=5,
    #                             act_path=temp_activations_path, corr_path=dumped_correlations_path)

    #     shutil.rmtree('temp_activations')
