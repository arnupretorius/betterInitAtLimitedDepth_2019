import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.utils import load_model, get_experiment_dicts

TEST_LIST_DIRECTORY = "{}/temp/".format(os.path.dirname(os.path.abspath(__file__)))
TEST_LIST_PATH = "{}/test_list.txt".format(TEST_LIST_DIRECTORY)
TEST_DICT_PATH = "{}/test_dict.npz".format(TEST_LIST_DIRECTORY)

def write_details_to_file(dictionary, dict_index, keys=[]):
    for key in dictionary:
        if isinstance(dictionary[key], dict):
            write_details_to_file(dictionary[key], dict_index, keys + [key])
        elif isinstance(dictionary[key], list):
            final_epoch_model_path = os.path.abspath(dictionary[key][-1])

            # write to file (append)
            with open(TEST_LIST_PATH, "a") as test_list_file:
                [noise_type, noise_level, hyperparam_index], init_index = keys, key
                test_list_file.write(
                    "{dict_index} {noise_type} {noise_level} {hyperparam_index} {init_index}\n".format(
                        dict_index=dict_index, noise_type=noise_type, noise_level=noise_level,
                        hyperparam_index=hyperparam_index, init_index=init_index
                    )
                )
        else:
            raise ValueError("The dictionary provided to the write_final_epoch_path_to_file function was not in the correct format.")

if __name__ == "__main__":
    root_dir = '../results/mnist'
    experiment_dicts = get_experiment_dicts(root_dir)

    paths_per_experiment_dict = []
    for experiment_dict in experiment_dicts:
        model_paths = load_model(experiment_dict, path_to_results=root_dir)
        paths_per_experiment_dict.append(model_paths)

    os.makedirs(TEST_LIST_DIRECTORY, exist_ok=True)
    open(TEST_LIST_PATH, "w").close()
    np.savez_compressed(TEST_DICT_PATH, data=paths_per_experiment_dict)

    for index, dictionary in enumerate(paths_per_experiment_dict):
        write_details_to_file(dictionary, dict_index=index)