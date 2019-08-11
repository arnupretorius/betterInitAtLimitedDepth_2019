import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.utils import load_model, get_experiment_dicts

TEST_LIST_DIRECTORY = "{}/temp/".format(os.path.dirname(os.path.abspath(__file__)))
TEST_LIST_PATH = "{}/compress_list.txt".format(TEST_LIST_DIRECTORY)

def write_path_to_file(dictionary, keys=[]):
    for key in dictionary:
        if isinstance(dictionary[key], dict):
            write_path_to_file(dictionary[key], keys + [key])
        elif isinstance(dictionary[key], list):
            model_paths = "\n".join(list(map(os.path.abspath, dictionary[key][:-1])))

            # write to file (append)
            with open(TEST_LIST_PATH, "a") as test_list_file:
                test_list_file.write(
                    "{model_path}\n".format(model_path=model_paths)
                )
        else:
            raise ValueError("The dictionary provided to the write_path_to_file function was not in the correct format.")

if __name__ == "__main__":
    root_dir = '../results'
    experiment_dicts = get_experiment_dicts(root_dir)

    paths_per_experiment_dict = []
    for experiment_dict in experiment_dicts:
        model_paths = load_model(experiment_dict, path_to_results=root_dir)
        paths_per_experiment_dict.append(model_paths)

    os.makedirs(TEST_LIST_DIRECTORY, exist_ok=True)
    open(TEST_LIST_PATH, "w").close()

    for dictionary in paths_per_experiment_dict:
        write_path_to_file(dictionary)
