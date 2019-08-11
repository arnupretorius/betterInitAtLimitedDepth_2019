import sys, os

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '..'))

from src.utils import load_structured_directory_data

TEST_LIST_PATH = sys.argv[1]
MODEL_STATES_DIRECTORY = sys.argv[2]

def write_paths_to_file(dictionary, keys=[]):
    for key in dictionary:
        if isinstance(dictionary[key], dict):
            write_paths_to_file(dictionary[key], keys + [key])
        elif isinstance(dictionary[key], list):
            for path in dictionary[key]:
                model_path = os.path.abspath(path)
                epoch = int(model_path.split(".")[-2].split("_")[-1])
                [noise_type, noise_level, hyperparam_index], init_index = keys, key

                if int(hyperparam_index) > 30:
                    # write to file (append)
                    with open(TEST_LIST_PATH, "a") as test_list_file:
                        test_list_file.write(
                            "{noise_type} {noise_level} {hyperparam_index} {init_index} {epoch} {model_path}\n".format(
                                noise_type=noise_type, noise_level=noise_level,
                                hyperparam_index=hyperparam_index, init_index=init_index,
                                epoch=epoch, model_path=model_path
                            )
                        )
        else:
            raise ValueError("The dictionary provided to the write_final_epoch_path_to_file function was not in the correct format.")

if __name__ == "__main__":
    temp_save_dir = os.path.join(file_dir, "temp_sensivitiy")
    os.makedirs(temp_save_dir, exist_ok=True)
    progress_file_path = os.path.join(temp_save_dir, "sensitivity_model_paths.npz")

    model_paths = load_structured_directory_data(MODEL_STATES_DIRECTORY, progress_file_name=progress_file_path, force_rescan=True)
    write_paths_to_file(model_paths)