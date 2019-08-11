import os, sys
import numpy as np

file_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(file_dir, ".."))

from src.utils import load_structured_directory_data

def save_csv(data, name):
    init_string = ",init_".join(map(str, range(data.shape[-1]-1)))
    column_headings = "hyper_param_index,init_{}".format(init_string)
    np.savetxt("{}.csv".format(name), data, delimiter=",", header=column_headings)

def fix_data(data):
    try:
        if len(data) == 0:
            return np.array([np.float("NaN")])
    except TypeError:
        return np.array([data])

    length = len(data)
    full_acc_index = np.argmax(data == 1.0)
    full_acc_achieved = data[full_acc_index] == 1.0

    if length > 501 or (full_acc_achieved and full_acc_index < length - 1):
        new_data = []
        previous_values = set()
        for index, value in enumerate(data):
            if value not in previous_values:
                previous_values.add(value)
                new_data.append(value)

            if value == 1.0:
                # print("got to 100% acc")
                break

            if len(new_data) > 500:
                # print("max len reached")
                break

        return np.array(new_data)
    else:
        return data

if __name__ == "__main__":
    dynamics_path = os.path.abspath(os.path.join(file_dir, "../new_plotting/training"))
    root_save_dir = os.path.join(file_dir, "../new_plotting/hypothesis_testing")
    temp_dir = os.path.join(file_dir, "temp")

    os.makedirs(root_save_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    progress_file_name = "temp_dynamics_paths.npz"
    data_file_name = "temp_dynamics_data.npz"
    progress_file_path = os.path.join(temp_dir, progress_file_name)
    data_file_path = os.path.join(temp_dir, data_file_name)

    dynamics_data = load_structured_directory_data(dynamics_path, progress_file_name=progress_file_path, data_file_name=data_file_path, load_data=True, force_rescan=True)

    ############################################################################
    hyper_param_white_list = np.arange(31, 103)
    # hyper_param_white_list = [20, 16, 26, 8, 3, 22]
    ############################################################################

    acc_thresholds = [0.3,]
    # acc_thresholds = np.linspace(0.1, 1.0, 10)

    # average data over hyper-parameters
    for dataset in dynamics_data:
        dataset_data = dynamics_data[dataset]
        for noise_type in dataset_data:
            noise_type_data = dataset_data[noise_type]
            for noise_level in noise_type_data:
                if float(noise_level) < 0.5:
                    continue

                noise_level_data = noise_type_data[noise_level]

                for acc_threshold in acc_thresholds:
                    hypoth_data = {}
                    for hyp_index in noise_level_data:
                        hyp_index_data = noise_level_data[hyp_index]

                        hypoth_data[hyp_index] = {}
                        for init_index in hyp_index_data:
                            if init_index not in hypoth_data[hyp_index]:
                                hypoth_data[hyp_index][init_index] = {}

                            init_index_data = hyp_index_data[init_index]
                            for data_type in init_index_data:
                                data_type_data = init_index_data[data_type]
                                for file_name in data_type_data:
                                    data = fix_data(data_type_data[file_name])

                                    if len(data) < 1:
                                        first_above = -2
                                    else:
                                        above_threshold = data >= acc_threshold
                                        first_above = np.argmax(above_threshold)
                                        if data[first_above] < acc_threshold:
                                            first_above = -1

                                        if "accuracy" in file_name and "test" in file_name:
                                            hypoth_data[hyp_index][init_index]["test"] = {
                                                "max": np.max(data)
                                            }
                                        elif "accuracy" in file_name and "train" in file_name:
                                            hypoth_data[hyp_index][init_index]["train"] = {
                                                "epoch": first_above
                                            }

                    # reshape hypoth_data into a 2D array
                    num_hyp = len(hypoth_data)
                    num_init = 0
                    for (key, value) in hypoth_data.items():
                        num_init = max(num_init, len(value))

                    # full_data_shape = (num_hyp, num_init + 1)
                    # full_generalisation_data = np.zeros(full_data_shape) - 2
                    # full_training_speed_data = np.zeros(full_data_shape) - 2

                    white_list_data_shape = (len(hyper_param_white_list), num_init + 1)
                    white_list_generalisation_data = np.zeros(white_list_data_shape) - 2
                    white_list_training_speed_data = np.zeros(white_list_data_shape) - 2

                    # full_hyp_counter = 0
                    white_list_hyp_counter = 0
                    for hyp_index_str in hypoth_data:
                        hyp_index = int(hyp_index_str)
                        hyp_data = hypoth_data[hyp_index_str]

                        if hyp_index in hyper_param_white_list:
                            white_list_generalisation_data[white_list_hyp_counter][0] = hyp_index
                            white_list_training_speed_data[white_list_hyp_counter][0] = hyp_index

                        # full_generalisation_data[full_hyp_counter][0] = hyp_index
                        # full_training_speed_data[full_hyp_counter][0] = hyp_index

                        for init_index_str in hyp_data:
                            init_index = int(init_index_str)
                            init_data = hyp_data[init_index_str]

                            # full_generalisation_data[full_hyp_counter][init_index+1] = init_data["max"]
                            # full_training_speed_data[full_hyp_counter][init_index+1] = init_data["epoch"]

                            if hyp_index in hyper_param_white_list:
                                try:
                                    white_list_generalisation_data[white_list_hyp_counter][init_index+1] = init_data["test"]["max"]
                                    white_list_training_speed_data[white_list_hyp_counter][init_index+1] = init_data["train"]["epoch"]
                                except KeyError:
                                    print("missing...")
                                    print("data", dataset)
                                    print("noise", noise_level)
                                    print("hyper_param", hyp_index)
                                    print("init", init_index)
                                    # raise

                        # full_hyp_counter += 1

                        if hyp_index in hyper_param_white_list:
                            white_list_hyp_counter += 1

                    acc_threshold = "{:.1f}".format(acc_threshold)
                    print("saving for {} {} {} acc threshold {}".format(dataset, noise_type, noise_level, acc_threshold))
                    base_path = os.path.join(root_save_dir, dataset, noise_type, noise_level, acc_threshold)
                    os.makedirs(base_path, exist_ok=True)

                    # # save data for all hyper_params
                    # generalisation_file_name = "{}_{}_{}_{}_full_generalisation".format(acc_threshold, dataset, noise_type, noise_level)
                    # training_speed_file_name = "{}_{}_{}_{}_full_training_speed".format(acc_threshold, dataset, noise_type, noise_level)

                    # save_csv(full_generalisation_data, os.path.join(base_path, generalisation_file_name))
                    # save_csv(full_training_speed_data, os.path.join(base_path, training_speed_file_name))

                    # save data for selected hyper_params
                    training_speed_file_name = "{}_{}_{}_{}_white_list_training_speed".format(acc_threshold, dataset, noise_type, noise_level)

                    save_csv(white_list_training_speed_data, os.path.join(base_path, training_speed_file_name))

                base_path = os.path.join(root_save_dir, dataset, noise_type, noise_level)
                generalisation_file_name = "{}_{}_{}_white_list_generalisation".format(dataset, noise_type, noise_level)
                save_csv(white_list_generalisation_data, os.path.join(base_path, generalisation_file_name))
