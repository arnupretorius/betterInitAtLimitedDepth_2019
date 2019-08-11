import os, sys
import numpy as np
import matplotlib.pyplot as plt

file_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(file_dir, ".."))

from src.utils import load_structured_directory_data

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

crit_init = 4
def plot_hypoth_data(gen_data, speed_data, name):
    print(gen_data.shape)
    plt.figure(figsize=(6, 4))

    for init in [3, 5, 2, 6, 1, 7, 0, 8, 9, 10, 4]:
    # for init in np.arange(11):
        if init < crit_init:
            colour = "blue"
            alpha = (init + 1) / 6
        elif 9 > init > crit_init:
            colour = "orange"
            alpha = (-(init - 8) + 1) / 6
        elif init >= 9:
            colour = "red"
            alpha = (-(init - 10) + 1) / 4
        else:
            colour = "green"
            alpha = 0.5

        plt.scatter(gen_data[:, :, :, init].reshape((-1,)), speed_data[:, :, :, init].reshape((-1,)), color=colour, alpha=alpha)

    plt.xlabel("Max Test Accuracy")
    plt.ylabel("Epochs to Reach 0.3 Accuracy")
    plt.savefig("{}.pdf".format(name))
    plt.close()

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

    dynamics_data = load_structured_directory_data(dynamics_path, progress_file_name=progress_file_path, data_file_name=data_file_path, load_data=True, force_rescan=False)

    ############################################################################
    hyper_param_white_list = np.arange(31, 103)
    ############################################################################

    acc_threshold = 0.3

    # average data over hyper-parameters
    d_map = {"mnist": 0, "cifar10": 1, "fashion_mnist": 2, "cifar100": 3}
    n_map = {"none 0": 0, "dropout 0.9": 1, "dropout 0.7": 2, "dropout 0.5": 3}
    num_datasets = len(d_map)
    num_noises = len(n_map)
    shape = (num_datasets, num_noises, 72, 11)
    generalisation_data = np.full(shape, np.float("NaN"))
    speed_data = np.full(shape, np.float("NaN"))

    for dataset in dynamics_data:
        print("dataset", dataset)
        dataset_data = dynamics_data[dataset]
        for noise_type in dataset_data:
            noise_type_data = dataset_data[noise_type]
            for noise_level in noise_type_data:
                noise_level_data = noise_type_data[noise_level]
                for hyp_index in noise_level_data:
                    if int(hyp_index) not in hyper_param_white_list:
                        continue

                    hyp_index_data = noise_level_data[hyp_index]
                    for init_index in hyp_index_data:
                        init_index_data = hyp_index_data[init_index]
                        for data_type in init_index_data:
                            data_type_data = init_index_data[data_type]
                            for file_name in data_type_data:
                                if "accuracy" not in file_name:
                                    continue

                                data = fix_data(data_type_data[file_name])

                                if len(data) > 0:
                                    above_threshold = data >= acc_threshold
                                    first_above = np.argmax(above_threshold)
                                    if data[first_above] < acc_threshold:
                                        first_above = np.float("NaN")

                                    data_index = d_map[dataset]
                                    noise_index = n_map["{} {}".format(noise_type, noise_level)]

                                    if "test" in file_name:
                                        generalisation_data[data_index][noise_index][int(hyp_index) - 31][int(init_index)] = np.max(data)
                                    elif "train" in file_name:
                                        speed_data[data_index][noise_index][int(hyp_index) - 31][int(init_index)] = first_above

    save_path = os.path.join(root_save_dir, "hypoth_data")
    plot_hypoth_data(generalisation_data, speed_data, save_path)

    save_path = os.path.join(root_save_dir, "mnist")
    plot_hypoth_data(generalisation_data[0][np.newaxis, :], speed_data[0][np.newaxis, :], save_path)

    save_path = os.path.join(root_save_dir, "cifar10")
    plot_hypoth_data(generalisation_data[1][np.newaxis, :], speed_data[1][np.newaxis, :], save_path)

    save_path = os.path.join(root_save_dir, "fashion_mnist")
    plot_hypoth_data(generalisation_data[2][np.newaxis, :], speed_data[2][np.newaxis, :], save_path)

    save_path = os.path.join(root_save_dir, "cifar100")
    plot_hypoth_data(generalisation_data[3][np.newaxis, :], speed_data[3][np.newaxis, :], save_path)