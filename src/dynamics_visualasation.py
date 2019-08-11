import os, sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool, Process

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.utils import load_structured_directory_data

def pad_to(array, final_length):
    padding_left_right = (0, final_length - len(array))
    return np.pad(array, padding_left_right, mode="edge")

def pad_const_to(array, final_length, value=np.float('NaN')):
    padding_left_right = (0, final_length - len(array))
    return np.pad(array, padding_left_right, mode="constant", constant_values=value)

def exp_smooth(array):
    new_array = np.empty(array.shape)
    new_array[0] = array[0]

    for i in np.arange(1, array.shape[0]):
        new_array[i] = 0.5 * array[i] + 0.5 * new_array[i-1]

    return new_array

def plot_trajectories(dataset, noise_type, noise_level, data, save_dir, hyp_index=None):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    # fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    crit_init = 4

    for init_str in data:
        init = int(init_str)


        if init < crit_init:
            colour = "blue"
            alpha = (init + 1) / 5
        elif 9 > init > crit_init:
            colour = "orange"
            alpha = (-(init - 8) + 1) / 5
        elif init >= 9:
            colour = "red"
            alpha = (-(init - 10) + 1) / 3
        else:
            colour = "green"
            alpha = 1.0

        # if init == 0:
        #     alpha = 1
        #     colour = "green"
        # elif init % 2 == 0:
        #     if init == 2:
        #         alpha = 1
        #     elif init == 6:
        #         alpha = 0.75
        #     elif init == 8:
        #         alpha = 0.5
        #     else:
        #         alpha = 0.25

        #     colour = "blue"
        # else:
        #     if init == 1:
        #         alpha = 1
        #     elif init == 5:
        #         alpha = 0.75
        #     elif init == 7:
        #         alpha = 0.5
        #     else:
        #         alpha = 0.25

        #     colour = "red"

        ax1.plot(data[init_str]["train"]["accuracy"], color=colour, alpha=alpha)
        # ax2.plot(data[init_str]["test"]["accuracy"], color=colour, alpha=alpha)

    ax1.set_ylabel("Train Accuracy")
    # ax1.set_title("Train Accuracy")
    # ax2.set_ylabel("Test Accuracy")
    # ax2.set_title("Test Accuracy")
    ax1.set_xlabel("Epoch")
    # ax2.set_xlabel("Epoch")
    # ax1.legend()


    if hyp_index:
        title = "Training dynamics for hyper-param {} {} of {} trained on {}".format(hyp_index, noise_type, noise_level, dataset)
    else:
        title = "Training dynamics for {} of {} trained on {}".format(noise_type, noise_level, dataset)
    # title = "Training dynamics for {}\nof {} trained on {}".format(noise_type, noise_level, dataset)
    # fig.suptitle(title, fontsize=12)
    plt.title("Training trajectory")

    plt.tight_layout()
    if hyp_index:
        file_name = "{}_{}_{}_{}.pdf".format(dataset, noise_type, noise_level, hyp_index)
    else:
        file_name = "{}_{}_{}.pdf".format(dataset, noise_type, noise_level)

    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    plt.close()

full_acc_count = 0
too_long_count = 0
fix_counter = 0
fine_counter = 0

def fix_data(data):
    global too_long_count, fix_counter, full_acc_count, fine_counter

    try:
        if len(data) == 0:
            return np.array([np.float("NaN")])
    except TypeError:
        return np.array([data])

    length = len(data)
    full_acc_index = np.argmax(data == 1.0)
    full_acc_achieved = data[full_acc_index] == 1.0

    if length > 501 or (full_acc_achieved and full_acc_index < length - 1):
        fix_counter += 1

        # data = np.sort(data)

        new_data = []
        previous_values = set()
        for index, value in enumerate(data):
            if value not in previous_values:
                previous_values.add(value)
                new_data.append(value)

            if value == 1.0:
                # print("got to 100% acc")
                full_acc_count += 1
                break

            if len(new_data) > 500:
                # print("max len reached")
                too_long_count += 1
                break

        return np.array(new_data)
    else:
        fine_counter += 1
        return data

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print("Scanning system for data...")
    dynamics_path = os.path.abspath(os.path.join(current_dir, "../new_plotting/training"))
    # dynamics_path = os.path.abspath(os.path.join(current_dir, "../plotting/training"))
    dynamics_data = load_structured_directory_data(dynamics_path, progress_file_name="temp_dynamics_paths.npz", data_file_name="temp_dynamics_data.npz", load_data=True, force_rescan=False)

    # not_fine_counter = 0
    # fix_counter = 0

    # average data over hyper-parameters
    for dataset in dynamics_data:
        dataset_data = dynamics_data[dataset]
        for noise_type in dataset_data:
            noise_type_data = dataset_data[noise_type]
            for noise_level in noise_type_data:
                noise_level_data = noise_type_data[noise_level]

                # insert things according to hyper-param so that I can average over that
                data_ordered_by_init = {}

                for hyp_index in noise_level_data:

                    if 30 < int(hyp_index) <= 102:

                        hyp_index_data = noise_level_data[hyp_index]
                        for init_index in hyp_index_data:
                            if init_index not in data_ordered_by_init:
                                data_ordered_by_init[init_index] = {
                                    "train": {"accuracy": []}
                                }
                                # data_ordered_by_init[init_index] = {
                                #     "train": {"loss": [], "accuracy": []},
                                #     "test": {"loss": [], "accuracy": []}
                                # }

                            init_index_data = hyp_index_data[init_index]
                            for data_type in init_index_data:
                                if data_type != "train":
                                    continue

                                data_type_data = init_index_data[data_type]
                                for file_name in data_type_data:
                                    data = data_type_data[file_name]

                                    for metric in ["accuracy"]:
                                    # for metric in ["loss", "accuracy"]:
                                        if metric in file_name:
                                            data_ordered_by_init[init_index][data_type][metric].append(fix_data(data))
                                            # try:
                                            #     if len(data) > 501:
                                            #         # print(data.shape)
                                            #         # print(file_name)
                                            #         data = fix_data(data)
                                            #         if len(data) > 501:
                                            #             print(data.shape)
                                            #             print(file_name)
                                            #             not_fine_counter += 1
                                            #         else:
                                            #             data_ordered_by_init[init_index][data_type][metric].append(data)
                                            #             fix_counter += 1
                                            #     elif len(data) == 0:
                                            #         print(data.shape)
                                            #         print(file_name)
                                            #         not_fine_counter += 1
                                            #     else:
                                            #         data_ordered_by_init[init_index][data_type][metric].append(data)
                                            #         fine_counter += 1
                                            # except TypeError:
                                            #     print("TypeError: len() of unsized object")
                                            #     print(data)
                                            #     not_fine_counter += 1



                        # hyp_plot_data = {
                        #     _init_index: {
                        #         "train": {
                        #             "loss": pad_to(data_ordered_by_init[_init_index]["train"]["loss"][-1], 501),
                        #             "accuracy":  pad_to(data_ordered_by_init[_init_index]["train"]["accuracy"][-1], 501)
                        #         },
                        #         "test": {
                        #             "loss": pad_to(data_ordered_by_init[_init_index]["test"]["loss"][-1], 501),
                        #             "accuracy": pad_to(data_ordered_by_init[_init_index]["test"]["accuracy"][-1], 501)
                        #         }
                        #     }
                        #     for _init_index in hyp_index_data.keys()
                        # }
                        # hyp_dir = os.path.join(dynamics_path, dataset, noise_type, noise_level, hyp_index)
                        # plot_trajectories(dataset, noise_type, noise_level, hyp_plot_data, hyp_dir, hyp_index)

                # average over data
                for init_index in data_ordered_by_init:
                    init_data = data_ordered_by_init[init_index]
                    for test_or_train in init_data:
                        test_or_train_data = init_data[test_or_train]
                        for loss_or_acc in test_or_train_data:
                            trajectory_per_hyper_param = test_or_train_data[loss_or_acc]

                            # find the length we need to pad arrays to
                            with Pool() as p:
                                lengths = p.map(len, trajectory_per_hyper_param)

                            # pad arrays to this length
                            max_len = np.max(lengths)
                            with Pool() as p:
                                padded_trajectories = np.array(p.map(partial(pad_to, final_length=max_len), trajectory_per_hyper_param))
                                # arguments = [
                                #     {"array": array, "final_length": max_len}
                                #     for array in trajectory_per_hyper_param
                                # ]
                                # padded_trajectories = np.array(p.map(pad_to, arguments))

                            # calculate mean for each column (epoch)
                            avg_trajectory = np.nanmean(padded_trajectories, axis=0)
                            avg_trajectory = exp_smooth(avg_trajectory)
                            test_or_train_data[loss_or_acc] = avg_trajectory

                noise_type_data[noise_level] = data_ordered_by_init
                print("plotting {} {} {}".format(dataset, noise_type, noise_level))
                plot_trajectories(dataset, noise_type, noise_level, data_ordered_by_init, dynamics_path)

    print("100% acc: ", full_acc_count)
    print("too long: ", too_long_count)
    print("fine: ", fine_counter)
    print("fixed: ", fix_counter)
    # print("NOT fine: ", not_fine_counter)