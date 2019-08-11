import sys, os
import numpy as np
import matplotlib.pyplot as plt

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '..'))

from src.utils import load_structured_directory_data
from src.sensitivity_utils import get_data

################################################################################
# load and plot results
################################################################################

def plot_path(data_loader):
    images = data_loader.dataset.data.reshape((-1, *data_loader.dataset.image_shape)).squeeze()

    if len(images.shape) > 3:
        images = images.transpose(0, 2, 3, 1)

    num_axes = 5
    num_images = images.shape[0]
    images_per_axis = num_images // (num_axes-1)

    fig, axes = plt.subplots(1, num_axes, figsize=(25, 5))

    for i, ax in enumerate(axes):
        index = i * images_per_axis
        ax.imshow(images[index])
        ax.set_title(index)

    plt.savefig("path.pdf")

# def plot_full_path(data_loader):
#     dir_name = "./full_path"
#     os.makedirs(dir_name, exist_ok=True)

#     images = data_loader.dataset.data.reshape((-1, *data_loader.dataset.image_shape)).squeeze()

#     for index, image in enumerate(images):
#         plt.figure(figsize=(4, 4))
#         plt.imshow(image)
#         plt.savefig(os.path.join(dir_name, "{}.pdf".format(index)))
#         plt.close()

def plot_test_results(sens, dists, targets, accuracy, accuracies, name="sens_test.pdf"):

    ############################################################################
    # TODO
    # * add a function to plot multiple initialisations on the same axis
    # * add accuracy threshold?
    # * add better titles, axis, etc
    ############################################################################

    plots = {i: dists[:, i] for i in range(10)}

    sens_divisor = np.max(sens)
    if sens_divisor == 0:
        sens_divisor = 1

    colours = plt.get_cmap("tab10")

    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    mean = np.mean(accuracies)
    ax1.hist(accuracies, bins=30, density=True, rwidth=1)
    line_height = ax1.get_ylim()[-1]
    ax1.plot([mean, mean], [0, line_height], label="mean", linestyle="--")
    ax1.plot([accuracy, accuracy], [0, line_height], label="mean of \nselected \nmodels", linestyle="--")

    ax1.set_title("Training accuracy distribution")
    ax1.set_xlabel("Accuracy")
    ax1.set_xlim(0, 1)
    ax1.legend()

    for key in plots:
        if key in range(10):
            new_plot = plots[key]
            ax2.plot(new_plot, label="model {}".format(key), alpha=0.7, color=colours(key))
            # ax2.plot(targets[:, key], label="target {}".format(key), alpha=0.4, color=colours(key))

    ax2.plot(sens / sens_divisor, label="sensitivity \n- {:.2E}".format(sens_divisor), color="red", linestyle="--")

    for key in plots:
        if key in range(10):
            new_plot = plots[key]
            # ax2.plot(new_plot, label=key, alpha=0.7, color=colours(key))
            ax2.plot(targets[:, key], label="target {}".format(key), alpha=0.4, color=colours(key))

    num_interpolations = sens.shape[-1]
    # ax2.set_xlim(0, num_interpolations)
    ax2.set_title("Sensitivity of selected models")
    ax2.set_xlabel("Interpolation space")
    ax2.set_xticks(np.linspace(0, num_interpolations, 11))
    ax2.set_xticklabels([4, 5, 7, 1, 9, 3, 8, 0, 6, 2, 4])
    # ax2.set_title("Accurcy: {}".format(accuracy))
    # ax2.legend()
    ax2.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=2)

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

if __name__ == "__main__":
    # dataset = "cifar10"
    # dataset = "mnist"
    for dataset in ["mnist", "cifar10"]:
        root_dir = os.path.join(file_dir, "../new_plotting/sensitivity", dataset)
        temp_save_dir = os.path.join(file_dir, "temp_sensivitiy")
        os.makedirs(temp_save_dir, exist_ok=True)
        progress_file_path = os.path.join(temp_save_dir, "{}_sensitivity_model_paths.npz".format(dataset))
        paths_to_results = load_structured_directory_data(root_dir, progress_file_name=progress_file_path, force_rescan=True)

        figures_dir = os.path.join(root_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        data = {}
        # data_loader = get_data(dataset, os.path.join(file_dir, "../data"), mode="inter_class")
        data_loader = get_data(dataset, os.path.join(file_dir, "../data"), mode="inter_class", re_generate=True)

        # plot_full_path(data_loader)
        # plot_path(data_loader)

        for noise_type in paths_to_results:

            if noise_type not in data:
                data[noise_type] = {}

            for noise_level in paths_to_results[noise_type]:

                if noise_level not in data[noise_type]:
                    data[noise_type][noise_level] = {}

                for hyperparam_index in paths_to_results[noise_type][noise_level]:

                    if int(hyperparam_index) < 31:
                        continue

                    for init_index in paths_to_results[noise_type][noise_level][hyperparam_index]:

                        if init_index not in data[noise_type][noise_level]:
                            data[noise_type][noise_level][init_index] = {}

                        if hyperparam_index not in data[noise_type][noise_level][init_index]:
                            data[noise_type][noise_level][init_index][hyperparam_index] = {}

                        for path in paths_to_results[noise_type][noise_level][hyperparam_index][init_index]:
                            epoch = path.split(".")[-2].split("_")[-1]
                            test_data = np.load(path)

                            sens = test_data["sensitivity"]
                            dists = test_data["output_distribution"]
                            acc = test_data["train_accuracy"]

                            data[noise_type][noise_level][init_index][hyperparam_index]["sens"] = sens
                            data[noise_type][noise_level][init_index][hyperparam_index]["dists"] = dists
                            data[noise_type][noise_level][init_index][hyperparam_index]["acc"] = acc

                            test_data.close()

                            # ############################################################################
                            # plot_test_results(
                            #     sens, dists, data_loader.dataset.targets, acc, [acc],
                            #     name=os.path.join(
                            #         figures_dir, "{}_{}_{}_{}_{}.pdf".format(
                            #             noise_type, noise_level, hyperparam_index, init_index, epoch
                            #         )
                            #     )
                            # )
                            # ############################################################################

        del paths_to_results

        targets = get_data(dataset, data_path=os.path.join(file_dir, "../data"), mode="inter_class").dataset.targets

        for noise_type in data:
            for noise_level in data[noise_type]:
                for init_index in data[noise_type][noise_level]:
                    print("plotting noise: {}, level: {} and init: {}".format(noise_type, noise_level, init_index))

                    # gather info, average it, plot it
                    num_hyperparam_samplings = len(data[noise_type][noise_level][init_index])

                    sens = np.empty((num_hyperparam_samplings, 1000))
                    dists = np.empty((num_hyperparam_samplings, 1000, 10))
                    acc = np.empty(num_hyperparam_samplings)

                    for i, hyperparam_index in enumerate(data[noise_type][noise_level][init_index]):
                        sens[i] = data[noise_type][noise_level][init_index][hyperparam_index]["sens"]
                        dists[i] = data[noise_type][noise_level][init_index][hyperparam_index]["dists"]
                        acc[i] = data[noise_type][noise_level][init_index][hyperparam_index]["acc"]

                    # sort accuracy in decending order
                    accuracy_order = np.argsort(acc)[::-1]

                    # calculate statistics for the models with the top n accuracies
                    num_models = 100
                    indeces_of_models = accuracy_order[:num_models]

                    avg_sens = np.mean(sens[indeces_of_models], axis=0)
                    avg_dists = np.mean(dists[indeces_of_models], axis=0)
                    avg_acc = np.mean(acc[indeces_of_models])

                    plot_test_results(
                        avg_sens, avg_dists, targets, avg_acc, acc,
                        name=os.path.join(
                            figures_dir, "{}_{}_{}.pdf".format(
                                noise_type, noise_level, init_index
                            )
                        )
                    )
