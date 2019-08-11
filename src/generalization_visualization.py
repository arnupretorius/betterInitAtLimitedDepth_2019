import os, sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.utils import load_model_results, get_experiment_dicts, load_structured_directory_data, variance_prop_depth, get_initialisations, hyperparam_indices_from_index, get_hyperparameters

accuracy_thresholds = np.linspace(0.1, 1.0, 10)
# accuracy_thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
init_map = np.array([4, 5, 3, 8, 0, 6, 2, 7, 1])
init_index_order = np.array([4, 8, 6, 2, 0, 1, 5, 7, 3])

def map_index_to_plot_position(index):
    return init_map[int(index)]

def plot_gen_gap_vs_init(data, avg_final_acc, avg_final_losses, averaging_mode="all", name="gen_gap_vs_init.pdf"):
    ############################################################################
    # change this plot so that the actual sigma_w values are used on the x axis?
    ############################################################################

    first = True
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

    box_plot_data = {"loss": [], "acc": [], "labels": []}
    for init_index in data:
        if averaging_mode == "per_acc":
            ####################################################################
            ####################################################################
            raise NotImplementedError("'per_acc' averaging has not been implemented yet.")
        else:
            loss_gaps = []
            accuracy_gaps = []
            for hyperparam_index in data[init_index]:
                current_test = data[init_index][hyperparam_index]
                loss_gaps += list(current_test["loss_gap"])
                accuracy_gaps += list(current_test["accuracy_gap"])

            avg_loss_gap = np.nanmean(loss_gaps)
            avg_accuracy_gap = np.nanmean(accuracy_gaps)

        # print("init index {}, loss {}".format(init_index, avg_loss_gap))
        # print("init index {}, accuracy {}".format(init_index, avg_accuracy_gap))

        if first:
            first = False
            ax1.scatter(map_index_to_plot_position(init_index), avg_loss_gap, label="mean", color="black")
        else:
            ax1.scatter(map_index_to_plot_position(init_index), avg_loss_gap, color="black")

        ax2.scatter(map_index_to_plot_position(init_index), avg_accuracy_gap, color="black")

        box_plot_data["loss"].append(loss_gaps)
        box_plot_data["acc"].append(accuracy_gaps)
        box_plot_data["labels"].append(map_index_to_plot_position(init_index))

    ax1.boxplot(box_plot_data["loss"], positions=box_plot_data["labels"])
    ax2.boxplot(box_plot_data["acc"], positions=box_plot_data["labels"])

    resolution = 2
    inits = np.arange(9)
    gaps = np.linspace(*ax2.get_ylim(), resolution)
    X, Y = np.meshgrid(inits, gaps, sparse=False, indexing='ij')

    # avg_final_acc = np.array([i/10 for i in range(9)])
    Z = np.array([[avg_final_acc[i]] * resolution for i in init_index_order])

    cmap = matplotlib.cm.get_cmap(name="Spectral_r")
    pcm = ax2.pcolormesh(X, Y, Z, cmap=cmap, shading="gouraud", linewidth=0, zorder=-999)
    cbar = fig.colorbar(pcm, ax=ax2, extend='max')
    cbar.ax.set_title('Average Final\nTest Accuracy')

    gaps = np.linspace(*ax1.get_ylim(), resolution)
    X, Y = np.meshgrid(inits, gaps, sparse=False, indexing='ij')

    # avg_final_acc = np.array([i/10 for i in range(9)])
    Z = np.array([[np.log(avg_final_losses[i])] * resolution for i in init_index_order])

    cmap = matplotlib.cm.get_cmap(name="Spectral_r")
    pcm = ax1.pcolormesh(X, Y, Z, cmap=cmap, shading="gouraud", linewidth=0, zorder=-999)
    cbar = fig.colorbar(pcm, ax=ax1, extend='max')
    cbar.ax.set_title('Average Final\nTest Log Loss')

    ax1.set_title("Loss gap")
    ax1.set_xlabel("Initialisation")
    ax1.set_ylabel("Generalization gap")
    ax1.set_yscale("log")
    ax1.set_xticks(np.arange(9))
    ax1.set_xticklabels(["L4", "L3", "L2", "L1", "C", "R1", "R2", "R3", "R4"])
    # ax1.set_xticklabels(init_index_order)
    ax1.legend()

    ax2.set_title("Accuracy gap")
    ax2.set_xlabel("Initialisation")
    ax2.set_ylabel("Generalization gap")
    ax2.set_yscale("log")
    ax2.set_xticks(np.arange(9))
    ax2.set_xticklabels(["L4", "L3", "L2", "L1", "C", "R1", "R2", "R3", "R4"])
    # ax2.set_xticklabels(init_index_order)
    # ax2.legend()

    # fig.suptitle("Generalisation gap vs initialisation", fontsize=20)

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plot_variance_prop_bounds_with_init_sampling(noise_type, noise_level, hyperparam_index, name="init_sampling.pdf"):
    hyperparam_indices = hyperparam_indices_from_index(hyperparam_index)
    hyperparams = get_hyperparameters(hyperparam_indices)
    n_layers = hyperparams[2]
    init_samples, [left_bound, right_bound] = get_initialisations(noise_type, noise_level, n_layers, give_boundaries=True)

    max_depth = 0
    init_variance_axis = np.linspace(0, right_bound * 1.1, 1000)
    depth_per_theory = variance_prop_depth(noise_type, init_variance_axis, float(noise_level))
    max_depth = np.max([max_depth, np.max(depth_per_theory)])
    crit_point = init_samples[0]

    plt.figure(figsize=(6, 3))

    plt.plot(init_variance_axis, depth_per_theory, label="depth bound", c='cyan', linewidth=3)
    height = plt.ylim()[-1]
    plt.plot([left_bound,]*2, [0, n_layers], color="red", linewidth=3)
    plt.plot([right_bound,]*2, [0, n_layers], color="red", linewidth=3)
    plt.plot([left_bound, right_bound], [n_layers,]*2, color="red", linewidth=3, label="boundary")

    plt.plot([crit_point,]*2, [0, height], color="blue", label="criticality", linewidth=3)

    labels = ["L4", "L3", "L2", "L1", "C", "R1", "R2", "R3", "R4"]
    for index, init in enumerate(np.sort(init_samples)):
        plt.plot([init,]*2, [0, height], color="black", linestyle="--", linewidth=2, dashes=(2, 2))
        plt.text(init + 0.05, 1 + (index % 2) * 10, labels[index], fontsize=15, color="green")

    # plt.text(crit_point + 0.1, 1, "C", fontsize=12, color="green")

    plt.ylim(0, 1.5 * n_layers)
    plt.xlim(0, right_bound * 1.1)
    plt.legend(loc="upper right")
    # ax1.set_xticks(inits[2:-2:3])
    # plt.text(0.2, 400, 'Underflow', fontsize=25, color="white")
    # plt.text(1.7, 400, 'Overflow', fontsize=25, color="white")

    if noise_type.lower() == "dropout":
        plt.title("Initialisation sampling technique for\n{} layer network with $p$ = {} {}".format(n_layers, noise_level, noise_type))
    elif noise_type.lower() == "gauss":
        plt.title("Initialisation sampling technique for\n{} layer network with $\sigma_\epsilon^2$ = {} {}".format(n_layers, noise_level, noise_type))
    else:
        plt.title("Initialisation sampling technique for\n{} layer network with {} {}".format(n_layers, noise_level, noise_type))

    plt.ylabel("Number of layers ($L$)")
    plt.xlabel("Initialisation variance  ($\sigma_w^2$)")
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plot_variance_prop_bounds_with_acc(noise_type, noise_level, hyperparam_index, bounds_with_acc_data, name="acc_within_bound.pdf"):
    hyperparam_indices = hyperparam_indices_from_index(hyperparam_index)
    hyperparams = get_hyperparameters(hyperparam_indices)
    n_layers = hyperparams[2]
    init_samples, [left_bound, right_bound] = get_initialisations(noise_type, noise_level, n_layers, give_boundaries=True)

    max_depth = 0
    init_variance_axis = np.linspace(0, right_bound * 1.1, 1000)
    depth_per_theory = variance_prop_depth(noise_type, init_variance_axis, float(noise_level))
    max_depth = np.max([max_depth, np.max(depth_per_theory)])
    crit_point = init_samples[0]

    fig = plt.figure(figsize=(6, 3))

    plt.plot(init_variance_axis, depth_per_theory, label="depth bound", c='cyan', linewidth=3)
    height = plt.ylim()[-1]
    plt.plot([left_bound,]*2, [0, n_layers], color="red", linewidth=3)
    plt.plot([right_bound,]*2, [0, n_layers], color="red", linewidth=3)
    plt.plot([left_bound, right_bound], [n_layers,]*2, color="red", linewidth=3)

    plt.plot([crit_point,]*2, [0, height], color="blue", label="criticality", linewidth=3)

    for index, init in enumerate(np.sort(init_samples)):
        plt.plot([init,]*2, [0, height], color="black", linestyle="--", linewidth=2, dashes=(2, 2))

    X, Y = np.meshgrid(np.sort(init_samples), np.linspace(0, n_layers, 2, dtype=int), sparse=False, indexing='ij')
    Z = np.array([[bounds_with_acc_data[index]] * 2 for index in np.argsort(init_samples)])

    cmap = matplotlib.cm.get_cmap(name="Spectral_r")
    pcm = plt.pcolormesh(X, Y, Z, cmap=cmap, shading="gouraud", linewidth=0, zorder=-999)
    cbar = fig.colorbar(pcm, extend='max')
    cbar.ax.set_title('Average Final\nTest Accuracy')

    # # check for overflow
    # nan_indices = np.isnan(bounds_with_acc_data)
    # if np.sum(nan_indices):
    #     first_nan = np.argmax(nan_indices)


    plt.ylim(0, 1.25 * n_layers)
    # plt.xlim(0, right_bound * 1.1)
    plt.xlim(left_bound * 0.9, right_bound * 1.1)
    plt.xscale("log")
    plt.legend(loc="upper left")
    # ax1.set_xticks(inits[2:-2:3])
    # plt.text(0.2, 400, 'Underflow', fontsize=25, color="white")
    # plt.text(1.7, 400, 'Overflow', fontsize=25, color="white")

    if noise_type.lower() == "dropout":
        plt.title("Training accuracy per initialisation for\n{} layer network with $p$ = {} {}".format(n_layers, noise_level, noise_type))
    elif noise_type.lower() == "gauss":
        plt.title("Training accuracy per initialisation for\n{} layer network with $\sigma_\epsilon^2$ = {} {}".format(n_layers, noise_level, noise_type))
    else:
        plt.title("Training accuracy per initialisation for\n{} layer network with {} {}".format(n_layers, noise_level, noise_type))

    plt.ylabel("Number of layers ($L$)")
    plt.xlabel("Initialisation variance  ($\sigma_w^2$)")
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plot_gen_gap_per_noise(data, averaging_mode="all", name="noise_plot.pdf"):
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    for init_index in data:
        if averaging_mode == "per_acc":
            ####################################################################
            ####################################################################
            raise NotImplementedError("'per_acc' averaging has not been implemented yet.")
        else:
            ####################################################################
            # add std contour lines?
            ####################################################################
            epochs = []
            loss_gaps = []
            accuracy_gaps = []
            for hyperparam_index in data[init_index]:
                current_test = data[init_index][hyperparam_index]
                epochs += list(current_test["epochs"])
                loss_gaps += list(current_test["loss_gap"])
                accuracy_gaps += list(current_test["accuracy_gap"])

            avg_epoch = np.nanmean(epochs)
            avg_loss_gap = np.nanmean(loss_gaps)
            avg_accuracy_gap = np.nanmean(accuracy_gaps)

        ax1.scatter(avg_epoch, avg_loss_gap, label=init_index)
        ax2.scatter(avg_epoch, avg_accuracy_gap, label=init_index)

    ax1.set_title("loss gap")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("generalization gap")
    ax1.set_yscale("log")
    ax1.legend()

    ax2.set_title("accuracy gap")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("generalization gap")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def plot_epoch_vs_gen_gap(data, name="plot.pdf"):
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    labled = []
    for hyperparam_index in data:
        epochs = data[hyperparam_index]["epochs"]
        for i in np.arange(epochs.shape[0]):
            if i not in labled:
                labled.append(i)
                ax1.scatter(epochs[i], data[hyperparam_index]["loss_gap"][i], color="blue", alpha=accuracy_thresholds[i], label="{:.1f}".format(accuracy_thresholds[i]))
                ax2.scatter(epochs[i], data[hyperparam_index]["accuracy_gap"][i], color="blue", alpha=accuracy_thresholds[i], label="{:.1f}".format(accuracy_thresholds[i]))
            else:
                ax1.scatter(epochs[i], data[hyperparam_index]["loss_gap"][i], color="blue", alpha=accuracy_thresholds[i])
                ax2.scatter(epochs[i], data[hyperparam_index]["accuracy_gap"][i], color="blue", alpha=accuracy_thresholds[i])

    # add means of each acc threshold as big crosses?
    avg_data = {
        i: {"epochs": [], "loss_gap": [], "accuracy_gap": []} for i in np.arange(10)
    }
    for hyperparam_index in data:
        epochs = data[hyperparam_index]["epochs"]
        for i in np.arange(epochs.shape[0]):
            avg_data[i]["epochs"].append(epochs[i])
            avg_data[i]["loss_gap"].append(data[hyperparam_index]["loss_gap"][i])
            avg_data[i]["accuracy_gap"].append(data[hyperparam_index]["accuracy_gap"][i])

    for i in avg_data:
        threshold_data = avg_data[i]

        if threshold_data["epochs"]:
            avg_epoch = np.nanmean(threshold_data["epochs"])

            if threshold_data["loss_gap"]:
                avg_loss_gap = np.nanmean(threshold_data["loss_gap"])
                ax1.scatter(avg_epoch, avg_loss_gap, color="blue", alpha=accuracy_thresholds[i], marker="x", s=200)

            if threshold_data["accuracy_gap"]:
                avg_accuracy_gap = np.nanmean(threshold_data["accuracy_gap"])
                ax2.scatter(avg_epoch, avg_accuracy_gap, color="blue", alpha=accuracy_thresholds[i], marker="x", s=200)

    ax1.set_title("loss gap")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("generalization gap")
    ax1.set_yscale("log")
    ax1.legend()

    ax2.set_title("accuracy gap")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("generalization gap")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(name)
    plt.close()

if __name__ == "__main__":
    plot_variance_prop_bounds_with_init_sampling("dropout", 0.7, 11)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.abspath(os.path.join(current_dir, "../plotting/generalization_figures"))
    os.makedirs(plot_dir, exist_ok=True)

    print("Scanning system for data...")
    print("training data...")
    train_path = os.path.join(current_dir, "../", "plotting", "train")
    train_data = load_structured_directory_data(train_path, progress_file_name="temp_train_paths.npz", data_file_name="temp_train_data.npz", load_data=True)

    print("validation data...")
    validation_path = os.path.join(current_dir, "../", "plotting", "test")
    validation_data = load_structured_directory_data(validation_path, progress_file_name="temp_validation_paths.npz", data_file_name="temp_validation_data.npz", load_data=True)

    print("Processing training data...")
    for noise_type in train_data:
        # print("Noise type: {}".format(noise_type))

        for noise_level in train_data[noise_type]:
            # print("Noise level: {}".format(noise_level))

            new_init_dict = {}

            for hyperparam_index in train_data[noise_type][noise_level]:
                # print("Hyper-param index: {}".format(hyperparam_index))

                for init_index in train_data[noise_type][noise_level][hyperparam_index]:

                    if init_index not in new_init_dict:
                        new_init_dict[init_index] = {}

                    if hyperparam_index not in new_init_dict[init_index]:
                        new_init_dict[init_index][hyperparam_index] = {}

                    new_init_dict[init_index][hyperparam_index]["losses"] = []
                    new_init_dict[init_index][hyperparam_index]["accuracies"] = []
                    new_init_dict[init_index][hyperparam_index]["epoch_order"] = []

                    for epoch in train_data[noise_type][noise_level][hyperparam_index][init_index]:
                        acc = train_data[noise_type][noise_level][hyperparam_index][init_index][epoch]["accuracy"][0]
                        loss = train_data[noise_type][noise_level][hyperparam_index][init_index][epoch]["loss"][0]

                        new_init_dict[init_index][hyperparam_index]["losses"].append(loss)
                        new_init_dict[init_index][hyperparam_index]["accuracies"].append(acc)
                        new_init_dict[init_index][hyperparam_index]["epoch_order"].append(epoch)

                        new_init_dict[init_index][hyperparam_index][epoch] = {
                            "loss": loss,
                            "accuracy": acc
                        }

                    epoch_sort_order = np.argsort(new_init_dict[init_index][hyperparam_index]["epoch_order"])
                    new_init_dict[init_index][hyperparam_index]["epoch_order"] = np.array(
                        new_init_dict[init_index][hyperparam_index]["epoch_order"]
                    )[epoch_sort_order]
                    new_init_dict[init_index][hyperparam_index]["accuracies"] = np.array(
                        new_init_dict[init_index][hyperparam_index]["accuracies"]
                    )[epoch_sort_order]
                    new_init_dict[init_index][hyperparam_index]["losses"] = np.array(
                        new_init_dict[init_index][hyperparam_index]["losses"]
                    )[epoch_sort_order]

                    difference = np.abs(
                        accuracy_thresholds -
                        new_init_dict[init_index][hyperparam_index]["accuracies"][:, np.newaxis]
                    )
                    index_when_accuracy_threshold_achieved = np.argmin(difference, axis=0) # double check that this is the correct axis!!!!!!!!

                    # check that accuracy values are actually reached (using some tolerance)
                    tolerance = 0.01
                    closest_differences = difference[index_when_accuracy_threshold_achieved]
                    indices_of_accuracies_out_of_tolerance = (
                        tolerance <
                        closest_differences.diagonal() *
                            np.less(
                                new_init_dict[init_index][hyperparam_index]["accuracies"][index_when_accuracy_threshold_achieved],
                                accuracy_thresholds
                            )
                    )
                    index_when_accuracy_threshold_achieved = np.ma.masked_array( # I may need to change the `fill_value` here
                        index_when_accuracy_threshold_achieved, mask=indices_of_accuracies_out_of_tolerance
                    )

                    new_init_dict[init_index][hyperparam_index]["accuracy_threshold_indices"] = index_when_accuracy_threshold_achieved

                    ###################################################################################################################
                    ###################################################################################################################
                    # If the accuracy jumps by two threshold values in one epoch, I'm not too sure how things will react...
                    ###################################################################################################################
                    ###################################################################################################################

            train_data[noise_type][noise_level] = new_init_dict

    # ################################################################################
    # # I can definitely merge these two loops
    # ################################################################################

    print("Processing validation data...")
    for noise_type in validation_data:
        # print("Noise type: {}".format(noise_type))

        for noise_level in validation_data[noise_type]:
            # print("Noise level: {}".format(noise_level))

            new_init_dict = {}

            for hyperparam_index in validation_data[noise_type][noise_level]:
                # print("Hyper-param index: {}".format(hyperparam_index))

                for init_index in validation_data[noise_type][noise_level][hyperparam_index]:

                    if init_index not in new_init_dict:
                        new_init_dict[init_index] = {}

                    if hyperparam_index not in new_init_dict[init_index]:
                        new_init_dict[init_index][hyperparam_index] = {}

                    new_init_dict[init_index][hyperparam_index]["losses"] = []
                    new_init_dict[init_index][hyperparam_index]["accuracies"] = []
                    new_init_dict[init_index][hyperparam_index]["epoch_order"] = []

                    for epoch in validation_data[noise_type][noise_level][hyperparam_index][init_index]:

                        acc = validation_data[noise_type][noise_level][hyperparam_index][init_index][epoch]["accuracy"][0]
                        loss = validation_data[noise_type][noise_level][hyperparam_index][init_index][epoch]["loss"][0]

                        new_init_dict[init_index][hyperparam_index]["losses"].append(loss)
                        new_init_dict[init_index][hyperparam_index]["accuracies"].append(acc)
                        new_init_dict[init_index][hyperparam_index]["epoch_order"].append(epoch)

                        new_init_dict[init_index][hyperparam_index][epoch] = {
                            "loss": loss,
                            "accuracy": acc
                        }

                    epoch_sort_order = np.argsort(new_init_dict[init_index][hyperparam_index]["epoch_order"])
                    new_init_dict[init_index][hyperparam_index]["epoch_order"] = np.array(
                        new_init_dict[init_index][hyperparam_index]["epoch_order"]
                    )[epoch_sort_order]
                    new_init_dict[init_index][hyperparam_index]["accuracies"] = np.array(
                        new_init_dict[init_index][hyperparam_index]["accuracies"]
                    )[epoch_sort_order]
                    new_init_dict[init_index][hyperparam_index]["losses"] = np.array(
                        new_init_dict[init_index][hyperparam_index]["losses"]
                    )[epoch_sort_order]

            validation_data[noise_type][noise_level] = new_init_dict

    # hyp_index = "7" # depth 50
    # hyp_index = "18" # depth 50
    # hyp_index = "25" # depth 30
    # hyp_index = "4" # depth 10
    # hyp_index = "14" # depth 25
    # hyp_index = "27" # depth 20
    hyp_index = "28" # depth 15
    bounds_with_acc_data = {}
    for init in validation_data["dropout"]["0.7"]:
        if hyp_index in validation_data["dropout"]["0.7"][init]:
            bounds_with_acc_data[int(init)] = validation_data["dropout"]["0.7"][init][hyp_index]["accuracies"][-1]
        else:
            # not entirely sure what is happening here, either the model nan-ed
            # in the first forward pass (which it shouldn't...) or we have lost the files...
            bounds_with_acc_data[int(init)] = 0
            # bounds_with_acc_data[int(init)] = np.float("nan")

    # bounds_with_acc_data = {int(init): validation_data["dropout"]["0.7"][init][hyp_index]["accuracies"][-1] for init in validation_data["dropout"]["0.7"]}
    plot_variance_prop_bounds_with_acc("dropout", 0.7, int(hyp_index), bounds_with_acc_data)

    generalization_data = {}
    print("Calculating generalisation data...")
    for noise_type in train_data:
        print("Noise type: {}".format(noise_type))

        if noise_type not in generalization_data:
            generalization_data[noise_type] = {}

        for noise_level in train_data[noise_type]:
            print("Noise level: {}".format(noise_level))

            if noise_level not in generalization_data[noise_type]:
                generalization_data[noise_type][noise_level] = {}

            avg_final_accs = np.zeros((9,))
            avg_final_losses = np.zeros((9,))
            for init_index in train_data[noise_type][noise_level]:
                print("Init index: {}".format(init_index))

                if init_index not in generalization_data[noise_type][noise_level]:
                    generalization_data[noise_type][noise_level][init_index] = {}

                final_hyperparam_losses = []
                final_hyperparam_accs = []
                for hyperparam_index in train_data[noise_type][noise_level][init_index]:
                    threshold_indices = train_data[noise_type][noise_level][init_index][hyperparam_index]["accuracy_threshold_indices"]

                    # removed invalid entries
                    threshold_indices = threshold_indices[True ^ threshold_indices.mask]

                    gen_gap_data = {"epochs": threshold_indices}

                    try:
                        gen_gap_data["loss_gap"] = np.abs(
                            train_data[noise_type][noise_level][init_index][hyperparam_index]["losses"][threshold_indices] -
                            validation_data[noise_type][noise_level][init_index][hyperparam_index]["losses"][threshold_indices]
                            # train_data[noise_type][noise_level][init_index][hyperparam_index]["losses"][threshold_indices] /
                            # validation_data[noise_type][noise_level][init_index][hyperparam_index]["losses"][threshold_indices]
                        )
                    except KeyError:
                        continue
                        gen_gap_data["loss_gap"] = np.array([ np.float("nan") ] * threshold_indices.shape[0])

                    try:
                        gen_gap_data["accuracy_gap"] = np.abs(
                            train_data[noise_type][noise_level][init_index][hyperparam_index]["accuracies"][threshold_indices] -
                            validation_data[noise_type][noise_level][init_index][hyperparam_index]["accuracies"][threshold_indices]
                        )
                    except KeyError:
                        continue
                        gen_gap_data["accuracy_gap"] = np.array([ np.float("nan") ] * threshold_indices.shape[0])

                    generalization_data[noise_type][noise_level][init_index][hyperparam_index] = gen_gap_data

                    try:
                        final_hyperparam_losses.append(validation_data[noise_type][noise_level][init_index][hyperparam_index]["losses"][-1])
                    except KeyError:
                        continue
                        final_hyperparam_losses.append(np.inf)

                    try:
                        final_hyperparam_accs.append(validation_data[noise_type][noise_level][init_index][hyperparam_index]["accuracies"][-1])
                    except KeyError:
                        continue
                        final_hyperparam_accs.append(0)

                avg_final_losses[int(init_index)] = np.nanmean(final_hyperparam_losses)
                avg_final_accs[int(init_index)] = np.nanmean(final_hyperparam_accs)
            #     plot_epoch_vs_gen_gap(
            #         generalization_data[noise_type][noise_level][init_index],
            #         name=os.path.join(
            #             plot_dir, "{}_{}_{}.pdf".format(
            #                 noise_type, noise_level, init_index
            #             )
            #         )
            #     )

            # plot_gen_gap_per_noise(
            #     generalization_data[noise_type][noise_level],
            #     name=os.path.join(
            #         plot_dir, "{}_{}.pdf".format(noise_type, noise_level)
            #     )
            # )
            plot_gen_gap_vs_init(
                generalization_data[noise_type][noise_level], avg_final_accs, avg_final_losses,
                name=os.path.join(
                    plot_dir,
                    "{}_{}_vs_init.pdf".format(noise_type, noise_level)
                )
            )
