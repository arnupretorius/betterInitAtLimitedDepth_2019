import sys, os
import numpy as np

from tqdm import tqdm
from shutil import copy2
from multiprocessing import Pool

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, ".."))

from src.utils import load_structured_directory_data

def copy(src, dst, skip_already_present=True):
    if os.path.exists(dst) and skip_already_present:
        print("{} already present at {}, skipping copy.".format(src, dst))
    else:
        print("Copying {} to {}...".format(src, dst))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        copy2(src, dst)

def copy_wrapper(args):
    copy(*args)

def multi_thread_copy(source_paths, destination_paths):
    with Pool() as p:
        p.map(copy_wrapper, zip(source_paths, destination_paths))

def nan_during_training(model_dir, trajectory_file):
    if os.path.exists(model_dir):
        directory_contents = os.listdir(model_dir)
        nan_files = filter(lambda x: "NaN.txt" in x, directory_contents)

        experiment_details = "_".join(trajectory_file.split(".")[-2].split("_")[:-2])

        for nan_file in nan_files:
            if experiment_details in nan_file:
                return True

    return False

def construct_command(dataset, trajectory_file):
    experiment_details = trajectory_file.split(".")[-2].split("_")[:-2]
    hyp_indices = "_".join(experiment_details[:-3])
    exp_indices = "_".join(experiment_details[-3:])
    return "python experiment.py {} {} {}".format(hyp_indices, exp_indices, dataset)

if __name__ == "__main__":
    num_epochs = int(sys.argv[1])
    local_old_training_dir = sys.argv[2]
    threads_dir = sys.argv[3]
    shared_training_dir = sys.argv[4]
    local_models_path = sys.argv[5]

    source_paths = []
    destination_paths = []
    completed_commands = []

    print("Decend local old plotting directory and see what is present...")
    if os.path.exists(local_old_training_dir):
        datasets = os.listdir(local_old_training_dir)
        for dataset in tqdm(datasets, desc="datasets"):
            relative_dataset_path = dataset
            current_path = os.path.join(local_old_training_dir, relative_dataset_path)
            if not os.path.isdir(current_path):
                continue

            noises = os.listdir(current_path)
            for noise in tqdm(noises, desc="noises"):
                relative_noise_path = os.path.join(relative_dataset_path, noise)
                current_path = os.path.join(local_old_training_dir, relative_noise_path)
                if not os.path.isdir(current_path):
                    continue

                noise_levels = os.listdir(current_path)
                for noise_level in tqdm(noise_levels, desc="noise_levels"):
                    relative_noise_level_path = os.path.join(relative_noise_path, noise_level)
                    current_path = os.path.join(local_old_training_dir, relative_noise_level_path)
                    if not os.path.isdir(current_path):
                        continue

                    hyper_param_indices = os.listdir(current_path)
                    for hyper_param_index in tqdm(hyper_param_indices, desc="hyper_param_indices"):
                        relative_hyp_path = os.path.join(relative_noise_level_path, hyper_param_index)
                        current_path = os.path.join(local_old_training_dir, relative_hyp_path)
                        if not os.path.isdir(current_path):
                            continue

                        init_indices = os.listdir(current_path)
                        for init_index in tqdm(init_indices, desc="init_indices"):
                            relative_init_path = os.path.join(relative_hyp_path, init_index)
                            current_path = os.path.join(local_old_training_dir, relative_init_path)
                            if not os.path.isdir(current_path):
                                continue

                            trajectory_files = os.listdir(current_path)
                            for trajectory_file in trajectory_files:
                                relative_trajectory_file_path = os.path.join(relative_init_path, trajectory_file)
                                current_path = os.path.join(local_old_training_dir, relative_trajectory_file_path)
                                if not os.path.isfile(current_path):
                                    continue

                                # do some checks to make sure these are the files we want
                                if (
                                    ".npy" in trajectory_file and
                                    any(item in trajectory_file for item in ["train", "test"]) and
                                    any(item in trajectory_file for item in ["loss", "accuracy"])
                                ):
                                    # add the needed paths to copy this file
                                    destination_path = os.path.join(shared_training_dir, relative_trajectory_file_path)
                                    source_paths.append(current_path)
                                    destination_paths.append(destination_path)

                                # check if the test accuracy file has all the data we need
                                if "test_accuracy.npy" in trajectory_file:
                                    try:
                                        data = np.load(current_path)
                                        test_complete = len(data) > num_epochs

                                        if not test_complete:
                                            # the full number of epochs are not here, so check if this model reached 100% accuracy
                                            test_complete = data[-1] == 1.0


                                        if not test_complete:
                                            # the full number of epochs are not here, so check if this model NaNed
                                            model_dir = os.path.join(local_models_path, relative_hyp_path)
                                            test_complete = nan_during_training(model_dir, trajectory_file)
                                    except OSError:
                                        test_complete = False

                                    # if no more training needs to take place, construct the command that would have been used to run it
                                    if test_complete:
                                        command = construct_command(dataset, trajectory_file)

                                        # add this command to an array
                                        completed_commands.append(command)
    else:
        print("No previous results exist.")

    # write completed commands to the previously completed node file
    # first clear the file
    previously_completed_path = os.path.join(threads_dir, "previously_completed.log")
    with open(previously_completed_path, "w") as f:
        f.write("")

    # now write the new information to the file
    with open(previously_completed_path, "a") as f:
        for command in completed_commands:
            f.write("{} {} {} {}\n".format(command, num_epochs, local_models_path, shared_training_dir))

    # copy the plotting files already present on this node to the shared space
    multi_thread_copy(source_paths, destination_paths)
