import os, sys

if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(curr_dir, "../results")
    extract_dir = os.path.join(curr_dir, sys.argv[1])
    tar_contents = sys.argv[2]
    files_to_extract = sys.argv[3]
    moves_for_extracted_files = sys.argv[4]

    # read in the file containing model state names that are present in the tar file
    with open(tar_contents, "r") as f:
        compressed_models = f.readlines()

    # find the last epoch for each test
    models_to_extract = {}
    for model in compressed_models:
        model = model.strip()
        noise_type, noise_level, hyper_param_index, model_file_name = model.split("/")[-4:]

        if ".txt" not in model_file_name:
            init_index, epoch = model_file_name.split(".")[0].split("_")[-2:]

            epoch = int(epoch)

            if int(hyper_param_index) > 30:
                continue

            if noise_type not in models_to_extract:
                models_to_extract[noise_type] = {}

            if noise_level not in models_to_extract[noise_type]:
                models_to_extract[noise_type][noise_level] = {}

            if hyper_param_index not in models_to_extract[noise_type][noise_level]:
                models_to_extract[noise_type][noise_level][hyper_param_index] = {}

            if init_index not in models_to_extract[noise_type][noise_level][hyper_param_index]:
                models_to_extract[noise_type][noise_level][hyper_param_index][init_index] = {
                    "epoch": -1,
                    "path": "",
                    "file_name": ""
                }

            current_max_epoch = int(models_to_extract[noise_type][noise_level][hyper_param_index][init_index]["epoch"])

            if epoch > current_max_epoch:
                models_to_extract[noise_type][noise_level][hyper_param_index][init_index]["path"] = model
                models_to_extract[noise_type][noise_level][hyper_param_index][init_index]["epoch"] = epoch
                models_to_extract[noise_type][noise_level][hyper_param_index][init_index]["file_name"] = model_file_name

    # check if this state is from the same or later epoch as already extracted files
    for noise_type in list(models_to_extract):
        noise_type_results_dir = os.path.join(results_dir, noise_type)
        if not os.path.exists(noise_type_results_dir):
            continue

        states_per_noise = models_to_extract[noise_type]
        for noise_level in list(states_per_noise):
            noise_level_results_dir = os.path.join(noise_type_results_dir, noise_level)
            if not os.path.exists(noise_level_results_dir):
                continue

            states_per_level = states_per_noise[noise_level]
            for hyper_param_index in list(states_per_level):
                hyper_param_results_dir = os.path.join(noise_level_results_dir, hyper_param_index)
                if not os.path.exists(hyper_param_results_dir):
                    continue

                model_states_in_dir = os.listdir(hyper_param_results_dir)

                states_per_hyper = states_per_level[hyper_param_index]
                for init_index in list(states_per_hyper):
                    states_per_init = states_per_hyper[init_index]
                    file_name = states_per_init["file_name"]
                    experiment_details = "_".join(file_name.split(".")[0].split("_")[:-1])
                    states_of_same_experiment = filter(
                        lambda name: experiment_details in name,
                        model_states_in_dir
                    )

                    # check if later epoch exists in directory
                    for state in states_of_same_experiment:
                        if state >= file_name:
                            # if so, remove this file from dict of files to extract
                            del models_to_extract[noise_type][noise_level][hyper_param_index][init_index]

                # do some clean up to make sure there aren't empty dictionaries
                if not models_to_extract[noise_type][noise_level][hyper_param_index]:
                    del models_to_extract[noise_type][noise_level][hyper_param_index]

            if not models_to_extract[noise_type][noise_level]:
                del models_to_extract[noise_type][noise_level]

        if not models_to_extract[noise_type]:
            del models_to_extract[noise_type]

    # create / clear files to store what will be extracted and to where
    open(files_to_extract, "w")
    open(moves_for_extracted_files, "w")

    if not models_to_extract:
        # end program / erase file contents?
        print("No files to be extracted.")

    else:
        with open(files_to_extract, "a") as extract_file:
            with open(moves_for_extracted_files, "a") as move_file:
                for noise_type in models_to_extract:
                    states_per_noise = models_to_extract[noise_type]
                    for noise_level in states_per_noise:
                        states_per_level = states_per_noise[noise_level]
                        for hyper_param_index in states_per_level:
                            states_per_hyper = states_per_level[hyper_param_index]
                            for init_index in states_per_hyper:
                                states_per_init = states_per_hyper[init_index]

                                path = states_per_init["path"]
                                extract_file.write("{}\n".format(path))

                                index_of_results_dir = -1
                                path_directories = path.split("/")
                                for i, directory in enumerate(path_directories):
                                    if directory == "results":
                                        index_of_results_dir = i
                                        break

                                new_path = os.path.abspath(os.path.join(
                                    results_dir, *path_directories[index_of_results_dir + 1 : -1]
                                ))
                                move_file.write("{} {}\n".format(
                                    os.path.join(extract_dir, path),
                                    new_path
                                ))
