import os, sys

if __name__ == "__main__":
    curr_dir = os.path.abspath(sys.argv[1])
    tar_contents = sys.argv[2]
    files_to_extract = sys.argv[3]

    # read in the file containing model state names that are present in the tar file
    with open(tar_contents, "r") as f:
        compressed_models = f.readlines()

    # find the last epoch for each test
    models_to_extract = {}
    for model in compressed_models:
        model = model.strip()

        directories = model.split("/")
        if len(directories) >= 2 and directories[-1]:
            hyper_param_index, model_file_name = directories[-2:]

            if ".txt" not in model_file_name:
                init_index, epoch = model_file_name.split(".")[0].split("_")[-2:]

                epoch = int(epoch)

                if int(hyper_param_index) <= 30:
                    if hyper_param_index not in models_to_extract:
                        models_to_extract[hyper_param_index] = {}

                    if init_index not in models_to_extract[hyper_param_index]:
                        models_to_extract[hyper_param_index][init_index] = {
                            "epoch": -1,
                            "path": "",
                            "file_name": ""
                        }

                    current_max_epoch = int(models_to_extract[hyper_param_index][init_index]["epoch"])

                    if epoch > current_max_epoch:
                        models_to_extract[hyper_param_index][init_index]["path"] = model
                        models_to_extract[hyper_param_index][init_index]["epoch"] = epoch

    # check if this state is from the same or later epoch as already extracted files
    for hyper_param_index in list(models_to_extract):
        hyper_param_results_dir = os.path.join(curr_dir, hyper_param_index)

        if os.path.exists(hyper_param_results_dir):
            model_states_in_dir = os.listdir(hyper_param_results_dir)

            states_per_hyper = models_to_extract[hyper_param_index]
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
                        del models_to_extract[hyper_param_index][init_index]

        # do some clean up to make sure there aren't empty dictionaries
        if not models_to_extract[hyper_param_index]:
            del models_to_extract[hyper_param_index]

    # create / clear files to store what will be extracted and to where
    open(files_to_extract, "w")

    if not models_to_extract:
        # end program / erase file contents?
        print("No files to be extracted.")

    else:
        with open(files_to_extract, "a") as extract_file:
            for hyper_param_index in models_to_extract:
                states_per_hyper = models_to_extract[hyper_param_index]
                for init_index in states_per_hyper:
                    states_per_init = states_per_hyper[init_index]

                    path = states_per_init["path"]
                    extract_file.write("{}\n".format(path))
