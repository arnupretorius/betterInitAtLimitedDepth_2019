import sys, os
import numpy as np

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIRECTORY, '..'))

from src.utils import get_hyperparameters, recreate_model
from src.sensitivity_utils import get_data, test_sensitivity, get_output_distributions

FORCE_TEST_RE_RUN = False

DATASET = sys.argv[1]
TRAINING_DIRECTORY = sys.argv[8]
BASE_SENSITIVITY_DIRECTORY = sys.argv[10]
RESULTS_DIRECTORY = os.path.join(BASE_SENSITIVITY_DIRECTORY, DATASET)

def fetch_model_acc(dataset, noise_type, noise_level, hyperparam_index, init_index, epoch, model_path):
    data_path = os.path.join(TRAINING_DIRECTORY, dataset, noise_type, noise_level, hyperparam_index, init_index)
    model_details = "_".join(model_path.split(".")[-2].split("/")[-1].split("_")[:-1])
    train_data_file_path = os.path.join(data_path, "{}_train_accuracy.npy".format(model_details))
    test_data_file_path = os.path.join(data_path, "{}_test_accuracy.npy".format(model_details))

    epoch = int(epoch)
    train_accuracy = np.load(train_data_file_path)[epoch]
    test_accuracy = np.load(test_data_file_path)[epoch]

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    # make results directory
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    # get details of the experiment that is to be run
    dataset = sys.argv[1]
    noise_type = sys.argv[2]
    noise_level = sys.argv[3]
    hyperparam_index = sys.argv[4]
    init_index = sys.argv[5]
    epoch = sys.argv[6]
    model_path = sys.argv[7]
    data_path = sys.argv[9]

    print("#######################################################################")
    print("Checking Experiment:")
    print("Dataset: {}".format(dataset))
    print("Noise type: {}".format(noise_type))
    print("Noise level: {}".format(noise_level))
    print("Hyper-parameter index: {}".format(hyperparam_index))
    print("Initialization index: {}".format(init_index))
    print("Epoch: {}".format(epoch))
    print("Model path: {}".format(model_path))
    print("#######################################################################")

    if int(hyperparam_index) <= 30:
        print("Don't analyse old test results, skipping...")
        exit(0)

    # check if model path has already produced results
    experiment_results_directory = os.path.abspath(os.path.join(
        RESULTS_DIRECTORY, noise_type, noise_level, hyperparam_index
    ))
    results_file_name = "{}.npz".format(model_path.split("/")[-1].split(".")[0])
    results_file_path = os.path.join(experiment_results_directory, results_file_name)

    print(results_file_path)

    # check if test has already been run
    if not FORCE_TEST_RE_RUN and os.path.exists(results_file_path):
        print("Test has already been completed, test will be skipped.")
    else:
        print("Test has not been run yet, test will start now.")
        os.makedirs(experiment_results_directory, exist_ok=True)

        # print some model details
        hyperparams = list(map(int, results_file_name.split(".")[0].split("_")))
        hyperparams = get_hyperparameters(hyperparams, noise_type, noise_level)
        [batch_size, depth, width], learning_rate, optimizer = hyperparams[1:4], hyperparams[5], hyperparams[7]
        print("#########################################")
        print("Model Hyper-parameters:")
        print("Batch size: {}".format(batch_size))
        print("Number of layers: {}".format(depth))
        print("Width: {}".format(width))
        print("Learning rate: {}".format(learning_rate))
        print("Optimizer: {}".format(optimizer))
        print("#########################################")

        ########################## perform the tests ########################
        # load the model
        model = recreate_model(model_path, dataset=dataset)

        # check the models training accuracy
        print("Fetching training accuracy...")
        # accuracy = check_accuracy(model)
        train_accuracy, test_accuracy = fetch_model_acc(dataset, noise_type, noise_level, hyperparam_index, init_index, epoch, model_path)

        # load the sensitivity dataset
        print("Loading sensitivity dataset...")
        # NEED TO CHECK BOTH MODES WITH MULTIPLE CLASSES!!!!!!
        sensitivity_data = get_data(dataset, data_path, mode="inter_class") # NEED TO CHECK BOTH MODES WITH MULTIPLE CLASSES!!!!!!
        # NEED TO CHECK BOTH MODES WITH MULTIPLE CLASSES!!!!!!

        # check the models output distribution on the sensitivity dataset
        print("Getting output distribution on sensitivity dataset...")
        output_distribution = get_output_distributions(model, sensitivity_data)

        # check the models sensitivity
        print("Testing sensitivity...")
        sensitivity = test_sensitivity(model, sensitivity_data)

        # save results
        np.savez_compressed(
            results_file_path, train_accuracy=train_accuracy, test_accuracy=test_accuracy, output_distribution=output_distribution, sensitivity=sensitivity
        )

        print("Test complete and results saved to {}!".format(results_file_path))
