#!/bin/bash

# Train models given hyperparam and experiment specification files
#
# Author: Ryan Eloff
# Contact: ryan.peter.eloff@gmail.com
# Date: November 2018


# Process keyword arguments
print_help_and_exit=false
while getopts ":p:-:h" opt; do
    # Check option not specified without arg (if arg is required; takes next option "-*" as arg)
    if [[ ${OPTARG} == -* ]]; then
        echo "Option -${opt} requires an argument." >&2 && print_help_and_exit=true
    fi
    # Handle specified option and arguments
    case $opt in
        h)  # print help
            print_help_and_exit=true
            ;;
        :)  # options specified without any arg evaluate to ":" (if arg is required)
            echo "Option -$OPTARG requires an argument." >&2 && print_help_and_exit=true
            ;;
        -)  # long options (--long-option arg)
            case $OPTARG in
                hp|hp=*)
                    val=${OPTARG#hp}  # remove "hp" from the opt arg
                    val=${val#*=}  # get value after "="
		            opt=${OPTARG%=$val}  # get option before "=value" (debug)
                    if [ -z "${val}" ]; then  # check value is not empty
                        echo "Option --${opt} is missing the hyperparams file argument!" >&2 && print_help_and_exit=true
                    else
                        HP_FILE=$val  # assign hyperparam file value
                    fi
		            ;;
                exp|exp=*)
                    val=${OPTARG#exp}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the experiments file argument!" >&2 && print_help_and_exit=true
                    else
                        EXP_FILE=$val
                    fi
                    ;;
                act|act=*)
                    val=${OPTARG#act}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the activation func argument!" >&2 && print_help_and_exit=true
                    else
                        ACTIVATION=$val
                    fi
                    ;;
                dataset|dataset=*)
                    val=${OPTARG#dataset}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the dataset argument!" >&2 && print_help_and_exit=true
                    else
                        DATASET=$val
                    fi
                    ;;
                epochs|epochs=*)
                    val=${OPTARG#epochs}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the epochs argument!" >&2 && print_help_and_exit=true
                    else
                        EPOCHS=$val
                    fi
                    ;;
                docker)
                    USE_DOCKER=true
                    ;;
                image|image=*)
                    val=${OPTARG#image}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the image argument!" >&2 && print_help_and_exit=true
                    else
                        DOCKER_IMAGE=$val
                    fi
                    ;;
                name|name=*)
                    val=${OPTARG#name}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the name argument!" >&2 && print_help_and_exit=true
                    else
                        DOCKER_NAME=$val
                    fi
                    ;;
                thread_id|thread_id=*)
                    val=${OPTARG#thread_id}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the thread_id argument!" >&2 && print_help_and_exit=true
                    else
                        THREAD_ID=$val
                    fi
                    ;;
                command_file|command_file=*)
                    val=${OPTARG#command_file}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the command_file argument!" >&2 && print_help_and_exit=true
                    else
                        COMMAND_FILE=$val
                    fi
                    ;;
                num_threads|num_threads=*)
                    val=${OPTARG#num_threads}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the num_threads argument!" >&2 && print_help_and_exit=true
                    else
                        NUM_THREADS=$val
                    fi
                    ;;
                thread_dir|thread_dir=*)
                    val=${OPTARG#thread_dir}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the thread_dir argument!" >&2 && print_help_and_exit=true
                    else
                        THREAD_DIR=$val
                    fi
                    ;;
		        help)
		            print_help_and_exit=true
		            ;;
                *)
                    echo "Invalid option --${OPTARG}" >&2 && print_help_and_exit=true
                    ;;
            esac
            ;;
        \?)
            echo "Invalid option -${opt} ${OPTARG}" >&2 && print_help_and_exit=true
            ;;
    esac
done


# Print help and exit for invalid input or -h/--help option
if [ "${print_help_and_exit}" = true ]; then
    echo ""
    echo "Usage: run_experiments.sh [OPTIONS]"
    echo ""
    echo "Train models given hyperparam and experiment specification files"
    echo "(Run script with sudo if Docker is used and not set up for non-root users)"
    echo ""
    echo "Options:"
    echo "        --hp string           Hyperparameters file in './experiments' (Default: hyperparams.txt)."
    echo "        --exp string          Experiments file in './experiments' (Default: experiments.txt)."
    echo "        --act string          Activation function to use in experiments (Default: relu)."
    echo "        --dataset string      Dataset to use in experiments (Default: mnist)."
    echo "        --epochs number       Maximum number of epochs to train models (Default: 250)."
    echo "        --docker              Specify that a Docker container will be used to run the experiments"
    echo "                              (Default: run experiments in current environment e.g. default python, venv, conda, etc.)."
    echo "        --image string        Image for the Docker container (Default: reloff/noisy-relu-shift)."
    echo "        --name string         Name for the Docker container (Default: noisy-relu-shift)."
    echo "        --thread_id int       Integer to label the current thread (Default: 1)."
    echo "        --command_file string       Path to a file that contains commands to run (Default: false)."
    echo "        --num_threads int       The number of threads to use to run the commands in command_file (Default: 1)."
    echo "        --thread_dir string       The path where thread info will be saved (Default: false)."
    echo "    -h, --help                Print this information and exit."
    echo ""
    exit 1
fi


# Set hyperparam file (default file: hyperparams.txt)
HP_FILE=${HP_FILE:-hyperparams.txt}
# Set experiments file (default file: experiments.txt)
EXP_FILE=${EXP_FILE:-experiments.txt}
# Set activation func (default function: relu)
ACTIVATION=${ACTIVATION:-relu}
# Set dataset (default: mnist)
DATASET=${DATASET:-mnist}
# Set number of epochs (default: 250)
EPOCHS=${EPOCHS:-500}
# Set thread id (default: 1)
THREAD_ID=${THREAD_ID:-1}
# Set ... (default: 1)
COMMAND_FILE=${COMMAND_FILE:-false}
# Set ... (default: 1)
NUM_THREADS=${NUM_THREADS:-1}
# Set ... (default: 1)
THREAD_DIR=${THREAD_DIR:-false}
# Specify whether to use Docker or current python environment (default false)
USE_DOCKER=${USE_DOCKER:-false}
# Set image for docker container (default image: reloff/noisy-relu-shift)
DOCKER_IMAGE=${DOCKER_IMAGE:-reloff/noisy-relu-shift}
# Set name of docker container (default name: noisy-relu-shift)
DOCKER_NAME=${DOCKER_NAME:-noisy-relu-shift-exp}


# Print some information on selected options
echo "Starting experiments!"
echo "(Run script with sudo if Docker is used and not set up for non-root users)"
echo ""
echo "Hyperparams file: ${HP_FILE}"
echo "Experiments file: ${EXP_FILE}"
echo "Activation function: ${ACTIVATION}"
echo "Dataset: ${DATASET}"
echo "Max. epochs: ${EPOCHS}"
if [ "$USE_DOCKER" = true ] ; then
    echo "Docker container image: ${DOCKER_IMAGE}"
    echo "Docker container name: ${DOCKER_NAME}"
else
    echo "Using current Python environment (not using Docker)"
fi
echo ""


# Create model results and data directory (if not exists)
mkdir -p results
mkdir -p data
mkdir -p plotting

if ! [ "$COMMAND_FILE" = false ] ; then
    cd experiments
    if [[ $NUM_THREADS -gt 1 ]] ; then
        if ! [ "$THREAD_DIR" = false ] ; then
            bash multiprocess.sh $NUM_THREADS $COMMAND_FILE $THREAD_DIR
        else
            echo "ERROR: thread_dir was NOT provided!"
        fi
    else
        while read command;
        do
            $command
        done < $COMMAND_FILE
    fi
else
    echo "no command file!"
    # if [ "$USE_DOCKER" = true ] ; then
    #     # Start Docker research container and run experiments
    #     docker run \
    #         --runtime=nvidia \
    #         -v $(pwd)/experiments:/experiments \
    #         -v $(pwd)/data:/data \
    #         -v $(pwd)/results:/results \
    #         -v $(pwd)/src:/src \
    #         -v $(pwd)/plotting:/plotting \
    #         -w /experiments \
    #         -u $(id -u):$(id -g) \
    #         -it \
    #         --rm \
    #         --name ${DOCKER_NAME} \
    #         ${DOCKER_IMAGE} \
    #         bash _run.sh ${HP_FILE} ${EXP_FILE} ${ACTIVATION} ${DATASET} ${EPOCHS} ${THREAD_ID}
    # else
    #     # Start experiments in current python environment (default, venv, or conda)
    #     cd experiments
    #     ./_run.sh ${HP_FILE} ${EXP_FILE} ${ACTIVATION} ${DATASET} ${EPOCHS} ${THREAD_ID}
    # fi
fi