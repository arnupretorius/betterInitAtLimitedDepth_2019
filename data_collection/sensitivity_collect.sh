# call python script that:
# * scans the file system and finds all models to test
# * saves this data in a file - to simplify things, this is a list of model paths

TEMP_DIR_NAME="temp"
DATA_PATH="$( realpath ../data )"
# MODEL_STATES_DIR=$( realpath "../results" )
# TRAINING_RESULTS_DIR=$( realpath "../plotting/training" )

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
                training_results_dir|training_results_dir=*)
                    val=${OPTARG#training_results_dir}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the training_results_dir argument!" >&2 && print_help_and_exit=true
                    else
                        TRAINING_RESULTS_DIR=$val
                    fi
                    ;;
                model_states_dir|model_states_dir=*)
                    val=${OPTARG#model_states_dir}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the model_states_dir argument!" >&2 && print_help_and_exit=true
                    else
                        MODEL_STATES_DIR=$val
                    fi
                    ;;
                sensitivity_dir|sensitivity_dir=*)
                    val=${OPTARG#sensitivity_dir}
                    val=${val#*=}
                    opt=${OPTARG%=$val}
                    if [ -z "${val}" ]; then
                        echo "Option --${opt} is missing the sensitivity_dir argument!" >&2 && print_help_and_exit=true
                    else
                        SENSITIVITY_DIR=$val
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
    echo "Usage: sensitivity_collect.sh [OPTIONS]"
    echo ""
    echo "Run a sensitivity analyses on the present model states."
    echo ""
    echo "Options:"
    echo "        --dataset       string      Dataset to use in experiments (Default: mnist)."
    echo "        --num_threads   int         Number of threads to use to perform tests (Default: 1)."
    echo "    -h, --help                      Print this information and exit."
    echo ""
    exit 1
fi

# Set dataset (default: mnist)
DATASET=${DATASET:-mnist}
# Set num_threads (default: 1)
NUM_THREADS=${NUM_THREADS:-1}

if [[ -z $MODEL_STATES_DIR ]] ; then
    echo "ERROR: missing the model_states_dir argument!"
    exit
fi

if [[ -z $TRAINING_RESULTS_DIR ]] ; then
    echo "ERROR: missing the training_results_dir argument!"
    exit
fi

if [[ -z $SENSITIVITY_DIR ]] ; then
    echo "ERROR: missing the sensitivity_dir argument!"
    exit
fi

mkdir -p "$TEMP_DIR_NAME"
TEST_LIST_PATH="$TEMP_DIR_NAME/test_list.txt"
COMMAND_LIST_PATH="$TEMP_DIR_NAME/command_list.txt"

touch $TEST_LIST_PATH
truncate -s 0 $TEST_LIST_PATH

touch $COMMAND_LIST_PATH
truncate -s 0 $COMMAND_LIST_PATH

python sensitivity_test_scan.py "$( realpath $TEST_LIST_PATH )" "$MODEL_STATES_DIR/$DATASET"

echo "All models to run tests on:"
echo "$(cat $TEST_LIST_PATH)"

# generate the sensitivity test data
python gen_sensitivity_data.py "$DATASET" "$DATA_PATH"

if [[ $NUM_THREADS -gt 1 ]] ;
then
  while read experiment_details;
  do
    echo "python sensitivity_collect.py $DATASET ${experiment_details[@]} $TRAINING_RESULTS_DIR $DATA_PATH $SENSITIVITY_DIR" >> $COMMAND_LIST_PATH
  done < $TEST_LIST_PATH

  IFS=$'\n' shuffled_commands=$( cat $COMMAND_LIST_PATH | shuf )

  truncate -s 0 $COMMAND_LIST_PATH
  for command in $shuffled_commands
  do
    echo $command >> $COMMAND_LIST_PATH
  done

  bash multiprocess.sh $NUM_THREADS $COMMAND_LIST_PATH
else
  while read experiment_details;
  do
    python sensitivity_collect.py $DATASET ${experiment_details[@]} $TRAINING_RESULTS_DIR $DATA_PATH $SENSITIVITY_DIR
  done < $TEST_LIST_PATH
fi

################################################################################
# Delete temp dir?
################################################################################
