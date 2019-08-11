# read in key word arguments
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
        node_id)                                node_id=${VALUE} ;;
        shared_master_commands_file_path)       shared_master_commands_file_path=${VALUE} ;;
        shared_node_info_path)                  shared_node_info_path=${VALUE} ;;
        local_node_info_path)                   local_node_info_path=${VALUE} ;;
        shared_new_plotting_path)               shared_new_plotting_path=${VALUE} ;;
        local_old_plotting_path)                local_old_plotting_path=${VALUE} ;;
        # local_old_models_path)                  local_old_models_path=${VALUE} ;;
        # local_new_models_path)                  local_new_models_path=${VALUE} ;;
        local_models_path)                      local_models_path=${VALUE} ;;
        num_threads)                            num_threads=${VALUE} ;;
        num_epochs)                             num_epochs=${VALUE} ;;
        *)
    esac
done

# check that necessary arguments have been provided
if [[ -z $node_id ]] ; then
    echo "ERROR: missing the node_id argument!"
    exit
fi

if [[ -z $shared_master_commands_file_path ]] ; then
    echo "ERROR: missing the shared_master_commands_file_path argument!"
    exit
else
    shared_master_commands_file_path=$( realpath $shared_master_commands_file_path )
fi

if [[ -z $shared_node_info_path ]] ; then
    echo "ERROR: missing the shared_node_info_path argument!"
    exit
else
    shared_node_info_path=$( realpath $shared_node_info_path )
fi

if [[ -z $local_node_info_path ]] ; then
    echo "ERROR: missing the local_node_info_path argument!"
    exit
else
    local_node_info_path=$( realpath $local_node_info_path )
fi

if [[ -z $shared_new_plotting_path ]] ; then
    echo "ERROR: missing the shared_new_plotting_path argument!"
    exit
else
    shared_new_plotting_path=$( realpath $shared_new_plotting_path )
fi

if [[ -z $local_old_plotting_path ]] ; then
    echo "ERROR: missing the local_old_plotting_path argument!"
    exit
else
    local_old_plotting_path=$( realpath $local_old_plotting_path )
fi

# if [[ -z $local_old_models_path ]] ; then
#     echo "ERROR: missing the local_old_models_path argument!"
#     exit
# else
#     local_old_models_path=$( realpath $local_old_models_path )
# fi

# if [[ -z $local_new_models_path ]] ; then
#     echo "ERROR: missing the local_new_models_path argument!"
#     exit
# else
#     local_new_models_path=$( realpath $local_new_models_path )
# fi

if [[ -z $local_models_path ]] ; then
    echo "ERROR: missing the local_models_path argument!"
    exit
else
    local_models_path=$( realpath $local_models_path )
fi

if [[ -z $num_threads ]] ; then
    echo "WARNING: missing the num_threads argument! Setting it to 1."
    num_threads=1
fi

if [[ -z $num_epochs ]] ; then
    echo "WARNING: missing the num_epochs argument! Setting it to 500."
    num_epochs=500
fi

# print argument values
echo "############################# ARGUMENT VALUES ###########################"
echo "node_id = $node_id"
echo "shared_master_commands_file_path = $shared_master_commands_file_path"
echo "shared_node_info_path = $shared_node_info_path"
echo "local_node_info_path = $local_node_info_path"
echo "shared_new_plotting_path = $shared_new_plotting_path"
echo "local_old_plotting_path = $local_old_plotting_path"
# echo "local_old_models_path = $local_old_models_path"
# echo "local_new_models_path = $local_new_models_path"
echo "local_new_models_path = $local_models_path"
echo "num_threads = $num_threads"
echo "num_epochs = $num_epochs"
echo "############################# ARGUMENT VALUES ###########################"

# define the path that the threads of this node should log in
threads_dir="$shared_node_info_path/node_$node_id"
shared_training_dir="$shared_new_plotting_path/training"
local_old_training_path="$local_old_plotting_path/training"
shared_sensitivity_dir="$shared_new_plotting_path/sensitivity"

# make the directories (in case they don't exist)
mkdir -p $threads_dir
mkdir -p $shared_training_dir
mkdir -p $shared_sensitivity_dir
mkdir -p $local_old_training_path
mkdir -p $local_node_info_path
mkdir -p $shared_new_plotting_path
mkdir -p $local_old_plotting_path
# mkdir -p $local_old_models_path
# mkdir -p $local_new_models_path
mkdir -p $local_models_path

# record the node_id of this node
echo $node_id > "$local_node_info_path/node_id.txt"
echo $node_id > "$local_node_info_path/node_$node_id.txt"

# change to the experiments directory (this is where all the relevant scripts are)
cd experiments

# check what is already locally present on this node
# write that to node log files
# copy plotting stuff to the shared space
echo "Checking this node's previous progress..."
python check_previous_node_progress.py $num_epochs $local_old_training_path $threads_dir $shared_training_dir $local_models_path > "$threads_dir/check_previous_node_progress_output.log"

# move cursor down a few lines (tqdm doesn't do this properly)
echo
echo
echo
echo

# download / make sure both datasets are on this node
echo "Ensuring datasets are available before training starts..."
python get_datasets.py

# start the threads on the training script - give each a node_id and thread_id
echo "Starting training..."
bash multiprocess.sh $num_threads $shared_master_commands_file_path $shared_node_info_path $threads_dir $num_epochs $local_models_path $shared_training_dir $node_id

# cd ../data_collection
# # start the threads on the sensitivity analyses
# echo "Starting sensitivity analyses..."
# bash sensitivity_collect.sh --dataset=mnist --num_threads=$num_threads --training_results_dir=$shared_training_dir --model_states_dir=$local_models_path --sensitivity_dir=$shared_sensitivity_dir
# bash sensitivity_collect.sh --dataset=cifar10 --num_threads=$num_threads --training_results_dir=$shared_training_dir --model_states_dir=$local_models_path --sensitivity_dir=$shared_sensitivity_dir
