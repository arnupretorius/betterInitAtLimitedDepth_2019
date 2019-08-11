num_threads=$1
command_file=$2
shared_node_info_dir=$3
shared_thread_dir=$4
num_epochs=$5
local_models_path=$6
shared_training_dir=$7
node_id=$8
threads=$( eval echo {0..$(($num_threads-1))} )

trap ctrl_c INT

function ctrl_c() {
    kill $( cat $shared_thread_dir/*/pid.log )
    exit
}

is_thread_available () {
    local thread_id=$1 && shift
    local pids=($@)
    local pid=${pids[$thread_id]}

    # check if thread already has a process id
    if [ -z "$pid" ]; then
        true
    else
        # check if this process id is running something
        if ! ps -p ${pids[$thread_id]} > /dev/null
        then
            true
        else
            false
        fi
    fi
}

get_available_thread () {
    local pids=($@)
    local thread_id=0

    # loop until an available thread is found
    while true
    do
        # check every thread to see if it is busy
        for thread_id in $threads
        do
            if is_thread_available $thread_id ${pids[@]}
            then
                echo $thread_id
                return
            fi
        done

        # if all threads are busy, wait 100ms and check again
        sleep 0.1
    done
}

in_file() {
    local file_path=$1 && shift
    local string="$@"

    if grep -Fxq "${string[@]}" "$file_path"
    then
        true
    else
        false
    fi
}

already_run_or_busy() {
    local thread_id=$1 && shift
    local command="$@"
    local thread_dir
    local thread_dirs
    local node_dir
    local node_dirs
    local node_previously_done

    # search through thread files in each node
    node_dirs=$( ls $shared_node_info_dir )
    for node_dir in $node_dirs
    do
        if [[ -d "$shared_node_info_dir/$node_dir" ]] ; then
            # check if this command was previously run by the node
            node_previously_done="$shared_node_info_dir/$node_dir/previously_completed.log"

            if in_file $node_previously_done ${command[@]}
            then
                true
                return
            fi

            # technically this program should never assign the same command to
            # two of the local threads, so I am being a bit over cautious here.
            thread_dirs=$( ls $shared_node_info_dir/$node_dir )
            for thread_dir in $thread_dirs
            do
                if [[ -d "$shared_node_info_dir/$node_dir/$thread_dir" ]] ; then
                    if ! [[ "node_$node_id" == "$node_dir" && "thread_$thread_id" == "$thread_dir" ]] ; then
                        thread_busy_file="$shared_node_info_dir/$node_dir/$thread_dir/busy.log"
                        thread_complete_file="$shared_node_info_dir/$node_dir/$thread_dir/complete.log"

                        # if the command is found either in the busy.log or the complete.log return true
                        if in_file $thread_busy_file ${command[@]} || in_file $thread_complete_file ${command[@]}
                        then
                            true
                            return
                        fi
                    fi
                fi
            done
        fi
    done

    # return false as the function will return before this if the command was found
    false
}

# create array to store process ids
pids=()

# make dir to store thread info and clear old log files
for thread_id in $threads
do
    thread_dir="$shared_thread_dir/thread_$thread_id"
    mkdir -p $thread_dir

    touch "$thread_dir/output.log"
    touch "$thread_dir/complete.log"
    touch "$thread_dir/busy.log"
    touch "$thread_dir/pid.log"

    truncate -s 0 "$thread_dir/output.log"
    truncate -s 0 "$thread_dir/complete.log"
    truncate -s 0 "$thread_dir/busy.log"
    truncate -s 0 "$thread_dir/pid.log"
done

# distribute tasks with a loop
while read command;
do
    # get an available thread
    thread_id=$( get_available_thread ${pids[@]} )

    # find the directory that this thread uses
    thread_dir="$shared_thread_dir/thread_$thread_id"

    # move the command that was previously busy to the complete log
    cat "$thread_dir/busy.log" >> "$thread_dir/complete.log"

    full_command="${command[@]} $num_epochs $local_models_path $shared_training_dir"

    # write the new command to the busy file
    echo "$full_command" > "$thread_dir/busy.log"

    # check if this command has already been run or is busy being run
    if already_run_or_busy $thread_id "${full_command[@]}"
    then
        # clear the busy file
        echo "" > "$thread_dir/busy.log"

        echo "Another node or thread has already run '$full_command'."
        echo "Moving onto next command..."
    else
        # give command to thread
        echo "executing '$full_command'..."
        $full_command &>> "$thread_dir/output.log" &

        # capture the command's process id
        pid=$!
        pids[$thread_id]=$pid
        echo $pid > "$thread_dir/pid.log"
    fi

done < $command_file

# wait for all threads to finish
for thread_id in $threads
do
    pid=${pids[$thread_id]}
    echo "waiting for $thread_id with pid $pid"
    wait $pid

    # thread no longer has a job, so clear the busy.log
    echo "" > "$thread_dir/busy.log"
done

echo "DONE!"
