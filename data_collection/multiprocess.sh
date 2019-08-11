num_threads=$1
command_file=$2
threads=$( eval echo {0..$(($num_threads-1))} )

trap ctrl_c INT

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

# create array to store process ids
pids=()

# create directories for thread information
temp_dir="./temp"
thread_dir="$temp_dir/threads"
logs_dir="$thread_dir/logs"
pids_dir="$thread_dir/pids"
mkdir -p $logs_dir
mkdir -p $pids_dir

function ctrl_c() {
    kill $( cat $pids_dir/*.log )
    exit
}

# clear old log files
for thread_id in $threads
do
    truncate -s 0 "$logs_dir/$thread_id.log"
    truncate -s 0 "$pids_dir/$thread_id.log"
done

# distribute tasks with a loop
while read command;
do
    # print some details for the command
    echo "executing '$command'..."

    # get an available thread
    thread_id=$( get_available_thread ${pids[@]} )

    # give command to thread
    $command &>> "$logs_dir/$thread_id.log" &

    # capture the command's process id
    pid=$!
    pids[$thread_id]=$pid
    echo $pid > "$pids_dir/$thread_id.log"
done < $command_file

# wait for all threads to finish
for thread_id in $threads
do
    pid=${pids[$thread_id]}
    echo "waiting for $thread_id with pid $pid"
    wait $pid
done

echo "DONE!"
