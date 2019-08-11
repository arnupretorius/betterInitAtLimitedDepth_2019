# use python to find all paths to model states that are not needed for the sensitivity test
python find_unneeded_sensitivity.py

TEST_LIST_PATH="temp/compress_list.txt"

IFS=$'\r' GLOBIGNORE='*' command eval 'lines=($(cat $TEST_LIST_PATH))'
lines=( $lines )

paths=$(
    for index in ${!lines[@]}
    do
        echo ${lines[index]}
    done
)

# echo $paths
# for path in $paths
# do
#     echo $path
# done

tar czf unneeded_sensitivity_model_states.tar.gz $paths --remove-files --verbose
