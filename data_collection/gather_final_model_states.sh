# call python script that:
# * scans the file system and finds all models to test
# * saves this data in a file - to simplify things, this is a list of model paths
python sensitivity_test_scan.py

TEST_LIST_PATH="temp/test_list.txt"

IFS=$'\r' GLOBIGNORE='*' command eval  'lines=($(cat $TEST_LIST_PATH))'
lines=( $lines )

paths=$(
    for index in ${!lines[@]}
    do
        if [[ "$(( $index % 5 ))" -eq "4" ]];
        then
            echo ${lines[index]}
        fi
    done
)

echo $paths

tar czfv sensitivity_model_states.tar.gz $paths
echo "DONE!"
