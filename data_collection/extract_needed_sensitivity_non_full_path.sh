CURR_DIR=$( pwd )
PYTHON_SCRIPT=$( realpath extract_needed_sensitivity_non_full_path.py )
CHECK_DIR="$1"

cd $CHECK_DIR

TAR_CONTENTS_PATH="tar_contents.txt"
FILES_TO_BE_EXTRACTED_PATH="files_to_extract.txt"

tar_files=$( ls . | grep ".tar")

echo " ########################### LIST OF TAR FILES ###########################"
# echo $tar_files
for tar_file in $tar_files
do
    echo $tar_file
done
echo " #########################################################################"

# list contents of compressed files
for tar_file in $tar_files
do
    echo " >>> Checking $tar_file"

    contents=$( tar -tf $tar_file )

    # write the contents of the tar file to a txt file that a python program can scan
    for file in $contents
    do
        echo $file
    done > $TAR_CONTENTS_PATH

    # call a python file that looks through the contents of this array
    python $PYTHON_SCRIPT "$( pwd )" $TAR_CONTENTS_PATH $FILES_TO_BE_EXTRACTED_PATH

    # print files that are to be extracted
    files_to_extract=$( cat $FILES_TO_BE_EXTRACTED_PATH )

    if [ -n "$files_to_extract" ];
    then
        echo " ####################### FILES TO BE EXTRACTED #######################"
        for file in $files_to_extract
        do
            echo $file
        done
        echo " #####################################################################"

        echo " ############################ EXTRACTING ############################"
        # extract the files
        tar -xf $tar_file -C . --verbose $files_to_extract
        echo " ####################################################################"
    fi
done

rm $TAR_CONTENTS_PATH
rm $FILES_TO_BE_EXTRACTED_PATH

cd $CURR_DIR
