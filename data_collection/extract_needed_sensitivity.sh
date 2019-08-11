# find all tars that we might want to extract?
# create a file we would like to extract them to?
RESULTS_DIR="../results"
TARGET_DIR="extracted_files/"
TAR_DIR="tar_dir/"

TAR_CONTENTS_PATH="tar_contents.txt"
FILES_TO_BE_EXTRACTED_PATH="files_to_extract.txt"
MOVES_FOR_EXTRACTED_FILES_PATH="moves_for_extracted_files.txt"

tar_files=$( ls $TAR_DIR )
tar_files=$(
    for tar_file in $tar_files
    do
        echo $( realpath "$TAR_DIR/$tar_file" )
    done
)

# make the directory that we will be extracting files to
mkdir $TARGET_DIR

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

    contents=$( tar -tzf $tar_file )

    # write the contents of the tar file to a txt file that a python program can scan
    for file in $contents
    do
        echo $file
    done > $TAR_CONTENTS_PATH

    # call a python file that looks through the contents of this array
    python files_to_be_extracted.py $TARGET_DIR $TAR_CONTENTS_PATH $FILES_TO_BE_EXTRACTED_PATH $MOVES_FOR_EXTRACTED_FILES_PATH

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
        tar -xzf $tar_file -C $TARGET_DIR --verbose $files_to_extract
        echo " ####################################################################"

        echo " ########################### MOVING FILES ###########################"
        # move the extracted files to the correct location for other python scripts
        while read line;
        do
            paths=( $line )
            mkdir -p ${paths[1]}
            mv -v $line
        done < $MOVES_FOR_EXTRACTED_FILES_PATH
        echo " ####################################################################"
    fi
done

rm -r $TARGET_DIR
rm $TAR_CONTENTS_PATH
rm $FILES_TO_BE_EXTRACTED_PATH
rm $MOVES_FOR_EXTRACTED_FILES_PATH
