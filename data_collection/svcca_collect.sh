# install the python package that is missing from the docker container
pip install tables

# call python script that:
# * scans the file system and finds all models to test
# * saves this data in a file - to simplify things, this is a list of model paths
python svcca_test_scan.py

TEST_LIST_PATH="temp/test_list.txt"

echo "All models to run tests on:"
echo "$(cat $TEST_LIST_PATH)"

# loop over and call a python script that:
# * checks which tests have already been done and skips them
while read experiment_details;
do
  python svcca_collect.py "$experiment_details"
done < $TEST_LIST_PATH