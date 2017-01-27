#!/bin/bash
# Your virtualenv here if used
source /home/filip/.virtualenvs/cv/bin/activate

STARTING_COMMAND="Welcome in basebuild script. It helps you to start your adventure with Semantic Image Analise Script.
                   Using this script you can build database to start recognising new images."
echo $STARTING_COMMAND
# Prepare folder paths
BASH_FILE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FOLDER_PATH="$(dirname $BASH_FILE_PATH)"
IMAGE_FOLDER_PATH="$FOLDER_PATH/data/base"
DESCRIPTORS_FOLDER_PATH="$FOLDER_PATH/data/descriptors"
BASEGEN_SCRIPT_PATH="$FOLDER_PATH/scripts/basegen.py"
INHIPOGEN_SCRIPT_PATH="$FOLDER_PATH/scripts/inhipogen.py"
SPATIAL_SCRIPT_PATH="$FOLDER_PATH/scripts/spatial_train.py"

# Run Basegen script
STIME=`date +%s`
echo "Starting basegen.py...[Path: $IMAGE_FOLDER_PATH]"
echo "$(python $BASEGEN_SCRIPT_PATH --path $IMAGE_FOLDER_PATH)"
ETIME=`date +%s`
RUNTIME=$((ETIME-STIME))
echo "Execution time: $RUNTIME seconds"

# Run inhipogen script
STIME=`date +%s`
echo "Starting inhipogen.py...[Path: $DESCRIPTORS_FOLDER_PATH]"
echo "$(python $INHIPOGEN_SCRIPT_PATH --path $DESCRIPTORS_FOLDER_PATH)"
ETIME=`date +%s`
RUNTIME=$((ETIME-STIME))
echo "Execution time: $RUNTIME seconds"

# Run spatial train script
STIME=`date +%s`
echo "Starting spatial_train.py...[Path: $DESCRIPTORS_FOLDER_PATH]"
echo "$(python $SPATIAL_SCRIPT_PATH --path $DESCRIPTORS_FOLDER_PATH)"
ETIME=`date +%s`
RUNTIME=$((ETIME-STIME))
echo "Execution time: $RUNTIME seconds"

echo "basebuild completed job."