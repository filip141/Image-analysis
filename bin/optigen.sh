#!/bin/bash
# Your virtualenv here if used
source /home/filip/.virtualenvs/cv/bin/activate

# Prepare folder paths
BASH_FILE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FOLDER_PATH="$(dirname $BASH_FILE_PATH)"
OPTIGEN_SCRIPT_PATH="$FOLDER_PATH/scripts/optigen.py"

for ((i=1;i<=$#;i++));
do
    if [ ${!i} = "--input" ]
    then ((i++))
        INPUT=${!i};
    elif [ ${!i} = "--population" ];
    then ((i++))
        POPULATION=${!i};
    elif [ ${!i} = "--mutation" ];
    then ((i++))
        MUTATION=${!i};

    elif [ ${!i} = "--generations" ];
    then ((i++))
        GENERATIONS=${!i};

    elif [ ${!i} = "--crossover" ];
    then ((i++))
        CROSSOVER=${!i};

    elif [ ${!i} = "--show" ] ; then
        SHOW=true;
    elif [ ${!i} = "--save" ] ; then
        SAVE=true;
    elif [ ${!i} = "--help" ] ; then ((i++))
        echo ""
        echo "    Available options for Optigen Script from Image-Analysis package:"
        echo ""
        echo "        --input: Path to input image,"
        echo "        --population: Population parameter. How many chromosomes in population,"
        echo "        --mutation: Mutation parameter, probability for gen mutation,"
        echo "        --crossover: How many chromosomes should exchange their genes,"
        echo "        --show: Show plot with best and worst chromosome,"
        echo "        --save: Save plots to file,"
        echo "        --help: Show help message."
        echo ""
    else
        echo "Unknown option. Please repeat... :("
        exit 0
    fi
done;

# Check input
if [ -z "$INPUT" ] ; then
    echo "Input image not specified! Exiting"
    exit 0
fi
# Check input file
if [ -f "$INPUT" ]
then
	echo "$INPUT found."
else
	echo "$INPUT NOT found.Exiting..."
	exit 0
fi

# Check generations
if [ -z "$GENERATIONS" ] ; then
    GENERATIONS="200"
fi

# Check Population size
if [ -z "$POPULATION" ] ; then
    POPULATION="200"
fi

# Check crossover
if [ -z "$CROSSOVER" ] ; then
    CROSSOVER="0.7"
fi

# Check mutation
if [ -z "$MUTATION" ] ; then
    MUTATION="0.008"
fi

echo "Starting optigen.py...[Path: $INPUT]"
if [ "$SAVE" = true ] ; then
    SAVE_OPT="--save"
else
    SAVE_OPT=""
fi

if [ "$SHOW" = true ] ; then
    SHOW_OPT="--show"
else
    SHOW_OPT=""
fi

# Run program
echo "Python command: "
set -o verbose
eval "python $OPTIGEN_SCRIPT_PATH --input $INPUT --generations $GENERATIONS --population $POPULATION --crossover $CROSSOVER --mutation $MUTATION $SAVE_OPT $SHOW_OPT"