#!/bin/bash
# ==============================================================================
# This script is an implementation of a simple Stochastic Gradient Descent with
# one neuron. The main goal of this was to practice the basics of the bash
# language, especially arithmetic expansion, arrays, sed and the bc math tool.
# ==============================================================================


# we check that we are in the right directory
if [[ ! -f "sgd.sh" ]]; then
    echo "Please go into the repo directory to run the bash script."
    echo "Exiting..."
    exit 1
fi


# VARIABLES ====================================================================
# We define some training variables.

e=2.71828

echo "# Parameters"
nb_epochs=${1:-"2"}
learning_rate=${2:-"0.001"}
echo "Epochs: $nb_epochs"
echo "Learning rate: $learning_rate"
echo


# READ DATASET =================================================================
# Here, we download the .h5 files, then we generate the .txt files from them.
# Then, we load them line by line into arrays. We separate the training data
# from the test data.

echo "# Dataset"

# we download the .h5 files if not already done before
TRAIN_H5_PATH="$PWD/dataset/train_catvnoncat.h5"
TEST_H5_PATH="$PWD/dataset/test_catvnoncat.h5"
TRAIN_H5_LINK="https://github.com/ridhimagarg/Cat-vs-Non-cat-Deep-learning-implementation/raw/master/datasets/train_catvnoncat.h5"
TEST_H5_LINK="https://github.com/ridhimagarg/Cat-vs-Non-cat-Deep-learning-implementation/raw/master/datasets/test_catvnoncat.h5"
PYTHON_GENERATOR_PATH="$PWD/dataset/generate.py"


if [[ ! -f $TRAIN_H5_PATH ]]; then
    echo "Downloading file $TRAIN_H5_PATH"
    wget -O $TRAIN_H5_PATH $TRAIN_H5_LINK
    if [[ $? -ne 0 ]]; then
        echo "Download of https://github.com/ridhimagarg/Cat-vs-Non-cat-Deep-learning-implementation/blob/master/datasets/train_catvnoncat.h5 has failed."
        echo "Exiting..."
        exit 1
    fi
    echo "File $TRAIN_H5_PATH downloaded"
fi
if [[ ! -f $TEST_H5_PATH ]]; then
    echo "Downloading file $TEST_H5_PATH"
    wget -O $TEST_H5_PATH $TEST_H5_LINK
    if [[ $? -ne 0 ]]; then
        echo "Download of https://github.com/ridhimagarg/Cat-vs-Non-cat-Deep-learning-implementation/blob/master/datasets/test_catvnoncat.h5 has failed."
        echo "Exiting..."
        exit 1
    fi
    echo "File $TEST_H5_PATH downloaded"
fi

echo ".h5 files downloaded."

# find whether to use python3 or python as a command
# surely not the best way to do this at all, any help is welcome
python3 --version > /dev/null 2>&1  # don't care about output so we put it in /dev/null
if [[ $? -eq 0 ]]; then
    PYTHON_SOFT="python3"
else
    python --version > /dev/null 2>&1  # don't care about output so we put it in /dev/null
    if [[ $? -eq 0 ]]; then
        PYTHON_SOFT="python"
    else
        echo "Python is not installed. Please install Python."
        echo "Exiting..."
        exit 1
    fi
fi

# checking that the generate.py file is present
if [[ ! -f $PYTHON_GENERATOR_PATH ]]; then
    echo "File $PYTHON_GENERATOR_PATH does not exist. Are you sure you cloned the whole repo at [link to repo]?"
    echo "Exiting..."
    exit 1
fi

# launching the generate.py script to produce .txt files
echo "Generating .txt dataset from .h5 files..."
python3 $PWD/dataset/generate.py --dataset_path $PWD/dataset
if [[ $? -ne 0 ]]; then
    echo "The python script has failed somehow."
    echo "Exiting..."
    exit 1
fi
echo ".txt dataset generated."

TRAIN_SAMPLES_PATH="$PWD/dataset/train_samples.txt"
TRAIN_LABELS_PATH="$PWD/dataset/train_labels.txt"
TEST_SAMPLES_PATH="$PWD/dataset/test_samples.txt"
TEST_LABELS_PATH="$PWD/dataset/test_labels.txt"

# check existence of .txt files
# (for safety, must have been created by the script before)
for file in $TRAIN_SAMPLES_PATH $TRAIN_LABELS_PATH \
            $TEST_SAMPLES_PATH $TEST_LABELS_PATH; do
    if [[ ! -f $file ]]; then
        echo "File $file does not exist. The python script must have failed somehow."
        echo "Exiting..."
        exit 1
    fi
done

# we remove all carriage return using sed
for file in $TRAIN_SAMPLES_PATH $TRAIN_LABELS_PATH \
            $TEST_SAMPLES_PATH $TEST_LABELS_PATH; do
    tempfile="$file.$$.tmp"
    sed 's/\r$//' $file > $tempfile
    mv $tempfile $file

done

# we put train/test samples/labels into arrays
# IFS and -d are delimiters
echo -ne 'Loading dataset:  [    ]  (0%)\r'
IFS=$'\n' read -d '' -a TRAIN_SAMPLES < $TRAIN_SAMPLES_PATH
echo -ne 'Loading dataset:  [#   ]  (25%)\r'
IFS=$'\n' read -d '' -a TRAIN_LABELS < $TRAIN_LABELS_PATH
echo -ne 'Loading dataset:  [##  ]  (50%)\r'
IFS=$'\n' read -d '' -a TEST_SAMPLES < $TEST_SAMPLES_PATH
echo -ne 'Loading dataset:  [### ]  (75%)\r'
IFS=$'\n' read -d '' -a TEST_LABELS < $TEST_LABELS_PATH
echo -ne 'Loading dataset:  [####]  (100%)\r'
echo

# get length of each array
nb_train_samples=${#TRAIN_SAMPLES[*]}
nb_train_labels=${#TRAIN_LABELS[*]}
nb_test_samples=${#TEST_SAMPLES[*]}
nb_test_labels=${#TEST_LABELS[*]}

# check each sample has a label
if [[ $nb_train_samples != $nb_train_labels ]]; then
    echo "Different number of samples ($nb_train_samples) and labels ($nb_train_labels)."
    echo "Exiting..."
    exit 1
fi
if [[ $nb_test_samples != $nb_test_labels ]]; then
    echo "Different number of samples ($nb_test_samples) and labels ($nb_test_labels)."
    echo "Exiting..."
    exit 1
fi

# get number of pixels in images
nb_pixels=`echo "${TRAIN_SAMPLES[0]}" | wc -w`

echo -e "training samples: $nb_train_samples    test samples: $nb_test_samples    pixels: $nb_pixels"
echo

# INITIALIZE MODEL =============================================================
# We initialize the model with parameters W (weights) and b (bias) equal to zero.

declare -a W
for (( p = 0; p < $nb_pixels; p++ )); do
    W[$p]="0"
done
b="0"


# MAIN LOOP ====================================================================
# For each epoch, we go through every training sample of the dataset and update
# the parameters of the model according to the cost function. Then we compute
# the accuracy of the model after this epoch and print it.

echo "# Training"


# loop over epochs
for (( epoch = 0; epoch < $nb_epochs; epoch++ )); do

    # loop over train samples
    for (( s = 0; s < $nb_train_samples; s++ )); do
        echo -ne "[$[$epoch+1]/$nb_epochs] training...  (`echo "100 * $s / $nb_train_samples" | bc`%)\r"

        # TRAIN ================================================================
        # Each training step is composed of a forward pass (in which we compute
        # the activation a, then the cost) and of a backward pass (in which we
        # compute the gradient), followed by the update of the parameters.


        # GET IMAGE AND LABEL --------------------------------------------------

        # get image
        declare -a IMAGE
        SAMPLE=${TRAIN_SAMPLES[s]}
        w=0
        for NUMBER in $SAMPLE; do
            IMAGE[$w]=`echo "scale=10; $NUMBER / 365" | bc`
            (( w++ ))
        done
        # get label
        label=${TRAIN_LABELS[s]}

        # FORWARD --------------------------------------------------------------

        # compute z
        z=0
        for (( p = 0; p < $nb_pixels; p++ )); do
            z=`echo "$z + ${W[$p]} * ${IMAGE[$p]} + $b" | bc`                   # z = Wx+b
        done
        # compute a
        a=`echo "1/(1+e((-1)*$z))" | bc -l`                                     # a = sigmoid(z)
        # compute cost
        cost=`echo "- ( $label * l($a) + (1-$label) * l(1-$a) )" | bc -l`       # cost = - ( y * log(a) + (1-y) * log(1-a) )


        # BACKWARD -------------------------------------------------------------

        # compute dW
        declare -a dW
        for (( p = 0; p < $nb_pixels; p++ )); do
            dW[$p]=`echo "${IMAGE[$p]} * ( $a - $label )" | bc`                 # dW = x * (a - y)
        done
        # compute db
        db=`echo "$a - $label" | bc`                                            # db = a - y


        # UPDATE WEIGHTS -------------------------------------------------------

        # update W
        for (( p = 0; p < $nb_pixels; p++ )); do
            W[$p]=`echo "${W[$p]} - $learning_rate * ${dW[$p]}" | bc`           # W = W - lr * dW
        done
        # update b
        b=`echo "$b - $learning_rate * $db" | bc`                               # b = b - lr * db

    done


    # loop over test samples
    test_cost=0
    for (( s = 0; s < $nb_test_samples; s++ )); do
        echo -ne "[$[$epoch+1]/$nb_epochs] testing...  (`echo "100 * $s / $nb_test_samples" | bc`%)  \r"

        # TEST =================================================================
        # Each test step is composed of a forward pass (in which we compute the
        # activation a, then the cost).


        # GET IMAGE AND LABEL --------------------------------------------------

        # get image
        # declare -ai IMAGE
        SAMPLE=${TEST_SAMPLES[s]}
        w=0
        for NUMBER in $SAMPLE; do
            IMAGE[$w]=`echo "scale=10; $NUMBER / 365" | bc`
            (( w++ ))
        done
        # get label
        label=${TEST_LABELS[s]}

        # FORWARD --------------------------------------------------------------

        # compute z
        z=0
        for (( p = 0; p < $nb_pixels; p++ )); do
            z=`echo "$z + ${W[$p]} * ${IMAGE[$p]} + $b" | bc`                   # z = Wx+b
        done
        # compute a
        a=`echo "1/(1+e((-1)*$z))" | bc -l`                                     # a = sigmoid(z)
        # compute cost
        cost=`echo "- ( $label * l($a) + (1-$label) * l(1-$a) )" | bc -l`       # cost = - ( y * log(a) + (1-y) * log(1-a) )
        test_cost=`echo "$test_cost + $cost" | bc -l`

    done

    test_cost=`echo "$test_cost / $nb_test_samples" | bc -l`


    echo "[$[$epoch+1]/$nb_epochs] test cost: `echo "scale=5; $test_cost / 1" | bc`  "  # '/ 1' to use the scale

done

echo

accuracy=0
for (( s = 0; s < $nb_test_samples; s++ )); do
    echo -ne "Training finished. Final accuracy: computing...  (`echo "100 * $s / $nb_test_samples" | bc`%)  \r"

    # GET IMAGE AND LABEL --------------------------------------------------

    # get image
    SAMPLE=${TEST_SAMPLES[s]}
    w=0
    for NUMBER in $SAMPLE; do
        IMAGE[$w]=`echo "scale=10; $NUMBER / 365" | bc`
        (( w++ ))
    done
    # get label
    label=${TEST_LABELS[s]}

    # FORWARD --------------------------------------------------------------

    # compute z
    z=0
    for (( p = 0; p < $nb_pixels; p++ )); do
        z=`echo "$z + ${W[$p]} * ${IMAGE[$p]} + $b" | bc`                   # z = Wx+b
    done
    # compute a
    a=`echo "1/(1+e((-1)*$z))" | bc -l`                                     # a = sigmoid(z)
    # compute difference between a and label to decide prediction
    diff=`echo "$a - $label" | bc`
    diff=${diff#-}
    if [[ `echo "$diff" | bc` < ".5" ]]; then
        # object is well-labeled
        (( accuracy++ ))
    fi

done

accuracy=`echo "100 * $accuracy / $nb_test_samples" | bc`

echo -ne "Training finished. Final accuracy: `echo "scale=0; $accuracy / 1" | bc`%                \n"
