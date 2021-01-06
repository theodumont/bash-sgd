# bash-sgd

## About
This script is an implementation of a simple Stochastic Gradient Descent with one neuron in bash. The main goal of this was to practice the basics of the bash language, especially arithmetic expansion, arrays, `sed` and the `bc` math tool.

## Usage

The script needs Python to convert the `.h5` dataset files to `.txt` files, readable by bash.

1. Clone the repo
   ```bash
   git clone https://github.com/theodumont/bash-sgd.git
   ```
2. Create a virtual environment, for instance using anaconda.
   ```bash
   conda create -n bash-sgd python==3.8
   conda activate bash-sgd
   ```
   The script will install the required packages for Python (`numpy` and `h5py`).
3. Run the bash script
   ```bash
   bash sgd.sh [nb_epochs] [learning_rate]
   ```

## Note
The script is in debug mode (only 2 epochs and small portion of dataset & images pixels are used).


## Acknowledgements

- I used the [deeplearning.ai](https://www.coursera.org/learn/neural-networks-deep-learning?) Cat-Non-Cat dataset that I found on [this non-official repository](https://github.com/ridhimagarg/Cat-vs-Non-cat-Deep-learning-implementation/tree/master/datasets).
