# bash-sgd

## About
This script is an implementation of a simple Stochastic Gradient Descent with one neuron in bash. The main goal of this was to practice the basics of the bash language, especially arithmetic expansion, arrays, `sed` and the `bc` math tool.

## Usage

The script needs Python to convert the `.h5` dataset files to `.txt` files, readable by bash.

1. Clone the repo
   ```bash
   git clone https://github.com/github_username/repo_name.git
   ```
2. Install pip packages (numpy, h5py)
   ```bash
   pip install -r requirements.txt
   ```
3. Run bash script
   ```bash
   bash sgd.sh [nb_epochs] [learning_rate]
   ```

## Acknowledgements

- I used the [deeplearning.ai](https://www.coursera.org/learn/neural-networks-deep-learning?) Cat-Non-Cat dataset that I found on [this non-official repository](https://github.com/ridhimagarg/Cat-vs-Non-cat-Deep-learning-implementation/tree/master/datasets).
