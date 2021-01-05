import os
import sys
import argparse
import numpy as np
import h5py


def get_args():
    parser = argparse.ArgumentParser(description="Process dataset.")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset folder.")

    return parser.parse_args()


def process_dataset(dataset_path):
    train_dataset = h5py.File('./dataset/train_catvnoncat.h5', "r")
    train_set_x = np.array(train_dataset["train_set_x"][:])             # train set features
    train_set_y = np.array(train_dataset["train_set_y"][:])             # train set labels

    test_dataset = h5py.File('./dataset/test_catvnoncat.h5', "r")
    test_set_x = np.array(test_dataset["test_set_x"][:])                # test set features
    test_set_y = np.array(test_dataset["test_set_y"][:])                # test set labels

    classes = np.array(test_dataset["list_classes"][:])                 # list of classes

    train_set_x = train_set_x.reshape((train_set_x.shape[0], -1))
    test_set_x = test_set_x.reshape((test_set_x.shape[0], -1))

    np.savetxt(os.path.join(dataset_path, 'train_samples.txt'), train_set_x[:20,:10], fmt='%i', delimiter=" ", newline="\n")
    np.savetxt(os.path.join(dataset_path, 'test_samples.txt'), test_set_x[:20,:10], fmt='%i', delimiter=" ", newline="\n")
    np.savetxt(os.path.join(dataset_path, 'train_labels.txt'), train_set_y[:20], fmt='%i', delimiter=" ", newline="\n")
    np.savetxt(os.path.join(dataset_path, 'test_labels.txt'), test_set_y[:20], fmt='%i', delimiter=" ", newline="\n")


if __name__ == '__main__':

    if sys.version_info < (3,5,0):
        sys.stderr.write("You need python 3.5 or later to run this script\n")
        sys.exit(1)

    try:
        args = get_args()
        process_dataset(args.dataset_path)
    except:
        print('Try $python generate.py --dataset_path /path/to/dataset/folder/')
