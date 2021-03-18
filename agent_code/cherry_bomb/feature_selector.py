import numpy as np
import pickle
import os
import time
import argparse

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

from agent_code.training_data.train_data_utils import read_train_data, read_h5f

IN_PATH = "../training_data"
OUT_PATH = "../transformers"


def select_features(selection_algorithm, training_set, transform_file, threshold=0.8):
    """
    Possible values for selection_algorithm: PCA, Kernel_PCA and Clustering
    Function updates the transformer variable which contains a fitted feature selection algo/object
    """

    # load data from .npy files
    # take states and next_states as input variables for the algorithms to increase the amount of data
    # samples row-wise, features column-wise
    # TODO: read only states
    rewards, actions, states, next_states = read_h5f(os.path.join(IN_PATH, f"{training_set}.h5"), "coin_collect_data")
    X = standardize(states)

    # selection algorithm
    if selection_algorithm == "PCA":
        # note: PCA does not support sparse matrices - use TruncatedSVD or SparsePCA in that case
        # linear separation
        transformer = PCA(n_components=threshold)

    elif selection_algorithm == "Kernel_PCA":
        # non-linear separation
        initial_transformer = KernelPCA(kernel='poly')
        variance_ratio_sum = initial_transformer.fit(X).explained_variance_ratio_.cumsum()
        n_components = np.argwhere(variance_ratio_sum > threshold)[0]
        transformer = KernelPCA(n_components=n_components, kernel='poly')

    # TODO:
    # elif selection_algorithm == "Clustering":
    # transformer = cluster.FeatureAgglomeration(n_clusters=n_components)

    # store transformer in file
    with open(os.path.join(OUT_PATH, f"{transform_file}.pt"), "wb") as file:
        pickle.dump(transformer.fit(X), file)


def standardize(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)


# create parser
parser = argparse.ArgumentParser()

# add arguments to the parser
parser.add_argument("--method", default="PCA", help="Choose Feature Selection method")
parser.add_argument("--train", type=str, default="default", help="Name of Training file")
parser.add_argument("--tform", type=str, default="default", help="Name of Transformer file")
parser.add_argument("--thold", type=float, default=None, help="Threshold for Feature Selection")

# parse the arguments
args = parser.parse_args()

# set the arguments value
method = args.method
train_file = args.train
tform_file = args.tform
thold = args.thold

# execute feature selection script
print(f'Starting feature selection with args [{method}, {train_file}, {tform_file}, {thold}]')
select_features(method, train_file, tform_file, thold)
print(f'Finished feature selection')
