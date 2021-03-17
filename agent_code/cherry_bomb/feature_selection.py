import numpy as np
import pickle
from sklearn.decomposition import PCA, KernelPCA
from sklearn import cluster
from agent_code.train_data_utils import read_train_data



def feature_selection(selection_algorithm, n_components):
    """
    Possible values for selection_algorithm: PCA, Kernel_PCA and Clustering
    Function updates the transformer variable which contains a fitted feature selection algo/object
    """
    PATH = "/home/marven/Programs/FML-project/fml-project/agent_code/resources/train_data.npy"

    # load data from .np files
    # take states and next_states as input variables for the algorithms to increase the amount of data
    # samples row-wise, features column-wise
    rewards, actions, states, next_states = read_train_data(PATH)

    X = states

    # selection algorithm
    if selection_algorithm == "PCA":
        # note: PCA does not support sparse matrices - use TruncatedSVD or SparsePCA in that case
        # linear separation
        transformer = PCA(n_components=n_components)
    elif selection_algorithm == "Kernel_PCA":
        # non-linear separation
        transformer = KernelPCA(n_components=n_components, kernel='poly')
    elif selection_algorithm == "Clustering":
        transformer = cluster.FeatureAgglomeration(n_clusters=n_components)

    # store transformer in file
    with open("transformer.pt", "wb") as file:
        pickle.dump(transformer.fit(X), file)
