import numpy as np
import pickle
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from agent_code.train_data_utils import read_train_data



def feature_selection(selection_algorithm, threshold=0.8):
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
    X = standardize(X)

    # selection algorithm
    if selection_algorithm == "PCA":
        # note: PCA does not support sparse matrices - use TruncatedSVD or SparsePCA in that case
        # linear separation
        initial_transformer = PCA()
        variance_ratio_sum = initial_transformer.fit(X).explained_variance_ratio_.cumsum()
        n_components = np.argwhere(variance_ratio_sum > threshold)[0]
        transformer = PCA(n_components=n_components)

    elif selection_algorithm == "Kernel_PCA":
        # non-linear separation
        initial_transformer = KernelPCA(kernel='poly')
        variance_ratio_sum = initial_transformer.fit(X).explained_variance_ratio_.cumsum()
        n_components = np.argwhere(variance_ratio_sum > threshold)[0]
        transformer = KernelPCA(n_components=n_components, kernel='poly')

    # TODO
    # elif selection_algorithm == "Clustering":
        # transformer = cluster.FeatureAgglomeration(n_clusters=n_components)

    # store transformer in file
    with open("transformer.pt", "wb") as file:
        pickle.dump(transformer.fit(X), file)


def standardize(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)