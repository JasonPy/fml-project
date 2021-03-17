import numpy as np
import os.path


def save_train_data(train_list, path):
    """
    Append a list of training data to a npy file.
    If file is not existent create a file.

    first col   : reward
    scnd col    : action
    rest        : features
    """
    train_np_array = np.array(train_list)

    if os.path.isfile(path):
        current_data_set = np.load(path)
        new_data_set = np.vstack((current_data_set, train_np_array))
        np.save(path, new_data_set)
    else:
        np.save(path, train_np_array)


def read_train_data(path):
    csv_as_array = np.load(path)
    rewards = csv_as_array[:, 0]
    actions = csv_as_array[:, 1]
    feature_length = csv_as_array.shape[1] - 2  # -2 because of rewards and actions
    old_state_max_index = int(feature_length/2) + 2
    old_state_features = csv_as_array[:, 2:old_state_max_index]
    new_state_features = csv_as_array[:, old_state_max_index:csv_as_array.shape[1]]

    return rewards, actions, old_state_features, new_state_features
