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
    train_data = np.load(path)
    rewards = train_data[:, 0]
    actions = train_data[:, 1]
    feature_length = train_data.shape[1] - 2  # -2 because of rewards and actions
    old_state_max_index = int(feature_length/2) + 2
    old_state_features = train_data[:, 2:old_state_max_index]
    new_state_features = train_data[:, old_state_max_index:train_data.shape[1]]

    return rewards, actions, old_state_features, new_state_features
