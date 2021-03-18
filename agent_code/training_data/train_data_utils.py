import numpy as np
import os.path
import tables


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


def save_to_h5_file(h5f_path, data_name, data, n_cols):
    """
    Writes a numpy matrix into a h5file / earray.
    If the h5file contains an earray with given data_name the data is appended

    h5f_path    : String, containing the path of the file
    data_name   : String, the name of the data
    data        : np array
    """
    h5f_array = None

    h5file = tables.open_file(h5f_path, mode='a')
    if not h5file.__contains__('/data'):
        group = h5file.create_group(h5file.root, 'data', 'Contains train data')

    for group in h5file.walk_groups("/"):
        if group.__contains__(data_name):
            for array in h5file.list_nodes(group):
                if array.name == data_name:
                    h5f_array = array
    if h5f_array is None:
        a = tables.FloatAtom()

        h5f_array = h5file.create_earray(group, data_name, a, (0, n_cols),
                                         "TEST")

    h5f_array.append(data)
    h5file.flush()
    h5file.close


def read_h5f(h5f_path, data_name):
    h5file = tables.open_file(h5f_path, mode='r')
    data = get_entry_from_h5f(h5file, data_name)
    if data is not None:
        data = np.array(data)
        return split_matrix(data)

    h5file.close()

    return data


def read_rows_h5f(h5f_path, data_name, row_indices):
    h5file = tables.open_file(h5f_path, mode='r')
    data = get_entry_from_h5f(h5file, data_name)
    if data is not None:
        data = np.array(data[row_indices, :])
        return split_matrix(data)
    h5file.close()

    return data


def get_entry_from_h5f(h5file, data_name):
    entry = None

    for group in h5file.walk_groups("/"):
        if group.__contains__(data_name):
            for array in h5file.list_nodes(group):
                if array.name == data_name:
                    entry = array

    return entry


def read_train_data(path):
    train_data = np.load(path)
    rewards = train_data[:, 0]
    actions = train_data[:, 1]
    feature_length = train_data.shape[1] - 2  # -2 because of rewards and actions
    old_state_max_index = int(feature_length / 2) + 2
    old_state_features = train_data[:, 2:old_state_max_index]
    new_state_features = train_data[:, old_state_max_index:train_data.shape[1]]

    return rewards, actions, old_state_features, new_state_features


def split_matrix(train_data):
    rewards = train_data[:, 0]
    actions = train_data[:, 1]
    feature_length = train_data.shape[1] - 2  # -2 because of rewards and actions
    old_state_max_index = int(feature_length / 2) + 2
    old_state_features = train_data[:, 2:old_state_max_index]
    new_state_features = train_data[:, old_state_max_index:train_data.shape[1]]

    return rewards, actions, old_state_features, new_state_features
