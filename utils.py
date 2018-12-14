import os
import json
import pickle


def load_json(path):
    """
    Load a JSON file.
    :param path:    Path to the file.
    :return:        The loaded JSON file.
    """

    with open(path, "r") as file:
        return json.load(file)


def save_json(path, data):
    """
    Save JSON into a file.
    :param path:      Where to save the JSON data.
    :param data:      The JSON data.
    :return:          None.
    """

    with open(path, "w") as file:
        json.dump(data, file, indent=4, sort_keys=True)


def read_pickle(path):
    """
    Read pickle from a file.
    :param path:    Path to pickle.
    :return:        Content of the pickle.
    """

    with open(path, "rb") as file:
        return pickle.load(file)


def write_pickle(path, data):
    """
    Write pickle to a file.
    :param path:    Path where to write the pickle.
    :param data:    Data to pickle.
    :return:        None.
    """

    with open(path, "wb") as file:
        pickle.dump(data, file)


def maybe_read_pickle(path):

    if os.path.isfile(path):
        with open(path, "rb") as file:
            return pickle.load(file)
    else:
        return {}


def maybe_makedirs(dir_path):

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
