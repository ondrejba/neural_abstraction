import json


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
