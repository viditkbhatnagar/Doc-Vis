import json
import numpy as np

def calculate_accuracy(relevant_data, whole_data):
    """
    Calculates the accuracy of the relevant extracted data against the whole data.
    Uses JSON serialization to handle dictionary comparison.

    :param relevant_data: List of dictionaries containing the relevant extracted entities.
    :param whole_data: List of dictionaries containing all extracted entities.
    :return: Accuracy percentage as a float.
    """
    # Convert list of dictionaries to a set of serialized strings for accurate comparison
    whole_data_set = set(json.dumps(d, sort_keys=True) for d in whole_data)
    correct_identifications = sum(1 for entity in relevant_data if json.dumps(entity, sort_keys=True) in whole_data_set)

    total_relevant = len(relevant_data)
    accuracy = (correct_identifications / total_relevant * 100) if total_relevant else 0
    return accuracy

def save_data(data, filename):
    """
    Saves data to a JSON file, ensuring all data types are serializable.

    :param data: Data to be serialized and saved.
    :param filename: Path to the file where data should be saved.
    """
    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError("Object of type {o.__class__.__name__} is not JSON serializable")

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, default=convert)
