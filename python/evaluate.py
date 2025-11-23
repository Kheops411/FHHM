import numpy as np

def calculate_match_rate(ground_truth, estimated):
    """
    Calculates the simple match rate between two fingering sequences.

    Args:
        ground_truth (list or np.ndarray): The ground truth fingering sequence.
        estimated (list or np.ndarray): The estimated fingering sequence.

    Returns:
        float: The match rate as a percentage.
    """
    if len(ground_truth) != len(estimated):
        raise ValueError("Input sequences must have the same length.")

    matches = np.sum(np.array(ground_truth) == np.array(estimated))
    match_rate = (matches / len(ground_truth)) * 100

    return match_rate
