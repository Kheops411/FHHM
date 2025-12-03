import numpy as np

def evaluate(score, fingering):
    """
    Compares the generated fingering with the ground truth from the score
    and calculates the accuracy.

    Args:
        score: The ScoreData object containing the ground truth.
        fingering: The numpy array of generated fingering.

    Returns:
        A float representing the accuracy (percentage of correct notes).
    """
    ground_truth = score.finger_gt

    # We only evaluate notes where a ground truth fingering is available ( > 0)
    valid_gt_mask = ground_truth > 0

    if np.sum(valid_gt_mask) == 0:
        return 100.0 # No ground truth to compare against

    correct_fingers = np.sum(fingering[valid_gt_mask] == ground_truth[valid_gt_mask])
    total_gt_notes = np.sum(valid_gt_mask)

    accuracy = (correct_fingers / total_gt_notes) * 100.0
    return accuracy
