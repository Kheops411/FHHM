import numpy as np
import sys
import os

# Add src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from eval import evaluate
from xml_parser import ScoreData

def test_evaluation():
    """
    Tests the accuracy calculation of the evaluate function.
    """
    # Create a mock ScoreData object
    score = ScoreData(n_notes=10)
    score.finger_gt = np.array([1, 2, 3, 4, 5, 1, 2, 3, 0, 0], dtype=np.int8) # Last 2 are un-fingered

    # Test case 1: Perfect match
    fingering_1 = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=np.int8)
    accuracy_1 = evaluate(score, fingering_1)
    assert np.isclose(accuracy_1, 100.0), f"Test Case 1 Failed: Expected 100.0, got {accuracy_1}"

    # Test case 2: Half match
    fingering_2 = np.array([1, 2, 3, 4, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    accuracy_2 = evaluate(score, fingering_2)
    assert np.isclose(accuracy_2, 50.0), f"Test Case 2 Failed: Expected 50.0, got {accuracy_2}"

    # Test case 3: No match
    fingering_3 = np.array([5, 4, 2, 1, 3, 5, 4, 2, 1, 3], dtype=np.int8)
    accuracy_3 = evaluate(score, fingering_3)
    assert np.isclose(accuracy_3, 0.0), f"Test Case 3 Failed: Expected 0.0, got {accuracy_3}"

    # Test case 4: No ground truth
    score_no_gt = ScoreData(n_notes=5)
    fingering_4 = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    accuracy_4 = evaluate(score_no_gt, fingering_4)
    assert np.isclose(accuracy_4, 100.0), f"Test Case 4 Failed: Expected 100.0, got {accuracy_4}"

    print("All evaluation tests passed!")

if __name__ == "__main__":
    test_evaluation()
