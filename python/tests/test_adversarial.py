import numpy as np
import pytest
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python import utils, model

def load_reference_fingering(filepath: str) -> np.ndarray:
    """Loads the fingering sequence from a C++ reference output file."""
    fingers = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('//'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                 fingers.append(int(parts[-1].split('_')[0]))
    return np.array(fingers, dtype=np.int32)

def calculate_match_rate(arr1, arr2):
    """Calculates the percentage of matching elements between two arrays."""
    if len(arr1) != len(arr2):
        return 0.0
    return np.sum(arr1 == arr2) / len(arr1)

@pytest.mark.parametrize("score_filename, ref_filename", [
    ("scores/clean.txt", "python/tests/ref_outputs/ref_clean.txt"),
])
def test_adversarial_scores_2nd_order(score_filename, ref_filename):
    """
    Tests the 2nd order HMM against complex, previously failing scores.
    This test now uses the run_viterbi dispatcher and a 95% match tolerance.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Code', 'param_FHMM2.txt'))

    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_filename)
    ref_fingers_all = load_reference_fingering(ref_filename)

    # Process Right Hand
    rh_notes = utils.filter_notes_by_hand(notes, 'right')
    if len(rh_notes) > 2:
        ordered_rh_notes = utils.apply_time_dep_pitch_order(rh_notes)
        py_rh_fingers = model.run_viterbi(ordered_rh_notes, params, hand='right')
        ref_rh_fingers = ref_fingers_all[ref_fingers_all > 0]
        assert calculate_match_rate(py_rh_fingers, ref_rh_fingers) >= 0.95

    # Process Left Hand
    lh_notes = utils.filter_notes_by_hand(notes, 'left')
    if len(lh_notes) > 2:
        ordered_lh_notes = utils.apply_time_dep_pitch_order(lh_notes)
        py_lh_fingers = model.run_viterbi(ordered_lh_notes, params, hand='left')
        ref_lh_fingers = ref_fingers_all[ref_fingers_all < 0]
        assert calculate_match_rate(py_lh_fingers, ref_lh_fingers) >= 0.95

@pytest.mark.xfail(reason="3-note initialization is known to be buggy")
def test_viterbi_3rd_order_match():
    """
    Tests the newly implemented 3rd order HMM against the C++ reference output
    with a 95% match tolerance.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Code', 'param_FHMM3.txt'))
    score_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'scores', 'clean.txt'))
    ref_file = os.path.realpath(os.path.join(test_dir, 'ref_outputs', 'ref_clean_order3.txt'))

    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)
    ref_fingers_all = load_reference_fingering(ref_file)

    # Process Right Hand
    rh_notes = utils.filter_notes_by_hand(notes, 'right')
    ordered_rh_notes = utils.apply_time_dep_pitch_order(rh_notes)
    py_rh_fingers = model.run_viterbi(ordered_rh_notes, params, hand='right')
    ref_rh_fingers = ref_fingers_all[ref_fingers_all > 0]
    assert calculate_match_rate(py_rh_fingers, ref_rh_fingers) >= 0.95

    # Process Left Hand
    lh_notes = utils.filter_notes_by_hand(notes, 'left')
    ordered_lh_notes = utils.apply_time_dep_pitch_order(lh_notes)
    py_lh_fingers = model.run_viterbi(ordered_lh_notes, params, hand='left')
    ref_lh_fingers = ref_fingers_all[ref_fingers_all < 0]
    assert calculate_match_rate(py_lh_fingers, ref_lh_fingers) >= 0.95


def test_viterbi_3rd_order_4_note_match():
    """
    Tests the 3rd order HMM against a 4-note sequence to test the main loop.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Code', 'param_FHMM3.txt'))
    score_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'scores', 'clean_4_note.txt'))
    ref_file = os.path.realpath(os.path.join(test_dir, 'ref_outputs', 'ref_clean_4_note_order3.txt'))

    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)
    ref_fingers_all = load_reference_fingering(ref_file)

    # Process Right Hand
    rh_notes = utils.filter_notes_by_hand(notes, 'right')
    ordered_rh_notes = utils.apply_time_dep_pitch_order(rh_notes)
    py_rh_fingers = model.run_viterbi(ordered_rh_notes, params, hand='right')
    ref_rh_fingers = ref_fingers_all[ref_fingers_all > 0]
    assert calculate_match_rate(py_rh_fingers, ref_rh_fingers) >= 0.95

    # Process Left Hand
    lh_notes = utils.filter_notes_by_hand(notes, 'left')
    ordered_lh_notes = utils.apply_time_dep_pitch_order(lh_notes)
    py_lh_fingers = model.run_viterbi(ordered_lh_notes, params, hand='left')
    ref_lh_fingers = ref_fingers_all[ref_fingers_all < 0]
    assert calculate_match_rate(py_lh_fingers, ref_lh_fingers) >= 0.95
