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
            # Handles cases like '4_1' by taking the first part
            fingers.append(int(parts[-1].split('_')[0]))
    return np.array(fingers, dtype=np.int32)

# Gold Standard Test for Right Hand
def test_viterbi_end_to_end_rh():
    """
    Tests the full pipeline for the right hand against the C++ reference output.
    This uses the main `run_viterbi` dispatcher function.
    """
    # 1. Setup Paths
    test_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Code', 'param_FHMM2.txt'))
    score_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'scores', '001-1_fingering.txt'))
    ref_file = os.path.realpath(os.path.join(test_dir, 'ref_outputs', 'ref_001.txt'))

    # 2. Load Data
    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)
    ref_fingers_all = load_reference_fingering(ref_file)

    # 3. Process Right Hand Notes
    rh_notes = notes[notes['finger'] > 0]
    ordered_notes = utils.apply_time_dep_pitch_order(rh_notes)

    # 4. Run Viterbi using the main dispatcher function
    py_fingers = model.run_viterbi(ordered_notes, params, hand=0)

    # 5. Compare
    ref_rh_fingers = ref_fingers_all[ref_fingers_all > 0]
    assert np.array_equal(py_fingers, ref_rh_fingers), "Right hand fingering sequence mismatch"

# Gold Standard Test for Left Hand
def test_viterbi_end_to_end_lh():
    """
    Tests the full pipeline for the left hand, accounting for the sign difference
    in the Python implementation's output.
    """
    # 1. Setup Paths
    test_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Code', 'param_FHMM2.txt'))
    score_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'scores', '001-1_fingering.txt'))
    ref_file = os.path.realpath(os.path.join(test_dir, 'ref_outputs', 'ref_001.txt'))

    # 2. Load Data
    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)
    ref_fingers_all = load_reference_fingering(ref_file)

    # 3. Process Left Hand Notes
    lh_notes = notes[notes['finger'] < 0]
    ordered_notes = utils.apply_time_dep_pitch_order(lh_notes)

    # 4. Run Viterbi using the main dispatcher function
    py_fingers = model.run_viterbi(ordered_notes, params, hand=1)

    # 5. Compare
    ref_lh_fingers = ref_fingers_all[ref_fingers_all < 0]

    # AUDIT FINDING: C++ outputs negative fingers, Python outputs positive.
    # We must compare the absolute values.
    assert np.array_equal(py_fingers, np.abs(ref_lh_fingers)), "Left hand fingering sequence mismatch"
