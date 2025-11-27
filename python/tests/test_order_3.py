import numpy as np
import pytest
import sys
import os
import subprocess

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python import utils, model

def load_reference_fingering(filepath: str) -> np.ndarray:
    """Loads the fingering sequence from a C++ reference output file."""
    fingers = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('//'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    fingers.append(int(parts[-1].split('_')[0]))
    except FileNotFoundError:
        pytest.fail(f"Reference fingering file not found at {filepath}", pytrace=False)
    return np.array(fingers, dtype=np.int32)

def calculate_match_rate(py_fingers, ref_fingers):
    """Calculates the percentage of matching elements, handling sign difference."""
    if len(py_fingers) == 0 and len(ref_fingers) == 0:
        return 1.0
    if len(py_fingers) != len(ref_fingers):
        return 0.0
    return np.sum(py_fingers == np.abs(ref_fingers)) / len(py_fingers)

def test_viterbi_3rd_order_end_to_end():
    """
    Critically validates the 3rd-order HMM implementation against a C++ reference.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Code', 'param_FHMM3.txt'))
    score_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'scores', '001-1_fingering.txt'))

    ref_dir = os.path.realpath(os.path.join(test_dir, 'ref_outputs'))
    ref_file = os.path.join(ref_dir, 'ref_001_order3.txt')

    # 1. Generate C++ reference output for the 3rd-order model
    hmm_runner = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Binary', 'FingeringHMM3_Run'))
    subprocess.run([hmm_runner, param_file, score_file, ref_file, "0.667", "0.5", "0.2", "0", "0.9", "-5"], check=True)

    # 2. Load data
    params = model.HMMParameters(param_file)
    assert params.order == 3 # Verify order detection
    notes = utils.load_pig_file(score_file)
    ref_fingers_all = load_reference_fingering(ref_file)

    # 3. Test Right Hand
    rh_notes = utils.filter_notes_by_hand(notes, 0)
    if len(rh_notes) > 3: # Order 3 needs at least 4 notes
        ordered_rh_notes = utils.apply_time_dep_pitch_order(rh_notes)
        py_rh_fingers = model.run_viterbi(ordered_rh_notes, params, hand=0)
        ref_rh_fingers = ref_fingers_all[ref_fingers_all > 0]
        assert calculate_match_rate(py_rh_fingers, ref_rh_fingers) == 1.0

    # 4. Test Left Hand
    lh_notes = utils.filter_notes_by_hand(notes, 1)
    if len(lh_notes) > 3:
        ordered_lh_notes = utils.apply_time_dep_pitch_order(lh_notes)
        py_lh_fingers = model.run_viterbi(ordered_lh_notes, params, hand=1)
        ref_lh_fingers = ref_fingers_all[ref_fingers_all < 0]
        assert calculate_match_rate(py_lh_fingers, ref_lh_fingers) == 1.0

    # Clean up
    os.remove(ref_file)
