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
            fingers.append(int(parts[-1].split('_')[0]))
    return np.array(fingers, dtype=np.int32)

# Gold Standard Test
def test_viterbi_end_to_end_match():
    # 1. Load Data
    test_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Code', 'param_FHMM2.txt'))
    score_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'scores', '001-1_fingering.txt'))
    ref_file = os.path.realpath(os.path.join(test_dir, 'ref_outputs', 'ref_001.txt'))

    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)
    ref_fingers_all = load_reference_fingering(ref_file)


    # 2. Process
    rh_notes = utils.filter_notes_by_hand(notes, 0)
    ordered_notes = utils.apply_time_dep_pitch_order(rh_notes)

    # 3. Run Viterbi
    py_fingers = model.viterbi_2nd_order_numba(
        ordered_notes,
        params.log_initial_prob,
        params.log_transition1_prob,
        params.log_transition2_prob,
        params.log_output1_prob,
        params.log_output2_prob,
        utils.PITCH_TO_KEYPOS_LUT,
        hand=0,
        w1=0.5, w2=0.5, short_time_cost=-5.0
    )

    # 4. Compare
    ref_rh_fingers = ref_fingers_all[ref_fingers_all > 0]
    assert np.array_equal(py_fingers, ref_rh_fingers)
