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
            # Handle cases where fingering might be missing
            if len(parts) >= 8:
                 fingers.append(int(parts[-1].split('_')[0]))
    return np.array(fingers, dtype=np.int32)

@pytest.mark.parametrize("score_filename, ref_filename", [
    ("scores/022-1_fingering.txt", "python/tests/ref_outputs/ref_adv_022.txt"),
    ("scores/002-1_fingering.txt", "python/tests/ref_outputs/ref_adv_002.txt"),
    ("scores/126-2_fingering.txt", "python/tests/ref_outputs/ref_adv_126.txt"),
])
def test_adversarial_scores(score_filename, ref_filename):
    # 1. Load Data
    test_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Code', 'param_FHMM2.txt'))

    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_filename)
    ref_fingers_all = load_reference_fingering(ref_filename)

    # 2. Process Right Hand
    rh_notes = utils.filter_notes_by_hand(notes, 0)
    if len(rh_notes) > 2:
        ordered_rh_notes = utils.apply_time_dep_pitch_order(rh_notes)
        py_rh_fingers = model.viterbi_2nd_order_numba(
            ordered_rh_notes,
            params.log_initial_prob,
            params.log_transition1_prob,
            params.log_transition2_prob,
            params.log_output1_prob,
            params.log_output2_prob,
            utils.PITCH_TO_KEYPOS_LUT,
            hand=0,
            w1=0.5, w2=0.5, short_time_cost=-5.0
        )
        ref_rh_fingers = ref_fingers_all[ref_fingers_all > 0]
        assert np.array_equal(py_rh_fingers, ref_rh_fingers)

    # 3. Process Left Hand
    lh_notes = utils.filter_notes_by_hand(notes, 1)
    if len(lh_notes) > 2:
        ordered_lh_notes = utils.apply_time_dep_pitch_order(lh_notes)
        py_lh_fingers = model.viterbi_2nd_order_numba(
            ordered_lh_notes,
            params.log_initial_prob,
            params.log_transition1_prob,
            params.log_transition2_prob,
            params.log_output1_prob,
            params.log_output2_prob,
            utils.PITCH_TO_KEYPOS_LUT,
            hand=1,
            w1=0.5, w2=0.5, short_time_cost=-5.0
        )
        ref_lh_fingers = ref_fingers_all[ref_fingers_all < 0]
        assert np.array_equal(py_lh_fingers, ref_lh_fingers)

def test_order3_hmm_is_not_implemented():
    """
    This test confirms the critical finding that the Order-3 Viterbi logic is missing.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'cpp', 'Code', 'param_FHMM3.txt'))
    score_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'scores', '001-1_fingering.txt'))

    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)
    rh_notes = utils.filter_notes_by_hand(notes, 0)
    ordered_rh_notes = utils.apply_time_dep_pitch_order(rh_notes)

    # This test is expected to fail because viterbi_3rd_order_numba does not exist.
    with pytest.raises(AttributeError, match="module 'python.model' has no attribute 'viterbi_3rd_order_numba'"):
        model.viterbi_3rd_order_numba(
            ordered_rh_notes,
            params, # Pass params object directly or unpack as needed by hypothetical function
            utils.PITCH_TO_KEYPOS_LUT,
            hand=0,
            w1=0.667, w2=0.5, w3=0.2, short_time_cost=-5.0
        )
