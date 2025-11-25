import numpy as np
import pytest
import os
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python import utils, model

# Paths
REF_DATA_FILE = "python/tests/ref_outputs/ref_001_parsed_integers.txt"
REF_PARAM3_FILE = "python/tests/ref_outputs/ref_param3_stats.txt"
SCORE_FILE = "scores/001-1_fingering.txt"
PARAM3_FILE = "cpp/Code/param_FHMM3.txt"

def test_finger_parsing_vs_cpp_ground_truth():
    """
    FAILING TEST: Verifies that Python parses '4_1' as int(4) and handles all data
    exactly like the C++ `ConvertFingerNumberToInt`.
    """
    # 1. Load Reference Data
    ref_data = np.loadtxt(REF_DATA_FILE)
    # ref columns: index, ontime, pitch, finger_int

    # 2. Load Python Data
    notes = utils.load_pig_file(SCORE_FILE)
    notes = utils.filter_notes_by_hand(notes, 0) # RH
    notes = utils.apply_time_dep_pitch_order(notes)

    # 3. Assert Lengths
    assert len(notes) == len(ref_data)

    # 4. Assert Finger Integers
    # YOUR CODE WILL FAIL HERE:
    # 'notes' likely has 'finger_str' but missing 'finger' (int) field,
    # or 'finger' field is not cleaned (still has '4_1' or 0).
    if 'finger' not in notes.dtype.names:
        pytest.fail("Structured array missing 'finger' (int) field.")

    py_fingers = notes['finger']
    ref_fingers = ref_data[:, 3].astype(int)

    # Find the specific index where substitution occurs (original idx 7 in file)
    # In C++, 4_1 becomes 4.
    mismatches = np.where(py_fingers != ref_fingers)[0]

    if len(mismatches) > 0:
        idx = mismatches[0]
        pytest.fail(f"Finger mismatch at sorted index {idx}. "
                    f"Python: {py_fingers[idx]}, C++ Ref: {ref_fingers[idx]}")

def test_hmm_order3_loading_vs_cpp_ground_truth():
    """
    FAILING TEST: Verifies that HMMParameters detects Order 3 and loads
    the specific values matching C++ dump.
    """
    # 1. Load Reference Values
    ref_vals = {}
    with open(REF_PARAM3_FILE) as f:
        for line in f:
            k, v = line.split()
            ref_vals[k] = float(v)

    # 2. Load Python Model
    # YOUR CODE WILL FAIL HERE: currently hardcoded to Order 2
    params = model.HMMParameters(PARAM3_FILE)

    # 3. Assert Order Detection
    assert getattr(params, 'order', None) == 3, "Failed to detect Order 3"

    # 4. Assert Value Match (Log vs Linear check)
    # Python loads as Log, C++ dump is Linear.
    # TR3_SAMPLE: Hand 0, 0,1,2 -> 3
    py_val = np.exp(params.log_transition3_prob[0, 0, 1, 2, 3])
    cpp_val = ref_vals["TR3_SAMPLE"]

    assert np.isclose(py_val, cpp_val, atol=1e-6), \
        f"Order 3 Transition mismatch. Py: {py_val}, C++: {cpp_val}"

    # OUT3_SAMPLE: Hand 0, prev3=0, curr=0, dx=0, dy=0 (idx 46)
    py_out = np.exp(params.log_output3_prob[0, 0, 0, 46])
    cpp_out = ref_vals["OUT3_SAMPLE"]

    assert np.isclose(py_out, cpp_out, atol=1e-6), \
        f"Order 3 Output mismatch. Py: {py_out}, C++: {cpp_out}"
