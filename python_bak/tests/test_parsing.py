import numpy as np
import pytest
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python import utils

@pytest.fixture(scope="module")
def score_data():
    """Load the test score file once."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    score_file = os.path.realpath(os.path.join(test_dir, '..', '..', 'scores', '001-1_fingering.txt'))
    return utils.load_pig_file(score_file)

@pytest.fixture(scope="module")
def ref_ordered_data():
    """Load the C++-generated reference ordering."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ref_file = os.path.realpath(os.path.join(test_dir, 'ref_outputs', 'score_order_reference.txt'))
    # The file has columns: index, ontime, pitch
    return np.loadtxt(ref_file)

def test_load_pig_file(score_data):
    """Test basic properties of the loaded score data."""
    # The parser should find 469 notes in this file.
    assert score_data.shape == (469,)
    assert score_data.dtype.names is not None
    # Check if the first note's pitch is correctly parsed from its string "C4"
    assert score_data[0]['pitch'] == 60

def test_filter_by_hand(score_data):
    """Test if hand filtering works as expected."""
    rh_notes = utils.filter_notes_by_hand(score_data, hand=0)
    lh_notes = utils.filter_notes_by_hand(score_data, hand=1)

    # Assert that the total number of notes is correct
    assert len(rh_notes) + len(lh_notes) == len(score_data)

    # Specific counts for this file
    assert len(rh_notes) == 250
    assert len(lh_notes) == 219


def test_time_dep_pitch_order_exact_match(score_data, ref_ordered_data):
    """
    Verify that the Python implementation of the note ordering matches the
    C++ ground truth bit-for-bit.
    """
    # 1. Filter for the right hand, as done in the C++ probe
    rh_notes = utils.filter_notes_by_hand(score_data, hand=0)

    # 2. Apply the ordering logic
    py_ordered_notes = utils.apply_time_dep_pitch_order(rh_notes)

    # 3. Compare the sequence of pitches and ontimes against the reference
    assert len(py_ordered_notes) == len(ref_ordered_data), "Output length mismatch"

    for i in range(len(py_ordered_notes)):
        ref_idx, ref_ontime, ref_pitch = ref_ordered_data[i]
        py_note = py_ordered_notes[i]

        # Check that the MIDI pitch is identical
        assert py_note['pitch'] == ref_pitch, f"Pitch mismatch at index {i}"

        # Check that the ontime is identical (within a tight tolerance)
        assert np.isclose(py_note['ontime'], ref_ontime, atol=1e-6), f"Ontime mismatch at index {i}"
