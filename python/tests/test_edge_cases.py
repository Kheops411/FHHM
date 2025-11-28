import numpy as np
import pytest
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python import utils, model

@pytest.fixture
def temp_score_file(tmp_path):
    """A fixture to create a temporary score file."""
    def _create_file(content):
        f = tmp_path / "score.txt"
        f.write_text(content)
        return str(f)
    return _create_file

def test_empty_score_file(temp_score_file):
    """Test that an empty score file is handled gracefully."""
    score_path = temp_score_file("")
    notes = utils.load_pig_file(score_path)
    assert len(notes) == 0

    param_path = 'cpp/Code/param_FHMM2.txt'
    params = model.HMMParameters(param_path)
    fingers = model.run_viterbi(notes, params)
    assert len(fingers) == 0

def test_single_note_score_file(temp_score_file):
    """Test that a score with one note is handled gracefully."""
    score_path = temp_score_file("0\t0.1\t0.2\tC4\t60\t80\t0\t1\n")
    notes = utils.load_pig_file(score_path)
    assert len(notes) == 1

    param_path = 'cpp/Code/param_FHMM2.txt'
    params = model.HMMParameters(param_path)
    fingers = model.run_viterbi(notes, params)
    assert len(fingers) == 1

def test_left_hand_only_score(temp_score_file):
    """Test a score with only left-hand notes."""
    content = ("0\t0.1\t0.2\tC3\t48\t80\t1\t-1\n"
               "1\t0.2\t0.3\tE3\t52\t80\t1\t-2\n"
               "2\t0.3\t0.4\tG3\t55\t80\t1\t-3\n")
    score_path = temp_score_file(content)
    notes = utils.load_pig_file(score_path)

    param_path = 'cpp/Code/param_FHMM2.txt'
    params = model.HMMParameters(param_path)

    rh_notes = utils.filter_notes_by_hand(notes, 0)
    assert len(rh_notes) == 0
    rh_fingers = model.run_viterbi(rh_notes, params, hand=0)
    assert len(rh_fingers) == 0

    lh_notes = utils.filter_notes_by_hand(notes, 1)
    assert len(lh_notes) == 3
    lh_fingers = model.run_viterbi(lh_notes, params, hand=1)
    assert len(lh_fingers) == 3
