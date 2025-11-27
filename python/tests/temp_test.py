import numpy as np
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python import utils, model

def run_verification():
    """
    Runs the Viterbi algorithm on a known-good score file and prints the results.
    """
    param_file = 'cpp/Code/param_FHMM2.txt'
    score_file = 'python/tests/ref_outputs/ref_adv_022.txt'

    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)

    # Process Right Hand
    rh_notes = utils.filter_notes_by_hand(notes, 'right')
    if len(rh_notes) > 2:
        ordered_rh_notes = utils.apply_time_dep_pitch_order(rh_notes)
        rh_fingers = model.run_viterbi(ordered_rh_notes, params, hand='right')
        print("Right Hand Fingers:", rh_fingers)

    # Process Left Hand
    lh_notes = utils.filter_notes_by_hand(notes, 'left')
    if len(lh_notes) > 2:
        ordered_lh_notes = utils.apply_time_dep_pitch_order(lh_notes)
        lh_fingers = model.run_viterbi(ordered_lh_notes, params, hand='left')
        print("Left Hand Fingers:", lh_fingers)

if __name__ == '__main__':
    run_verification()
