import numpy as np
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python import utils, model

def main():
    """
    Runs the Viterbi algorithm with debug prints.
    """
    param_file = 'cpp/Code/param_FHMM2.txt'
    score_file = 'scores/031-1_fingering.txt'

    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)
    rh_notes = utils.filter_notes_by_hand(notes, 0)
    ordered_notes = utils.apply_time_dep_pitch_order(rh_notes)

    print("--- RIGHT HAND ---")
    model.run_viterbi(ordered_notes, params, hand=0)

if __name__ == '__main__':
    main()
