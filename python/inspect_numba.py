import sys
import os
import numpy as np

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python import utils, model

# We need to call the function with representative types for compilation to occur
def inspect_viterbi_types():
    """
    Triggers Numba compilation and inspects the generated types.
    """
    param_file = os.path.realpath(os.path.join('cpp', 'Code', 'param_FHMM2.txt'))
    score_file = os.path.realpath(os.path.join('scores', '001-1_fingering.txt'))

    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)
    rh_notes = utils.filter_notes_by_hand(notes, 0)
    ordered_notes = utils.apply_time_dep_pitch_order(rh_notes)

    # Trigger compilation
    model.viterbi_2nd_order_numba(
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

    # Inspect the types
    print("--- Numba JIT Compilation Report for viterbi_2nd_order_numba ---")
    model.viterbi_2nd_order_numba.inspect_types()

if __name__ == '__main__':
    inspect_viterbi_types()
