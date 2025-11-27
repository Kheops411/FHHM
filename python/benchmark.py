import timeit
import numpy as np
import os
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python import utils, model

def run_viterbi_benchmark():
    """
    Sets up and runs the Viterbi algorithm on a complex score file.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.realpath(os.path.join(test_dir, '..', 'cpp', 'Code', 'param_FHMM2.txt'))
    score_file = os.path.realpath(os.path.join(test_dir, '..', 'scores', '126-2_fingering.txt'))

    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)

    # Process both hands to simulate a full run
    rh_notes = utils.filter_notes_by_hand(notes, 0)
    ordered_rh_notes = utils.apply_time_dep_pitch_order(rh_notes)

    lh_notes = utils.filter_notes_by_hand(notes, 1)
    ordered_lh_notes = utils.apply_time_dep_pitch_order(lh_notes)

    # The main logic to be timed
    model.viterbi_2nd_order_numba(
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
    model.viterbi_2nd_order_numba(
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

if __name__ == '__main__':
    # Run the benchmark 10 times to get a stable average
    number_of_runs = 10
    total_time = timeit.timeit(run_viterbi_benchmark, number=number_of_runs)

    avg_time = total_time / number_of_runs
    print(f"Benchmarked on 'scores/126-2_fingering.txt'")
    print(f"Number of runs: {number_of_runs}")
    print(f"Average execution time: {avg_time:.4f} seconds")
