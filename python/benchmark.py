import numpy as np
import timeit
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python import utils, model

def benchmark_viterbi():
    """
    Benchmarks the viterbi_2nd_order_numba function.
    """
    # Setup
    param_file = 'cpp/Code/param_FHMM2.txt'
    score_file = 'scores/110-1_fingering.txt'

    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)
    rh_notes = utils.filter_notes_by_hand(notes, 0)
    ordered_notes = utils.apply_time_dep_pitch_order(rh_notes)

    # The function to benchmark
    def run():
        model.run_viterbi(ordered_notes, params, hand=0)

    # Time it
    number_of_runs = 10
    total_time = timeit.timeit(run, number=number_of_runs)

    avg_time = total_time / number_of_runs
    print(f"--- Viterbi Benchmark ---")
    print(f"Score file: {score_file}")
    print(f"Number of notes: {len(ordered_notes)}")
    print(f"Number of runs: {number_of_runs}")
    print(f"Average execution time: {avg_time:.6f} seconds")

if __name__ == '__main__':
    benchmark_viterbi()
