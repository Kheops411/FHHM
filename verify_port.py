import numpy as np
import os
import sys
import argparse

# Add the python directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

from python.utils import parse_fingering_file
from python.model import HMMModel, viterbi_decode
from python.evaluate import calculate_match_rate

def run_python_viterbi(score_filepath, params_filepath):
    """
    Runs the Python Viterbi decoder on a given score file.
    """
    # Load the model and parameters
    model = HMMModel(w1=0.5, w2=0.5, lam1=0, short_time_cost=-5)
    model.load_cpp_parameters(params_filepath)

    # Parse the score file
    notes_data = parse_fingering_file(score_filepath)

    # Separate notes by hand
    rh_notes_indices = np.where(notes_data['hand'] == 0)[0]
    lh_notes_indices = np.where(notes_data['hand'] == 1)[0]

    rh_notes = notes_data[rh_notes_indices]
    lh_notes = notes_data[lh_notes_indices]

    # --- Run Viterbi for each hand ---
    rh_fingering, lh_fingering = [], []

    # Right Hand
    if len(rh_notes) > 0:
        rh_fingering = viterbi_decode(
            rh_notes,
            model.transition_matrix_2nd[0],
            model.transition_matrix_1st[0],
            model.initial_probabilities[0],
            model.output_prob_1st[0],
            model.output_prob_2nd[0],
            model.widthX, model.w1, model.w2, model.short_time_cost, 0
        )

    # Left Hand
    if len(lh_notes) > 0:
        lh_fingering = viterbi_decode(
            lh_notes,
            model.transition_matrix_2nd[1],
            model.transition_matrix_1st[1],
            model.initial_probabilities[1],
            model.output_prob_1st[1],
            model.output_prob_2nd[1],
            model.widthX, model.w1, model.w2, model.short_time_cost, 1
        )

    # Combine the results
    estimated_fingering = np.zeros(len(notes_data), dtype=int)
    if len(rh_fingering) > 0:
        estimated_fingering[rh_notes_indices] = rh_fingering
    if len(lh_fingering) > 0:
        estimated_fingering[lh_notes_indices] = lh_fingering

    return estimated_fingering

def read_cpp_output(filepath):
    """
    Reads the fingering sequence from a C++ output file.
    """
    fingering = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('//'):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                finger_str = parts[7].split('_')[0]
                finger = abs(int(finger_str))
                fingering.append(finger)
    return fingering


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify Python HMM port against C++ ground truth.")
    parser.add_argument('--score_file', type=str, default='./scores/001-1_fingering.txt',
                        help='Path to the input score file.')
    parser.add_argument('--params_file', type=str, default='./cpp/param_FHMM2.txt',
                        help='Path to the HMM parameters file.')
    parser.add_argument('--ground_truth_file', type=str, default='./cpp_ground_truth.txt',
                        help='Path to the C++ ground truth output file.')
    args = parser.parse_args()

    # --- Run Python Implementation ---
    print(f"Running Python Viterbi on {args.score_file}...")
    python_output = run_python_viterbi(args.score_file, args.params_file)
    print("Python Viterbi finished.")

    # --- Compare with C++ Ground Truth ---
    if not os.path.exists(args.ground_truth_file):
        print(f"Ground truth file not found: {args.ground_truth_file}")
        print("Please run the C++ code first and save its output.")
    else:
        print(f"Loading C++ ground truth from {args.ground_truth_file}...")
        cpp_output = read_cpp_output(args.ground_truth_file)

        if len(python_output) != len(cpp_output):
            print("\n--- Mismatch in sequence length! ---")
            print(f"Python output length: {len(python_output)}")
            print(f"C++ output length:    {len(cpp_output)}")
        else:
            match_rate = calculate_match_rate(cpp_output, python_output)
            print(f"\nMatch Rate: {match_rate:.2f}%")

            if match_rate == 100.0:
                print("Success! Python and C++ outputs are identical.")
            else:
                print("Outputs do not match. See comparison below:")
                print("Note | C++ | Python")
                print("-------------------")
                for i, (cpp_f, py_f) in enumerate(zip(cpp_output, python_output)):
                    if cpp_f != py_f:
                        print(f"{i:4d} | {cpp_f:3d} | {py_f:3d}  <-- Mismatch")
