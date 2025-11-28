import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from python import model, utils

def main():
    if len(sys.argv) != 5:
        print(f"Usage: python {sys.argv[0]} <model_order> <param_file> <input_score> <output_file>")
        sys.exit(1)

    model_order = int(sys.argv[1])
    param_file = sys.argv[2]
    input_score = sys.argv[3]
    output_file = sys.argv[4]

    # Load parameters
    params = model.HMMParameters(param_file)
    if params.order != model_order:
        print(f"Warning: Model order mismatch. Expected {model_order}, but found {params.order} in {param_file}.")

    # Load and process score
    notes = utils.load_pig_file(input_score)

    # Separate notes by hand based on initial fingering
    # This replicates the C++ logic of processing hands independently
    right_hand_notes = notes[notes['channel'] == 0]
    left_hand_notes = notes[notes['channel'] == 1]

    # Run Viterbi for each hand
    if len(right_hand_notes) > 0:
        right_fingers = model.run_viterbi(right_hand_notes, params, hand=0)
        right_hand_notes['finger'] = right_fingers

    if len(left_hand_notes) > 0:
        left_fingers = model.run_viterbi(left_hand_notes, params, hand=1)
        left_hand_notes['finger'] = -left_fingers # Left hand fingers are negative

    # Combine results and restore original order
    all_notes = np.concatenate((right_hand_notes, left_hand_notes)) if len(right_hand_notes) > 0 and len(left_hand_notes) > 0 else \
                right_hand_notes if len(right_hand_notes) > 0 else left_hand_notes

    # The C++ binary writes out the notes in the original order found in the file.
    # We must sort by the original_idx to match this behavior.
    final_notes = np.sort(all_notes, order='original_idx', kind='stable')

    # Write output file
    with open(output_file, 'w') as f:
        for note in final_notes:
            finger_str = str(note['finger'])
            # This formatting aims to be close to the C++ output, but may need adjustment
            f.write(f"{note['original_idx']}\t{note['ontime']:.6f}\t{note['offtime']:.6f}\t"
                    f"{note['pitch_str']}\t{note['velocity']}\t{note['velocity']}\t" # onvel and offvel are the same in the source data
                    f"{note['channel']}\t{finger_str}\n")

if __name__ == "__main__":
    main()
