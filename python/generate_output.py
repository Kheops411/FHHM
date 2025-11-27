import numpy as np
import sys
import os
import argparse

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python import utils, model

def generate_fingering_output(score_file: str, param_file: str, output_file: str):
    """
    Generates a fingering output file for a given score using the HMM model.
    """
    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)

    notes_to_combine = []

    # Process Right Hand
    rh_notes = utils.filter_notes_by_hand(notes, 'right')
    if len(rh_notes) > 2:
        ordered_rh_notes = utils.apply_time_dep_pitch_order(rh_notes)
        rh_fingers = model.run_viterbi(ordered_rh_notes, params, hand='right')

        # Create a map from original_idx to finger
        rh_finger_map = {note['original_idx']: finger for note, finger in zip(ordered_rh_notes, rh_fingers)}

        # Apply the fingerings back to the unsorted rh_notes array
        for i in range(len(rh_notes)):
            original_idx = rh_notes[i]['original_idx']
            if original_idx in rh_finger_map:
                rh_notes[i]['finger'] = rh_finger_map[original_idx]

    if len(rh_notes) > 0:
        notes_to_combine.append(rh_notes)

    # Process Left Hand
    lh_notes = utils.filter_notes_by_hand(notes, 'left')
    if len(lh_notes) > 2:
        ordered_lh_notes = utils.apply_time_dep_pitch_order(lh_notes)
        lh_fingers = model.run_viterbi(ordered_lh_notes, params, hand='left')

        # Create a map from original_idx to finger
        lh_finger_map = {note['original_idx']: finger for note, finger in zip(ordered_lh_notes, lh_fingers)}

        # Apply the fingerings back to the unsorted lh_notes array
        for i in range(len(lh_notes)):
            original_idx = lh_notes[i]['original_idx']
            if original_idx in lh_finger_map:
                lh_notes[i]['finger'] = lh_finger_map[original_idx]

    if len(lh_notes) > 0:
        notes_to_combine.append(lh_notes)

    # Combine and write output
    if not notes_to_combine:
        print("No notes to process.")
        return

    all_notes = np.concatenate(notes_to_combine) if len(notes_to_combine) > 1 else notes_to_combine[0]
    # Sort by original index to preserve the original order
    all_notes = all_notes[all_notes['original_idx'].argsort()]

    with open(output_file, 'w') as f:
        f.write("//Version: Python_Output\n")
        for note in all_notes:
            f.write(f"{note['original_idx']}\t{note['ontime']}\t{note['offtime']}\t{note['pitch_str']}\t{note['pitch']}\t{note['velocity']}\t0\t{note['finger']}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate fingering output for a score file.')
    parser.add_argument('score_file', type=str, help='Path to the input score file (.txt)')
    parser.add_argument('param_file', type=str, help='Path to the HMM parameter file (.txt)')
    parser.add_argument('output_file', type=str, help='Path to the output fingering file (.txt)')
    args = parser.parse_args()

    generate_fingering_output(args.score_file, args.param_file, args.output_file)
    print(f"Fingering output generated at: {args.output_file}")
