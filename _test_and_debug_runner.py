
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('./python'))

import python.model as model
import python.utils as utils

def save_output_file(filepath, original_notes, right_hand_notes, right_fingering, left_hand_notes, left_fingering):
    """Saves the fingering output in the same format as the C++ tools."""

    right_finger_map = {note['original_idx']: finger for note, finger in zip(right_hand_notes, right_fingering)}
    left_finger_map = {note['original_idx']: finger for note, finger in zip(left_hand_notes, left_fingering)}

    with open(filepath, 'w') as f:
        f.write("//Version: Python_HMM_v1_Final\n")
        sorted_notes = np.sort(original_notes, order='original_idx')
        for note in sorted_notes:
            oid = note['original_idx']
            finger = 0
            if oid in right_finger_map:
                finger = right_finger_map[oid]
            elif oid in left_finger_map:
                finger = -left_finger_map[oid]

            line = (f"{note['original_idx']}\t{note['ontime']:.6f}\t{note['offtime']:.6f}\t"
                    f"{note['pitch_str']}\t{note['velocity']}\t{note['velocity']}\t"
                    f"{note['channel']}\t{finger}\n")
            f.write(line)

def parse_cpp_output(filepath):
    """Parses the C++ output file to extract a map of {original_idx: finger}."""
    finger_map = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('//'):
                continue
            parts = line.strip().split()
            if len(parts) == 8:
                original_idx = int(parts[0])
                finger = int(parts[7])
                finger_map[original_idx] = finger
    return finger_map

def run_and_validate():
    score_file = 'scores/001-1_fingering.txt'
    param_file = 'cpp/Code/param_FHMM2.txt'
    cpp_output_file = 'output_cpp_hmm2.txt'
    python_output_file = 'output_python_final.txt'

    print("--- Running python/ version with final Viterbi fix ---")
    notes = utils.load_pig_file(score_file)
    params = model.HMMParameters(param_file)

    right_hand_notes = utils.filter_notes_by_hand(notes, 0)
    right_hand_notes = utils.apply_time_dep_pitch_order(right_hand_notes)
    fingering_right = model.run_viterbi(right_hand_notes, params, hand=0)

    left_hand_notes = utils.filter_notes_by_hand(notes, 1)
    left_hand_notes = utils.apply_time_dep_pitch_order(left_hand_notes)
    fingering_left = model.run_viterbi(left_hand_notes, params, hand=1)

    save_output_file(python_output_file, notes, right_hand_notes, fingering_right, left_hand_notes, fingering_left)
    print(f"Python output saved to {python_output_file}")

    print("\n--- Comparing Python output with C++ ground truth ---")
    cpp_fingers = parse_cpp_output(cpp_output_file)
    python_fingers = parse_cpp_output(python_output_file)

    mismatches = 0
    total_notes = len(cpp_fingers)
    for oid, cpp_finger in cpp_fingers.items():
        if oid not in python_fingers or python_fingers[oid] != cpp_finger:
            mismatches += 1

    if mismatches == 0:
        print(f"\nSuccess! All {total_notes} notes are identical to the C++ output.")
    else:
        print(f"\nFound {mismatches} mismatches out of {total_notes} notes.")

if __name__ == '__main__':
    run_and_validate()
