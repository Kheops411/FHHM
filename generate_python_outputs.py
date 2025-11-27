import numpy as np
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python import utils, model

def generate_python_output(param_file, score_file, output_file):
    """Generates fingering output from the Python model."""
    params = model.HMMParameters(param_file)
    notes = utils.load_pig_file(score_file)

    # Process Right Hand
    rh_notes = utils.filter_notes_by_hand(notes, 0)
    if len(rh_notes) > 2:
        ordered_rh_notes = utils.apply_time_dep_pitch_order(rh_notes)
        py_rh_fingers = model.run_viterbi(ordered_rh_notes, params, hand=0)
        # Combine with original notes to write output file
        rh_indices = np.where(notes['finger'] > 0)[0]
        for i, finger in enumerate(py_rh_fingers):
            notes[rh_indices[i]]['finger'] = finger

    # Process Left Hand
    lh_notes = utils.filter_notes_by_hand(notes, 1)
    if len(lh_notes) > 2:
        ordered_lh_notes = utils.apply_time_dep_pitch_order(lh_notes)
        py_lh_fingers = model.run_viterbi(ordered_lh_notes, params, hand=1)
        lh_indices = np.where(notes['finger'] < 0)[0]
        for i, finger in enumerate(py_lh_fingers):
            notes[lh_indices[i]]['finger'] = finger

    # Write output in a format similar to the C++ version
    with open(output_file, 'w') as f:
        f.write("//Version: Python_Output\n")
        for note in notes:
            f.write(f"{note['original_idx']}\t{note['ontime']}\t{note['offtime']}\t{note['pitch_str']}\t{note['pitch']}\t{note['velocity']}\t{note['channel']}\t{note['finger']}\n")

if __name__ == '__main__':
    score_file = 'scores/031-1_fingering.txt'

    # 2nd Order
    param_file_2 = 'cpp/Code/param_FHMM2.txt'
    output_file_2 = 'fingering_output_py_order2.txt'
    generate_python_output(param_file_2, score_file, output_file_2)

    # 3rd Order
    param_file_3 = 'cpp/Code/param_FHMM3.txt'
    output_file_3 = 'fingering_output_py_order3.txt'
    generate_python_output(param_file_3, score_file, output_file_3)
