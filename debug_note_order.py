import numpy as np
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python import utils

def main():
    """
    Loads a score, filters for the right hand, applies the ordering,
    and prints the ontime and pitch for each note.
    """
    score_file = "scores/031-1_fingering.txt"
    notes = utils.load_pig_file(score_file)
    rh_notes = utils.filter_notes_by_hand(notes, 0)
    ordered_notes = utils.apply_time_dep_pitch_order(rh_notes)

    print("--- Python Note Order ---")
    for i, note in enumerate(ordered_notes):
        print(f"{i} {note['ontime']:.6f} {note['pitch']}")

if __name__ == '__main__':
    main()
