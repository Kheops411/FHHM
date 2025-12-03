import numpy as np
import sys
import os

# Add src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from xml_parser import MusicXMLParser, get_valid_mask
from engine import find_fingerings

def test_left_hand_scale():
    """
    Left-Hand Scale Test: Load `resources/scale_c_major_desc.xml`.
    - Assert: The output array is valid (values 0-4).
    - Assert: A finger-over cross occurs.
    """
    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'scale_c_major_desc.xml'))
    parser = MusicXMLParser(xml_path)
    score = parser.parse()

    fingering = find_fingerings(score, side="left")

    # Filter out invalid fingerings (-1) for the assertions
    valid_fingering = fingering[fingering != -1]

    assert np.all((valid_fingering >= 0) & (valid_fingering <= 4)), "Fingering values must be between 0 and 4"

    # --- DEBUG PRINTS ---
    print("\n--- Left-Hand Scale Test Debug Info ---")
    note_names = ['C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4']
    valid_indices = np.where(fingering != -1)[0]
    print("Pitches:    ", [note_names[i] for i in range(len(valid_indices))])
    print("Fingering:  ", valid_fingering)

    from utils import generate_luts
    keypos_lut, _ = generate_luts()
    x_pos = keypos_lut[score.pitch[valid_indices]]
    print("X Positions:", -x_pos) # Negated for left hand view
    # --- END DEBUG ---

    # Check for finger-over cross: x_pos increases while finger index increases
    finger_over_found = False
    for i in range(1, len(valid_fingering)):
        # For left hand descending, pitch goes down, but x_pos becomes less negative (increases)
        if -x_pos[i] > -x_pos[i-1] and valid_fingering[i] > valid_fingering[i-1]:
             # A finger cross-over (e.g., 3 over 1)
             if valid_fingering[i-1] == 0:
                finger_over_found = True
                break
    assert finger_over_found, "A finger-over cross should occur in the descending C major scale for the left hand"
    print("Test Left-Hand Scale: Passed")

if __name__ == "__main__":
    test_left_hand_scale()
    print("\nAll Left-Hand tests passed!")
