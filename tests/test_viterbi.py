import numpy as np
import sys
import os

# Add src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from xml_parser import MusicXMLParser, get_valid_mask
from engine import find_fingerings

def run_viterbi_on_file(xml_path):
    """Helper function to run the full pipeline on an XML file."""
    parser = MusicXMLParser(xml_path)
    score = parser.parse()
    fingering = find_fingerings(score)

    valid_mask = get_valid_mask(score)
    return fingering, score

def test_viterbi_scale():
    """
    Scale Test: Load `resources/scale_c_major.xml`.
    - Assert: The output array is valid (values 0-4).
    - Assert: A thumb cross occurs.
    """
    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'scale_c_major.xml'))
    fingering, score = run_viterbi_on_file(xml_path)

    assert np.all((fingering >= 0) & (fingering <= 4)), "Fingering values must be between 0 and 4"

    # Check for thumb cross: pitch increases while finger index decreases
    thumb_cross_found = False
    valid_mask = get_valid_mask(score)
    pitches = score.pitch[valid_mask]
    for i in range(1, len(fingering)):
        if pitches[i] > pitches[i-1] and fingering[i] < fingering[i-1]:
            if fingering[i] == 0: # Thumb
                thumb_cross_found = True
                break
    assert thumb_cross_found, "A thumb cross should occur in the C major scale"
    print("Test Viterbi Scale: Passed")


def test_viterbi_chord():
    """
    Chord Test: Load `resources/chord_c_major.xml`.
    - Assert: The output corresponding to the chord contains unique values.
    """
    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'chord_c_major.xml'))
    fingering, _ = run_viterbi_on_file(xml_path)

    assert len(np.unique(fingering)) == len(fingering), "Chord fingering must be unique"
    print("Test Viterbi Chord: Passed")


def test_viterbi_black_keys():
    """
    Black Key Test: Load `resources/chromatic_black_keys.xml`.
    - Assert: Finger 0 (Thumb) is not used for these black keys.
    """
    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'chromatic_black_keys.xml'))
    fingering, _ = run_viterbi_on_file(xml_path)

    assert 0 not in fingering, "Thumb (0) should not be used for black keys in this test case"
    print("Test Viterbi Black Keys: Passed")


if __name__ == "__main__":
    test_viterbi_scale()
    test_viterbi_chord()
    test_viterbi_black_keys()
    print("\nAll Viterbi tests passed!")
