import numpy as np
import sys
import os

# Add src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from xml_parser import MusicXMLParser
from engine import preprocess_data

def test_segmentation():
    """
    Mandatory Test (`tests/test_segmentation.py`):
    1. Case Tie: Load `resources/ties_no_gap.xml`.
        - Assert: Returns 1 segment.
    2. Case Silence: Load `resources/silence_gap.xml`.
        - Assert: Returns 2 segments.
    3. Case Chord: Load `resources/chord_c_major.xml`.
        - Assert: All notes have the same `event_id`.
    """

    # --- Case Tie ---
    tie_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'ties_no_gap.xml'))
    tie_parser = MusicXMLParser(tie_path)
    tie_score = tie_parser.parse()
    tie_segments = preprocess_data(tie_score)
    assert len(tie_segments) == 1, f"Expected 1 segment for ties_no_gap.xml, but got {len(tie_segments)}"
    print("Test Case Tie: Passed")

    # --- Case Silence ---
    silence_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'silence_gap.xml'))
    silence_parser = MusicXMLParser(silence_path)
    silence_score = silence_parser.parse()
    silence_segments = preprocess_data(silence_score)
    assert len(silence_segments) == 2, f"Expected 2 segments for silence_gap.xml, but got {len(silence_segments)}"
    print("Test Case Silence: Passed")

    # --- Case Chord ---
    chord_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'chord_c_major.xml'))
    chord_parser = MusicXMLParser(chord_path)
    chord_score = chord_parser.parse()
    chord_segments = preprocess_data(chord_score)
    assert len(chord_segments) == 1, "Expected 1 segment for chord_c_major.xml"
    chord_ids = chord_segments[0]['chord_id']
    assert len(np.unique(chord_ids)) == 1, "All notes in the chord should have the same chord_id"
    print("Test Case Chord: Passed")

    print("\nAll tests in test_segmentation passed!")


if __name__ == "__main__":
    test_segmentation()
