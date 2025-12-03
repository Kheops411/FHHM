import numpy as np
import sys
import os

# Add src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from xml_parser import MusicXMLParser, get_valid_mask

def test_get_valid_mask():
    """
    Mandatory Test (`tests/test_parser.py`):
    - Load `resources/silence_gap.xml`.
    - Assert: `score.pitch` contains `0` values (silences).
    - Assert: `get_valid_mask(score)` returns `False` at indices where pitch is `0`.
    """
    # Path to the test resource file
    xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources', 'silence_gap.xml'))

    # Load the score
    parser = MusicXMLParser(xml_path)
    score = parser.parse()

    # Assert: `score.pitch` contains `0` values (silences).
    assert 0 in score.pitch

    # Assert: `get_valid_mask(score)` returns `False` at indices where pitch is `0`.
    valid_mask = get_valid_mask(score)
    zero_pitch_indices = np.where(score.pitch == 0)[0]

    assert not np.any(valid_mask[zero_pitch_indices])

    print("All tests in test_get_valid_mask passed!")


if __name__ == "__main__":
    test_get_valid_mask()
