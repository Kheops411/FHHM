import numpy as np
import sys
import os
import pytest

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from python import utils

@pytest.fixture
def pig_file_content():
    return """0	0.1	0.2	C4	80	0	0	1
"""

def test_load_pig_file(pig_file_content):
    """
    Tests the load_pig_file function with a known-good PIG file.
    """
    filepath = 'test.pig'
    with open(filepath, 'w') as f:
        f.write(pig_file_content)

    notes = utils.load_pig_file(filepath)
    os.remove(filepath)

    assert len(notes) == 1
    assert notes[0]['original_idx'] == 0
    assert notes[0]['ontime'] == 0.1
    assert notes[0]['offtime'] == 0.2
    assert notes[0]['pitch_str'] == 'C4'
    assert notes[0]['pitch'] == 60
    assert notes[0]['velocity'] == 80
    assert notes[0]['channel'] == 0
    assert notes[0]['finger_str'] == '1'
    assert notes[0]['finger'] == 1
