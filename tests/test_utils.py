import unittest
import numpy as np
import os
import sys

# Add the python directory to the path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from utils import get_midi_to_lattice_lut, parse_all_scores

class TestUtils(unittest.TestCase):

    def test_midi_to_lattice_lut(self):
        """
        Tests the MIDI-to-Lattice LUT for correctness.
        """
        lut = get_midi_to_lattice_lut()

        # Test shape
        self.assertEqual(lut.shape, (128, 2))

        # Test some known values based on C++ conversion
        # C4 (MIDI 60) should be (0,0)
        self.assertTrue(np.allclose(lut[60], [0., 0.]))
        # C#4 (MIDI 61)
        self.assertTrue(np.allclose(lut[61], [0., 1.]))
        # E4 (MIDI 64)
        self.assertTrue(np.allclose(lut[64], [2., 0.]))
        # F4 (MIDI 65)
        self.assertTrue(np.allclose(lut[65], [3., 0.]))

    def test_parse_all_scores(self):
        """
        Tests that the parser can process all .txt files in the ./scores directory
        without raising an exception.
        """
        scores_dir = './scores'
        try:
            parsed_scores = parse_all_scores(scores_dir)
            # Check that at least one score was parsed
            self.assertGreater(len(parsed_scores), 0)

            # Check the dtype of the first parsed score
            first_score = next(iter(parsed_scores.values()))

            self.assertEqual(first_score.dtype.names, ('pitch', 'onset', 'duration', 'lattice_x', 'lattice_y', 'hand', 'finger'))

        except (ValueError, IndexError) as e:
            self.fail(f"Parsing scores failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
