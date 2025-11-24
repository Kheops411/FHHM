import unittest
import numpy as np
from python.utils import PITCH_TO_KEYPOS_LUT, pitch_to_keypos, subtract_keypos

class TestLattice(unittest.TestCase):

    def test_specific_notes(self):
        """
        Tests the coordinates of specific notes as mentioned in the C++ comments.
        """
        # C4 (60) -> (0,0)
        self.assertEqual(tuple(PITCH_TO_KEYPOS_LUT[60]), (0, 0))
        # D4 (62) -> (1,0)
        self.assertEqual(tuple(PITCH_TO_KEYPOS_LUT[62]), (1, 0))
        # Eb4 (63) -> (1,1)
        self.assertEqual(tuple(PITCH_TO_KEYPOS_LUT[63]), (1, 1))

    def test_octave_shifts(self):
        """
        Tests octave shifts, specifically C5.
        """
        # C5 (72) -> (7,0)
        self.assertEqual(tuple(PITCH_TO_KEYPOS_LUT[72]), (7, 0))

    def test_lut_vs_direct_function(self):
        """
        Ensures the LUT matches the output of the direct function call for all pitches.
        """
        for i in range(128):
            with self.subTest(pitch=i):
                expected = pitch_to_keypos(i)
                self.assertEqual(tuple(PITCH_TO_KEYPOS_LUT[i]), expected)

    def test_subtract_keypos(self):
        """
        Tests the key position subtraction function.
        """
        kp1 = (5, 2)
        kp2 = (3, 1)
        self.assertEqual(subtract_keypos(kp1, kp2), (2, 1))
        self.assertEqual(subtract_keypos(kp2, kp1), (-2, -1))
        self.assertEqual(subtract_keypos(kp1, kp1), (0, 0))

if __name__ == '__main__':
    unittest.main()
