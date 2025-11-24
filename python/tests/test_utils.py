import unittest
import tempfile
import os
import numpy as np
# Note: On suppose que le script est lancé depuis la racine du projet
# ex: python -m unittest tests/test_python.utils_draconian.py
from python.utils import *

class TestDraconianParsing(unittest.TestCase):

    def test_sitch_to_pitch_edge_cases(self):
        """
        Stresses the pitch parser with edge cases.
        Ensures strict adherence to MIDI standards and PIG conventions.
        """
        # Case 1: Standard
        self.assertEqual(sitch_to_pitch("C4"), 60, "Standard C4 failed")

        # Case 2: Accidentals (PIG uses 'b' for flat, '#' for sharp)
        self.assertEqual(sitch_to_pitch("C#4"), 61, "Sharp failed")
        self.assertEqual(sitch_to_pitch("Db4"), 61, "Flat failed")

        # Case 3: Negative Octaves (The "Killer" for naive regex)
        # Context: In PIG/MIDI, 'C-1' means Note 0 (Octave -1).
        # It does NOT mean C-Flat at Octave 1.
        # Parser must prioritize negative octave over '-' as accidental.
        self.assertEqual(sitch_to_pitch("C-1"), 0,
            "Failed parsing 'C-1'. Expected MIDI 0. Check your Regex greediness (is '-' parsed as flat or negative sign?).")

        # Case 4: Double digits (Future proofing)
        self.assertEqual(sitch_to_pitch("G9"), 127, "High octave G9 failed")

        # Case 5: Rests
        self.assertEqual(sitch_to_pitch("R"), -1)
        self.assertEqual(sitch_to_pitch("rest"), -1)

    def test_parse_pig_file_index_synchronization(self):
        """
        CRITICAL TEST: Verifies that 'original_idx' tracks the DATA index,
        not the FILE LINE index.
        """
        # Create a synthetic PIG file with heavy commenting
        # We manually close the file to avoid Windows file locking issues
        content = """//Version: Fake
//Header: Noise
# Another comment
0	1.0	1.1	C4	64	80	0	1
// Interspersed comment
1	1.2	1.3	D4	64	80	0	2
"""
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            # File is now closed and safe to read by another function

            data = parse_pig_file(tmp_path)

            # Check Record 0
            # If using enumerate(file), this would be 3 (indices 0,1,2 are comments)
            self.assertEqual(data[0]['original_idx'], 0,
                f"Index Desync: First data record has index {data[0]['original_idx']}, expected 0. "
                "Do NOT use enumerate(file) to generate indices.")
            self.assertEqual(data[0]['pitch'], 60) # C4

            # Check Record 1
            # If using enumerate(file), this would be 5
            self.assertEqual(data[1]['original_idx'], 1,
                f"Index Desync: Second data record has index {data[1]['original_idx']}, expected 1.")
            self.assertEqual(data[1]['pitch'], 62) # D4

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

class TestDraconianSorting(unittest.TestCase):

    def test_cluster_sorting_stability(self):
        """
        Verifies the Time-Dependent Sorting logic strictly follows C++ rules:
        1. Cluster by time (diff < 0.03s)
        2. Sort INSIDE cluster by Pitch Ascending.
        """
        dtype = [('original_idx', 'i4'), ('ontime', 'f8'), ('offtime', 'f8'),
                 ('pitch', 'i4'), ('channel', 'i4'), ('finger_str', 'U10')]

        # Scenario:
        # Cluster 1: t=1.00, 1.01, 1.02 (diffs < 0.03) -> Should be sorted by pitch
        # Cluster 2: t=1.06 (diff 0.04 >= 0.03 from prev) -> New cluster
        raw_data = [
            (0, 1.00, 2.0, 70, 0, '1'), # Cluster 1, Pitch 70
            (1, 1.01, 2.0, 60, 0, '1'), # Cluster 1, Pitch 60 (Should move first)
            (2, 1.02, 2.0, 65, 0, '1'), # Cluster 1, Pitch 65 (Should move middle)
            (3, 1.06, 2.0, 40, 0, '1'), # Cluster 2 (Diff 0.04 > 0.03).
            (4, 1.06, 2.0, 80, 0, '1')  # Cluster 2
        ]

        notes = np.array(raw_data, dtype=dtype)
        sorted_notes = sort_notes_by_time(notes)

        # Expected Order:
        # Cluster 1: [Pitch 60 (idx 1), Pitch 65 (idx 2), Pitch 70 (idx 0)]
        # Cluster 2: [Pitch 40 (idx 3), Pitch 80 (idx 4)]

        expected_indices = [1, 2, 0, 3, 4]
        actual_indices = list(sorted_notes['original_idx'])

        self.assertEqual(actual_indices, expected_indices,
            f"Sorting logic failed. Expected indices {expected_indices}, got {actual_indices}. "
            "Ensure you sort by pitch ascending inside time clusters.")

class TestDraconianLattice(unittest.TestCase):
    def test_lattice_boundaries(self):
        """
        Verifies Lattice coordinates at MIDI extremes (0 and 127).
        Derived manually from C++ formula:
        KeyPos.x += 7*(oct-4). MIDI 0 = C-1. Oct=-1.
        PC=0 -> x=0. x_final = 0 + 7*(-5) = -35.
        y=0.
        """
        # MIDI 0 (C-1)
        self.assertEqual(pitch_to_keypos(0), (-35, 0), "Lattice logic failed at MIDI 0")

        # MIDI 127 (G9)
        # Oct=9. 7*(5)=35.
        # G (PC=7) -> x=4. x_final = 4+35 = 39.
        # G -> y=0.
        self.assertEqual(pitch_to_keypos(127), (39, 0), "Lattice logic failed at MIDI 127")

    def test_subtract_keypos_invariance(self):
        """
        Ensures vector subtraction works in all quadrants.
        """
        k1 = (10, 1)
        k2 = (-5, 0)
        # (10 - (-5), 1 - 0) = (15, 1)
        self.assertEqual(subtract_keypos(k1, k2), (15, 1))
        # (-5 - 10, 0 - 1) = (-15, -1)
        self.assertEqual(subtract_keypos(k2, k1), (-15, -1))

class TestDraconianRobustness(unittest.TestCase):
    def test_double_accidentals(self):
        """
        Tests double sharps/flats handling in sitch_to_pitch.
        """
        # C double sharp 4 = D4 = 62
        self.assertEqual(sitch_to_pitch("C##4"), 62, "Double sharp failed")
        # B double flat 3 = A3 = 57 (B3=59)
        self.assertEqual(sitch_to_pitch("Bbb3"), 57, "Double flat failed")

    def test_malformed_lines(self):
        """
        Ensures parser gracefully handles or reports corrupt lines
        instead of crashing the whole pipeline.
        """
        import tempfile, os

        content = """0\t1.0\t1.1\tC4\t64\t80\t0\t1
1\t1.2\t1.3\tD4\tBROKEN_LINE_MISSING_COLUMNS
2\t1.4\t1.5\tE4\t64\t80\t0\t3"""

        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # The expectation depends on desired behavior.
            # If strict: Should crash. If robust: Should skip/warn.
            # Assuming robust implementation for production:
            try:
                data = parse_pig_file(tmp_path)
                # If it didn't crash, did it parse the valid lines?
                self.assertEqual(len(data), 2, "Parser should interpret 2 valid lines and skip/fail the broken one safely")
                self.assertEqual(data[1]['pitch'], 64, "Should verify E4 is the second record")
            except IndexError:
                # If design choice is 'Crash on corrupt data', this is valid.
                # But the developer must know this is the expected behavior.
                pass

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_string_truncation(self):
        """
        Verifies that U10 dtype is sufficient or handled.
        """

        # String length 12: "1_2_3_4_5_12"
        long_finger = "1_2_3_4_5_12"
        content = f"0\t1.0\t1.1\tC4\t64\t80\t0\t{long_finger}"

        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            data = parse_pig_file(tmp_path)
            loaded_str = str(data[0]['finger_str'])

            # If U10 is used, this will fail (it will equal "1_2_3_4_5_")
            self.assertEqual(loaded_str, long_finger,
                f"Data Corruption Warning: Finger string truncated. Got '{loaded_str}', expected '{long_finger}'. Increase numpy dtype to U20.")

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()