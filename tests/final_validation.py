import numpy as np
import time
import os
import sys
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from soft_position_hmm.core import SoftPositionModel, ANCHORS, compute_inertia_cost
from soft_position_hmm.inference import run_forward_pass, backtracking
from soft_position_hmm.training import SoftPositionTrainer
from soft_position_hmm.interface import predict_fingering
from xml_parser import MusicXMLParser, Hand

# Helper from milestone 4 to generate test data
def midi_to_sitch(pitch):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    note_name = note_names[pitch % 12]
    return f"{note_name}{octave}"

def generate_synthetic_data(file_path="tests/synthetic_data.pig"):
    with open(file_path, "w") as f:
        f.write(f"1\t0.1\t0.2\t{midi_to_sitch(60)}\t100\t64\t0\t1\n")
        f.write(f"2\t0.3\t0.4\t{midi_to_sitch(58)}\t100\t64\t0\t1\n")
        f.write(f"3\t0.5\t0.6\t{midi_to_sitch(59)}\t100\t64\t0\t1\n")
        f.write(f"4\t0.7\t0.8\t{midi_to_sitch(72)}\t100\t64\t0\t5\n")
        f.write(f"5\t0.9\t1.0\t{midi_to_sitch(74)}\t100\t64\t0\t5\n")
        f.write(f"6\t1.1\t1.2\t{midi_to_sitch(73)}\t100\t64\t0\t5\n")
    return file_path

class FinalValidationTests(unittest.TestCase):

    def test_1_performance_stress_test(self):
        print("\n--- 1. Performance Stress Test ---")
        n_obs = 2000
        notes_pitch = np.random.randint(40, 80, n_obs, dtype=np.int32)
        notes_ontime = np.linspace(0, 200, n_obs, dtype=np.float64)

        model = SoftPositionModel()
        agility_matrix = np.zeros((5, 5, 5), dtype=np.float64)

        # Warm-up Numba compilation
        predict_fingering(notes_pitch[:10], notes_ontime[:10], model, agility_matrix)

        start_time = time.time()
        fingers, anchors = predict_fingering(notes_pitch, notes_ontime, model, agility_matrix)
        end_time = time.time()

        duration = end_time - start_time
        print(f"Execution time: {duration:.2f}s")

        self.assertLess(duration, 3.0, "Performance test failed: Execution took too long.")
        self.assertFalse(np.isinf(fingers).any(), "Result contains inf.")
        self.assertFalse(np.isnan(fingers).any(), "Result contains NaN.")
        print("[PASS] Performance")

    def test_2_biomechanics_sanity_check(self):
        print("\n--- 2. Biomechanics Sanity Check ---")
        model = SoftPositionModel()
        self.assertTrue(np.all(model.rbf_sigma > 0), "Sigma must be positive.")

        cost_zero = compute_inertia_cost(0, 0.1, model.time_slope, model.time_center, model.inertia_weight)
        cost_octave = compute_inertia_cost(12, 0.1, model.time_slope, model.time_center, model.inertia_weight)

        self.assertGreater(cost_octave, cost_zero, "Octave jump should have higher inertia cost.")
        print("[PASS] Biomechanics")

    def test_3_musical_logic_trills(self):
        print("\n--- 3. Musical Logic: Trills ---")
        notes_pitch = np.array([60, 62, 60, 62, 60, 62], dtype=np.int32)
        notes_ontime = np.arange(0, 0.6, 0.1, dtype=np.float64)
        model = SoftPositionModel()

        # Add a penalty for using the same finger twice
        agility_matrix = np.zeros((5, 5, 5), dtype=np.float64)
        for i in range(5):
            agility_matrix[:, i, i] = -1.0 # Log-probability penalty

        fingers, anchors = predict_fingering(notes_pitch, notes_ontime, model, agility_matrix=agility_matrix, smoothing_weight=1.0)

        # Assert fingers alternate
        self.assertNotEqual(fingers[0], fingers[1], "Fingers should alternate in a trill.")
        self.assertEqual(fingers[0], fingers[2], "Fingers should follow a pattern (e.g., 1-2-1-2).")

        # Assert anchor stability
        self.assertEqual(anchors[0], anchors[1], "Anchor should be stable during a fast trill.")
        self.assertTrue(np.all(anchors == anchors[0]), "Anchor should not move during the trill.")
        print("[PASS] Trills")

    def test_4_musical_logic_chords(self):
        print("\n--- 4. Musical Logic: Chords ---")
        notes_pitch = np.array([60, 64], dtype=np.int32)
        notes_ontime = np.array([0.0, 0.0], dtype=np.float64) # Same onset time
        model = SoftPositionModel()

        fingers, _ = predict_fingering(notes_pitch, notes_ontime, model)

        self.assertNotEqual(fingers[0], fingers[1], "Different fingers must be used for a chord.")
        self.assertLess(fingers[0], fingers[1], "Lower note should use a thumb-side finger.")
        print("[PASS] Chords")

    def test_5_training_convergence(self):
        print("\n--- 5. Training Convergence ---")
        data_path = generate_synthetic_data()
        trainer = SoftPositionTrainer()

        # Capture log likelihoods (returned by train)
        # Suppress prints to keep test output clean
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            history = trainer.train([data_path], n_iterations=3)
        finally:
            sys.stdout.close() # Close the null stream
            sys.stdout = original_stdout

        print(f"Likelihood History: {history}")

        self.assertEqual(len(history), 3)

        # Check Monotonicity: Likelihood should NOT decrease significantly
        # (Small fluctuations can happen due to float precision, but trend should be up)
        improvement = history[-1] - history[0]
        self.assertGreaterEqual(improvement, -1e-5, "Model diverged (Likelihood decreased).")

        print("[PASS] Training Convergence")

    def test_6_edge_case_impossible_jump(self):
        print("\n--- 6. Edge Case: Impossible Jump ---")
        notes_pitch = np.array([60, 100], dtype=np.int32)
        notes_ontime = np.array([0.0, 0.01], dtype=np.float64)
        model = SoftPositionModel()

        fingers, anchors = predict_fingering(notes_pitch, notes_ontime, model)

        self.assertEqual(len(fingers), 2)
        self.assertFalse(np.all(fingers == -1), "Returned path should be valid.")
        print("[PASS] Impossible Jump")

    def test_7_full_pipeline_integration(self):
        print("\n--- 7. Full Pipeline Integration ---")
        xml_path = "Prélude No. 1 en C Majeur-Piano.xml"
        parser = MusicXMLParser(xml_path)
        all_notes = parser.parse()
        lh_notes = [n for n in all_notes if n.hand == Hand.LEFT and n.pitch is not None]

        pitches = np.array([n.pitch for n in lh_notes], dtype=np.int32)
        ontimes = np.array([n.onset_seconds for n in lh_notes], dtype=np.float64)

        model = SoftPositionModel()
        fingers, _ = predict_fingering(pitches, ontimes, model)

        self.assertEqual(len(fingers), len(lh_notes), "Output length must match input length.")
        print("[PASS] Full Pipeline Integration")

def run_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(FinalValidationTests))
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)

    print("\nFINAL VALIDATION REPORT")
    print("-----------------------")
    for i, test in enumerate(suite):
        status = "[PASS]" if result.wasSuccessful() else "[FAIL]"
        print(f"{status} {i+1}. {test.shortDescription()}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
