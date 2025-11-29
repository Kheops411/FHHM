import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from soft_position_hmm.training import SoftPositionTrainer

# Helper to convert MIDI pitch to a SITCH string (simplified)
def midi_to_sitch(pitch):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    note_name = note_names[pitch % 12]
    return f"{note_name}{octave}"

def generate_synthetic_data(file_path="tests/synthetic_data.pig"):
    """
    Generates a synthetic PIG file for testing the trainer.
    - Thumb (1) plays notes to the left of the hand.
    - Pinky (5) plays notes to the right of the hand.
    """
    with open(file_path, "w") as f:
        # Note Original_idx, Ontime, Offtime, Pitch, On_velocity, Off_velocity, Channel, Finger
        # Thumb playing low notes
        f.write(f"1\t0.1\t0.2\t{midi_to_sitch(60)}\t100\t64\t0\t1\n")
        f.write(f"2\t0.3\t0.4\t{midi_to_sitch(58)}\t100\t64\t0\t1\n")
        f.write(f"3\t0.5\t0.6\t{midi_to_sitch(59)}\t100\t64\t0\t1\n")
        # Pinky playing high notes
        f.write(f"4\t0.7\t0.8\t{midi_to_sitch(72)}\t100\t64\t0\t5\n")
        f.write(f"5\t0.9\t1.0\t{midi_to_sitch(74)}\t100\t64\t0\t5\n")
        f.write(f"6\t1.1\t1.2\t{midi_to_sitch(73)}\t100\t64\t0\t5\n")
    return file_path

def test_training_convergence():
    """
    Tests if the SoftPositionTrainer learns from synthetic data.
    """
    print("--- Running Test: Training Convergence ---")

    # 1. Generate synthetic data
    data_path = generate_synthetic_data()

    # 2. Run Trainer
    trainer = SoftPositionTrainer()

    # Store initial mu values for comparison
    initial_mu_thumb = trainer.model.rbf_mu[0]
    initial_mu_pinky = trainer.model.rbf_mu[4]

    print(f"Initial mu for thumb: {initial_mu_thumb:.4f}")
    print(f"Initial mu for pinky: {initial_mu_pinky:.4f}")

    trainer.train([data_path], n_iterations=5)

    # 3. Check Convergence
    final_mu_thumb = trainer.model.rbf_mu[0]
    final_mu_pinky = trainer.model.rbf_mu[4]

    print(f"Final mu for thumb: {final_mu_thumb:.4f}")
    print(f"Final mu for pinky: {final_mu_pinky:.4f}")

    # Check that parameters have changed from their initial values
    assert final_mu_thumb != initial_mu_thumb, "Thumb's rbf_mu did not change."
    assert final_mu_pinky != initial_mu_pinky, "Pinky's rbf_mu did not change."

    # Thumb's ideal position should be to the left (negative delta)
    assert final_mu_thumb < initial_mu_thumb, "Thumb's rbf_mu should become more negative."
    # Pinky's ideal position should be to the right (positive delta)
    assert final_mu_pinky > initial_mu_pinky, "Pinky's rbf_mu should become more positive."

    # Clean up the synthetic data file
    os.remove(data_path)

    print("--- Test Passed ---")

if __name__ == "__main__":
    test_training_convergence()
