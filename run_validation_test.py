import numpy as np
import sys
import os

# Add the project sub-directory to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'soft_position_hmm'))

from soft_position_hmm.core import SoftPositionModel
from soft_position_hmm.training import SoftPositionTrainer
from soft_position_hmm.interface import predict_fingering

def create_c_major_arpeggio_data():
    """Creates a C major arpeggio, which is hard to play with one finger."""
    pitches = [60, 64, 67, 72] # C4, E4, G4, C5
    ontimes = np.arange(len(pitches)) * 0.3
    # A plausible ground truth fingering for training
    fingers = [1, 2, 3, 5]

    notes_pitch = np.array(pitches, dtype=np.int32)
    notes_ontime = np.array(ontimes, dtype=np.float64)

    # Create a structured array for the trainer
    notes_data = np.zeros(len(pitches), dtype=[('pitch', 'i4'), ('ontime', 'f8'), ('finger', 'i4')])
    notes_data['pitch'] = notes_pitch
    notes_data['ontime'] = ontimes
    notes_data['finger'] = fingers

    print("--- Generated Test Data (C Major Arpeggio) ---")
    print(f"Pitches: {notes_pitch}")
    print(f"A plausible fingering: {fingers}")
    print("-" * 20)

    return notes_pitch, notes_ontime, notes_data

if __name__ == '__main__':
    print(">>> Starting Validation Test for soft_position_hmm <<<")

    # 1. Prepare data
    notes_pitch, notes_ontime, notes_data_for_training = create_c_major_arpeggio_data()

    # 2. Setup Trainer and Model
    trainer = SoftPositionTrainer()
    mock_file_paths = ["dummy_arpeggio"]

    # Monkey-patch the loader to return our synthetic data
    def mock_load_pig_file(path): return notes_data_for_training
    def mock_apply_order(notes): return notes
    import soft_position_hmm.training as training_module
    training_module.load_pig_file = mock_load_pig_file
    training_module.apply_time_dep_pitch_order = mock_apply_order

    # 3. Run Training for one iteration
    print("\n>>> Testing Training Step (1 iteration)...")
    try:
        history = trainer.train(mock_file_paths, n_iterations=1)
        print(f"Training successful. Final Log Likelihood: {history[-1]}")
        assert np.isfinite(history[-1]), "Log Likelihood is not a finite number!"
    except Exception as e:
        print(f"!!! TRAINING FAILED: {e}")
        sys.exit(1)

    # 4. Run Prediction (Inference)
    print("\n>>> Testing Prediction Step...")
    try:
        predicted_fingers, _ = predict_fingering(
            notes_pitch=notes_pitch,
            notes_ontime=notes_ontime,
            model=trainer.model,
            agility_matrix=trainer.agility_matrix
        )
        print(f"Prediction successful.")
        print(f"Predicted Fingering: {predicted_fingers}")

        # ROBUSTNESS CHECK
        unique_fingers_used = len(set(predicted_fingers))
        print(f"Number of unique fingers used: {unique_fingers_used}")
        assert unique_fingers_used > 1, "VALIDATION FAILED: Model collapsed to a single finger!"

    except Exception as e:
        print(f"!!! PREDICTION FAILED or VALIDATION FAILED: {e}")
        sys.exit(1)

    print("\n>>> Validation Test Passed Successfully! <<<")
