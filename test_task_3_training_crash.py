import numpy as np
from soft_position_hmm.training import SoftPositionTrainer
from soft_position_hmm.utils import FINGER_UNKNOWN

def test_training_with_unknowns():
    # 1. Mock the load_pig_file to return data with unknown fingers
    # We will bypass file loading and inject data directly into the trainer logic
    # But since the trainer is monolithic, we will subclass/mock strictly for this test
    # OR simply create a dummy PIG file. Creating a dummy file is safer for you.

    filename = "dummy_test.txt"
    with open(filename, "w") as f:
        # Columns: ID Onset Offset Note Velocity OffVel Hand Finger
        # Line 1: Normal
        f.write("0 0.0 0.1 C4 64 64 0 1\n")
        # Line 2: Unknown Finger (Parser returns FINGER_UNKNOWN)
        f.write("1 0.2 0.3 D4 64 64 0 ?\n")
        # Line 3: Normal
        f.write("2 0.4 0.5 E4 64 64 0 2\n")

    trainer = SoftPositionTrainer()

    print("Attempting to train on file with unknown fingers...")
    try:
        # Train for 1 iteration to trigger the statistics collection
        trainer.train([filename], n_iterations=1)
        print("TASK 3-BIS SUCCESS: Training loop handled unknown fingers without crashing.")
    except IndexError as e:
        print(f"TASK 3-BIS FAILED: IndexError detected! {e}")
        raise
    except Exception as e:
        print(f"TASK 3-BIS FAILED: {e}")
        raise
    finally:
        import os
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_training_with_unknowns()
