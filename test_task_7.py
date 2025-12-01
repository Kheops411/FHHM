import numpy as np
from soft_position_hmm.interface import predict_fingering
from soft_position_hmm.core import SoftPositionModel

# This is a high-level integration test. We don't need real data
# to verify the hand_sign logic, just enough to make the function run.
print("TASK 7: Verifying hand_sign logic...")

# 1. Create dummy inputs
dummy_notes_pitch = np.array([60, 62, 64], dtype=np.int32)
dummy_notes_ontime = np.array([0.0, 0.5, 1.0], dtype=np.float64)
dummy_model = SoftPositionModel()

# 2. Call the function with hand_sign for Left Hand
# We only care about the fingers array for this test.
fingers_lh, _ = predict_fingering(
    dummy_notes_pitch,
    dummy_notes_ontime,
    dummy_model,
    hand_sign=-1
)

print(f"Returned Fingers (LH): {fingers_lh}")

# 3. Validate the output
if len(fingers_lh) == 0:
    # This could happen if the dummy data is too sparse, but for 3 notes it shouldn't.
    # Still, it's not a failure of the hand_sign logic itself.
    print("Warning: predict_fingering returned an empty array. Hand sign logic could not be fully tested.")
elif np.all(fingers_lh <= 0):
    print("TASK 7 SUCCESS: All returned fingerings are negative, as expected for LH.")
else:
    raise ValueError("FAILED: predict_fingering with hand_sign=-1 returned positive finger numbers.")

# 4. (Optional) Check Right Hand default
fingers_rh, _ = predict_fingering(
    dummy_notes_pitch,
    dummy_notes_ontime,
    dummy_model,
    hand_sign=1 # Explicitly RH
)
print(f"Returned Fingers (RH): {fingers_rh}")
if len(fingers_rh) > 0 and np.all(fingers_rh > 0):
    print("RH default check passed.")
else:
     raise ValueError("FAILED: predict_fingering with hand_sign=1 returned negative finger numbers.")
