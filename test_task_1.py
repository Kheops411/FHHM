import numpy as np
from soft_position_hmm.training import SoftPositionTrainer

trainer = SoftPositionTrainer()
# Check initialization
print(f"Init Mean: {np.mean(trainer.agility_matrix)}")
assert np.all(trainer.agility_matrix > 0), "Matrix must be positive before log"

# Check Update Logic
counts = np.zeros((5,5,5)) # Empty counts
trainer._update_agility_parameters(counts)
print(f"Log Agility Max: {np.max(trainer.agility_matrix)}")
print(f"Log Agility Min: {np.min(trainer.agility_matrix)}")

# FAILURE CONDITION: If min is -inf or nan.
if not np.isfinite(trainer.agility_matrix).all():
    raise ValueError("Agility matrix contains Inf or NaN!")
print("TASK 1 SUCCESS")
