import numpy as np
from soft_position_hmm.training import SoftPositionTrainer

trainer = SoftPositionTrainer()
trainer.model.rbf_mu[0] = 5.0
# Simulate data that would pull mu to -5.0
deltas = [[-5.0]*10, [], [], [], []]

trainer._update_emission_parameters(deltas)
new_mu = trainer.model.rbf_mu[0]
print(f"Old Mu: 5.0, Target: -5.0, New Mu (Momentum): {new_mu}")

# Expected: 0.9*5 + 0.1*(-5) = 4.5 - 0.5 = 4.0
if not (3.5 < new_mu < 4.5):
    raise ValueError("Momentum logic is incorrect or missing.")
print("TASK 6 SUCCESS")
