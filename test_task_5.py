from soft_position_hmm.core import compute_inertia_cost
# Scenario: Impossible jump (distance 100), short time
cost = compute_inertia_cost(100.0, 0.1, 10.0, 0.2, 1.0)
print(f"Capped Cost: {cost}")
if cost > 8.0001:
    raise ValueError("Inertia cost was not capped!")
print("TASK 5 SUCCESS")
