from soft_position_hmm.core import compute_inertia_cost

# Scenario: Two notes in a chord (dt=0.001), distance is 5 semitones
cost_chord = compute_inertia_cost(physical_distance=5.0, dt=0.001, slope=10.0, center=0.2, weight=1.0)
print(f"Cost Chord: {cost_chord}")

# Scenario: Fast scale (dt=0.1), distance 5
cost_scale = compute_inertia_cost(physical_distance=5.0, dt=0.1, slope=10.0, center=0.2, weight=1.0)
print(f"Cost Scale: {cost_scale}")

if cost_chord != 0.0:
    raise ValueError("Chords (dt < 0.03) must have 0 inertia cost.")
if cost_scale == 0.0:
    raise ValueError("Scales must have non-zero inertia.")
print("TASK 2 SUCCESS")
