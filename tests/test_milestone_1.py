import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the test can find the 'soft_position_hmm' package
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from soft_position_hmm.core import (
    SoftPositionModel,
    compute_emission_score,
    compute_inertia_cost,
    ANCHORS
)

def test_hand_shape_visualizer():
    """
    Test A: Validates the emission score function by plotting the 'hand shape'.
    """
    print("--- Running Test A: Hand Shape Visualizer ---")

    deltas = np.arange(-15, 16)

    # Finger 1 (Thumb) is index 0
    # Finger 5 (Pinky) is index 4
    thumb_scores = [compute_emission_score(d, 0) for d in deltas]
    pinky_scores = [compute_emission_score(d, 4) for d in deltas]

    plt.figure(figsize=(10, 6))
    plt.plot(deltas, thumb_scores, label='Finger 1 (Thumb)', marker='o')
    plt.plot(deltas, pinky_scores, label='Finger 5 (Pinky)', marker='x')
    plt.title('Emission Score vs. Delta Pitch (Hand Shape)')
    plt.xlabel('Delta Pitch (Note - Anchor)')
    plt.ylabel('Log Probability (Emission Score)')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plot_path = os.path.join(os.path.dirname(__file__), 'hand_shape_test.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Debug Print: Print the max value coordinate for both fingers
    thumb_max_idx = np.argmax(thumb_scores)
    pinky_max_idx = np.argmax(pinky_scores)
    print(f"Thumb max score at delta = {deltas[thumb_max_idx]}")
    print(f"Pinky max score at delta = {deltas[pinky_max_idx]}")
    print("-------------------------------------------\n")


def test_time_inertia_check():
    """
    Test B: Validates the inertia cost function by checking costs at different time deltas.
    """
    print("--- Running Test B: Time-Inertia Check ---")

    model = SoftPositionModel()

    k_prev_idx = 4 # Anchor 0
    k_curr_idx = 7 # Anchor 9

    dts = [0.05, 0.2, 1.5]

    print("DT (sec) | Stiffness (0-1) | Total Cost")
    print("---------------------------------------")

    for dt in dts:
        # Calculate stiffness separately for the table
        stiffness = 1.0 / (1.0 + np.exp(model.time_slope * (dt - model.time_center)))

        cost = compute_inertia_cost(
            k_prev_idx,
            k_curr_idx,
            dt,
            model.time_slope,
            model.time_center,
            model.inertia_weight
        )
        print(f"{dt:<8.2f} | {stiffness:<15.4f} | {cost:<10.4f}")

    print("-------------------------------------------\n")


if __name__ == "__main__":
    # This allows running the tests directly
    test_hand_shape_visualizer()
    test_time_inertia_check()
    print("All milestone 1 tests completed.")
