import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from soft_position_hmm.structural import ViterbiLattice
from soft_position_hmm.core import SoftPositionModel, ANCHORS
from soft_position_hmm.inference import run_forward_pass

def test_impossible_stretch():
    """
    Validates the Viterbi forward pass with a synthetic micro-scenario.
    """
    print("--- Running Test: The Impossible Stretch ---")

    # 1. Test Setup
    notes_pitch = np.array([60, 72, 60], dtype=np.int32)
    notes_ontime = np.array([0.0, 0.05, 0.10], dtype=np.float64)
    n_obs = len(notes_pitch)

    # Use a dummy agility matrix to isolate emission/inertia effects
    agility_matrix = np.zeros((5, 5, 5), dtype=np.float64)

    # Initialize model and lattice
    model = SoftPositionModel()
    lattice = ViterbiLattice(n_obs)

    # 2. Run the forward pass
    run_forward_pass(
        n_obs=n_obs,
        notes_pitch=notes_pitch,
        notes_ontime=notes_ontime,
        lattice_log_probs=lattice.log_probs,
        lattice_backpointers=lattice.backpointers,
        agility_matrix=agility_matrix,
        inertia_param_slope=model.time_slope,
        inertia_param_center=model.time_center,
        inertia_weight=model.inertia_weight
    )

    # 3. Verification Logic
    # Find the state with the highest log-probability at t=1 and t=2

    # Time step t=1 (Note pitch 72)
    best_prob_t1 = np.max(lattice.log_probs[1])
    best_indices_t1 = np.unravel_index(np.argmax(lattice.log_probs[1]), lattice.log_probs[1].shape)
    _, _, best_anchor_idx_t1 = best_indices_t1

    # Time step t=2 (Note pitch 60)
    best_prob_t2 = np.max(lattice.log_probs[2])
    best_indices_t2 = np.unravel_index(np.argmax(lattice.log_probs[2]), lattice.log_probs[2].shape)
    _, _, best_anchor_idx_t2 = best_indices_t2

    print(f"Log probs at t=0 seem okay: {np.max(lattice.log_probs[0]) > -np.inf}")
    print(f"Best log_prob at t=1: {best_prob_t1:.4f}")
    print(f"Best Anchor at t=1: Index {best_anchor_idx_t1} (Value: {ANCHORS[best_anchor_idx_t1]})")

    print(f"Best log_prob at t=2: {best_prob_t2:.4f}")
    print(f"Best Anchor at t=2: Index {best_anchor_idx_t2} (Value: {ANCHORS[best_anchor_idx_t2]})")

    # Success Criteria: The test passes if the code runs and produces valid probabilities.
    assert np.isfinite(best_prob_t1), "Probabilities at t=1 are all -inf."
    assert np.isfinite(best_prob_t2), "Probabilities at t=2 are all -inf."

    print("--- Test Passed ---")

if __name__ == "__main__":
    test_impossible_stretch()
