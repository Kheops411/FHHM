import numpy as np
import numba as nb
from .core import compute_emission_score, compute_inertia_cost, ANCHORS
from .structural import N_FINGERS, N_ANCHORS

@nb.njit(cache=True)
def run_forward_pass(
    n_obs: int,
    notes_pitch: np.ndarray, # Int32 array of pitches
    notes_ontime: np.ndarray, # Float64 array of onsets
    lattice_log_probs: np.ndarray,
    lattice_backpointers: np.ndarray,
    agility_matrix: np.ndarray, # Shape (5, 5, 5) -> log probs
    inertia_param_slope: float,
    inertia_param_center: float,
    inertia_weight: float
):
    """
    Fills the lattice_log_probs and lattice_backpointers in-place.
    """

    # A. Initialization (t = 0)
    for f_curr in range(N_FINGERS):
        for k_curr in range(N_ANCHORS):
            delta_pitch = -ANCHORS[k_curr]
            emit = compute_emission_score(delta_pitch, f_curr)
            # Broadcast to all f_prev states
            for f_prev in range(N_FINGERS):
                lattice_log_probs[0, f_prev, f_curr, k_curr] = emit

    # B. Recursion (t = 1 to n_obs - 1)
    for t in range(1, n_obs):
        dt = notes_ontime[t] - notes_ontime[t-1]
        for f_curr in range(N_FINGERS):
            for k_curr in range(N_ANCHORS):
                # Emission score is constant for this target state
                delta_pitch = -ANCHORS[k_curr]
                emit = compute_emission_score(delta_pitch, f_curr)

                for f_prev in range(N_FINGERS):
                    max_prob = -np.inf
                    best_k_prev = -1
                    best_f_prev2 = -1

                    for k_prev in range(N_ANCHORS):
                        # Calculate physical distance for inertia
                        hand_pos_prev = notes_pitch[t-1] + ANCHORS[k_prev]
                        hand_pos_curr = notes_pitch[t] + ANCHORS[k_curr]
                        dist = np.abs(hand_pos_curr - hand_pos_prev)
                        inertia = compute_inertia_cost(dist, dt, inertia_param_slope, inertia_param_center, inertia_weight)

                        for f_prev2 in range(N_FINGERS):
                            prev_prob = lattice_log_probs[t-1, f_prev2, f_prev, k_prev]
                            agility = agility_matrix[f_prev2, f_prev, f_curr]

                            candidate = prev_prob + agility - inertia + emit

                            if candidate > max_prob:
                                max_prob = candidate
                                best_k_prev = k_prev
                                best_f_prev2 = f_prev2

                    # Store result
                    lattice_log_probs[t, f_prev, f_curr, k_curr] = max_prob
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 0] = best_f_prev2
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 1] = f_prev
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 2] = best_k_prev
