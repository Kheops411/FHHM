import numpy as np
import numba as nb
from .core import compute_emission_score, compute_inertia_cost, ANCHORS
from .structural import N_FINGERS, N_ANCHORS
from .utils import PITCH_TO_KEYPOS_LUT, FINGER_UNKNOWN

@nb.njit(cache=True)
def _clip(val, min_val, max_val):
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val

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
    inertia_weight: float,
    rbf_mu: np.ndarray,
    rbf_sigma: np.ndarray,
    smoothing_weight: float
):
    """
    Fills the lattice_log_probs and lattice_backpointers in-place.
    """

    # A. Initialization (t = 0)
    for f_curr in range(N_FINGERS):
        for k_curr in range(N_ANCHORS):
            note_coord_x = PITCH_TO_KEYPOS_LUT[notes_pitch[0]][0]
            hand_pos_center_x = note_coord_x + ANCHORS[k_curr]
            emit = compute_emission_score(
                note_coord_x,
                hand_pos_center_x,
                f_curr,
                rbf_mu,
                rbf_sigma
            )
            # Broadcast to all f_prev states
            for f_prev in range(N_FINGERS):
                lattice_log_probs[0, f_prev, f_curr, k_curr] = emit

    # B. Recursion (t = 1 to n_obs - 1)
    for t in range(1, n_obs):
        dt = notes_ontime[t] - notes_ontime[t-1]
        for f_curr in range(N_FINGERS):
            for k_curr in range(N_ANCHORS):
                # Emission score is constant for this target state
                note_coord_x = PITCH_TO_KEYPOS_LUT[notes_pitch[t]][0]
                hand_pos_center_x = note_coord_x + ANCHORS[k_curr]
                emit = compute_emission_score(
                    note_coord_x,
                    hand_pos_center_x,
                    f_curr,
                    rbf_mu,
                    rbf_sigma
                )

                for f_prev in range(N_FINGERS):
                    max_prob = -np.inf
                    best_k_prev = -1
                    best_f_prev2 = -1

                    for k_prev in range(N_ANCHORS):
                        # Calculate physical distance for inertia
                        note_coord_x_prev = PITCH_TO_KEYPOS_LUT[notes_pitch[t-1]][0]
                        hand_center_x_prev = note_coord_x_prev + ANCHORS[k_prev]

                        note_coord_x_curr = PITCH_TO_KEYPOS_LUT[notes_pitch[t]][0]
                        hand_center_x_curr = note_coord_x_curr + ANCHORS[k_curr]

                        dist = np.abs(hand_center_x_curr - hand_center_x_prev)

                        inertia = compute_inertia_cost(dist, dt, inertia_param_slope, inertia_param_center, inertia_weight)

                        for f_prev2 in range(N_FINGERS):
                            # Heuristic bonus for smooth finger transitions
                            agility_bonus = 0.0
                            if abs(f_curr - f_prev) <= 1:
                                agility_bonus = 0.3  # Reward for adjacent fingers
                            else:
                                agility_bonus = -0.3 # Penalty for jumps
                            prev_prob = lattice_log_probs[t-1, f_prev2, f_prev, k_prev]
                            agility = agility_matrix[f_prev2, f_prev, f_curr]

                            smoothing = np.abs(ANCHORS[k_curr] - ANCHORS[k_prev]) * smoothing_weight
                            candidate = prev_prob + agility - inertia - smoothing + emit + agility_bonus

                            if candidate > max_prob:
                                max_prob = candidate
                                best_k_prev = k_prev
                                best_f_prev2 = f_prev2

                    # Store result
                    lattice_log_probs[t, f_prev, f_curr, k_curr] = max_prob
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 0] = best_f_prev2
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 1] = f_prev
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 2] = best_k_prev

@nb.njit(cache=True)
def run_constrained_forward_pass(
    n_obs: int,
    notes_pitch: np.ndarray,
    notes_ontime: np.ndarray,
    true_fingers: np.ndarray,
    lattice_log_probs: np.ndarray,
    lattice_backpointers: np.ndarray,
    agility_matrix: np.ndarray,
    inertia_param_slope: float,
    inertia_param_center: float,
    inertia_weight: float,
    rbf_mu: np.ndarray,
    rbf_sigma: np.ndarray,
    smoothing_weight: float
):
    """
    A constrained version of the forward pass that forces the path to use
    the provided true_fingers.
    """
    # A. Initialization (t = 0)
    if true_fingers[0] != FINGER_UNKNOWN:
        f_curr = true_fingers[0] - 1
        for k_curr in range(N_ANCHORS):
            note_coord_x = PITCH_TO_KEYPOS_LUT[notes_pitch[0]][0]
            hand_pos_center_x = note_coord_x + ANCHORS[k_curr]
            emit = compute_emission_score(
                note_coord_x,
                hand_pos_center_x,
                f_curr,
                rbf_mu,
                rbf_sigma
            )
            for f_prev in range(N_FINGERS):
                lattice_log_probs[0, f_prev, f_curr, k_curr] = emit
    else:  # Unconstrained init
        for f_curr in range(N_FINGERS):
            for k_curr in range(N_ANCHORS):
                note_coord_x = PITCH_TO_KEYPOS_LUT[notes_pitch[0]][0]
                hand_pos_center_x = note_coord_x + ANCHORS[k_curr]
                emit = compute_emission_score(
                    note_coord_x,
                    hand_pos_center_x,
                    f_curr,
                    rbf_mu,
                    rbf_sigma
                )
                for f_prev in range(N_FINGERS):
                    lattice_log_probs[0, f_prev, f_curr, k_curr] = emit

    # B. Recursion (t = 1 to n_obs - 1)
    for t in range(1, n_obs):
        dt = notes_ontime[t] - notes_ontime[t-1]

        if true_fingers[t] != FINGER_UNKNOWN and true_fingers[t-1] != FINGER_UNKNOWN:
            # --- CONSTRAINED PATH ---
            f_curr = true_fingers[t] - 1
            f_prev = true_fingers[t-1] - 1

            for k_curr in range(N_ANCHORS):
                note_coord_x = PITCH_TO_KEYPOS_LUT[notes_pitch[t]][0]
                hand_pos_center_x = note_coord_x + ANCHORS[k_curr]
                emit = compute_emission_score(
                    note_coord_x,
                    hand_pos_center_x,
                    f_curr,
                    rbf_mu,
                    rbf_sigma
                )

                max_prob = -np.inf
                best_k_prev = -1
                best_f_prev2 = -1

                for k_prev in range(N_ANCHORS):
                    note_coord_x_prev = PITCH_TO_KEYPOS_LUT[notes_pitch[t-1]][0]
                    hand_center_x_prev = note_coord_x_prev + ANCHORS[k_prev]

                    note_coord_x_curr = PITCH_TO_KEYPOS_LUT[notes_pitch[t]][0]
                    hand_center_x_curr = note_coord_x_curr + ANCHORS[k_curr]

                    dist = np.abs(hand_center_x_curr - hand_center_x_prev)

                    inertia = compute_inertia_cost(dist, dt, inertia_param_slope, inertia_param_center, inertia_weight)

                    for f_prev2 in range(N_FINGERS):
                        # Heuristic bonus for smooth finger transitions
                        agility_bonus = 0.0
                        if abs(f_curr - f_prev) <= 1:
                            agility_bonus = 0.3  # Reward for adjacent fingers
                        else:
                            agility_bonus = -0.3 # Penalty for jumps
                        prev_prob = lattice_log_probs[t-1, f_prev2, f_prev, k_prev]
                        agility = agility_matrix[f_prev2, f_prev, f_curr]

                        smoothing = np.abs(ANCHORS[k_curr] - ANCHORS[k_prev]) * smoothing_weight
                        candidate = prev_prob + agility - inertia - smoothing + emit + agility_bonus

                        if candidate > max_prob:
                            max_prob = candidate
                            best_k_prev = k_prev
                            best_f_prev2 = f_prev2

                lattice_log_probs[t, f_prev, f_curr, k_curr] = max_prob
                lattice_backpointers[t, f_prev, f_curr, k_curr, 0] = best_f_prev2
                lattice_backpointers[t, f_prev, f_curr, k_curr, 1] = f_prev
                lattice_backpointers[t, f_prev, f_curr, k_curr, 2] = best_k_prev
        else:
            # --- UNCONSTRAINED (STANDARD FORWARD) PATH ---
            for f_curr in range(N_FINGERS):
                for k_curr in range(N_ANCHORS):
                    note_coord_x = PITCH_TO_KEYPOS_LUT[notes_pitch[t]][0]
                    hand_pos_center_x = note_coord_x + ANCHORS[k_curr]
                    emit = compute_emission_score(
                        note_coord_x,
                        hand_pos_center_x,
                        f_curr,
                        rbf_mu,
                        rbf_sigma
                    )

                    for f_prev in range(N_FINGERS):
                        max_prob = -np.inf
                        best_k_prev = -1
                        best_f_prev2 = -1

                        for k_prev in range(N_ANCHORS):
                            note_coord_x_prev = PITCH_TO_KEYPOS_LUT[notes_pitch[t-1]][0]
                            hand_center_x_prev = note_coord_x_prev + ANCHORS[k_prev]

                            note_coord_x_curr = PITCH_TO_KEYPOS_LUT[notes_pitch[t]][0]
                            hand_center_x_curr = note_coord_x_curr + ANCHORS[k_curr]

                            dist = np.abs(hand_center_x_curr - hand_center_x_prev)

                            inertia = compute_inertia_cost(dist, dt, inertia_param_slope, inertia_param_center, inertia_weight)

                            for f_prev2 in range(N_FINGERS):
                                # Heuristic bonus for smooth finger transitions
                                agility_bonus = 0.0
                                if abs(f_curr - f_prev) <= 1:
                                    agility_bonus = 0.3  # Reward for adjacent fingers
                                else:
                                    agility_bonus = -0.3 # Penalty for jumps
                                prev_prob = lattice_log_probs[t-1, f_prev2, f_prev, k_prev]
                                agility = agility_matrix[f_prev2, f_prev, f_curr]

                                smoothing = np.abs(ANCHORS[k_curr] - ANCHORS[k_prev]) * smoothing_weight
                                candidate = prev_prob + agility - inertia - smoothing + emit + agility_bonus

                                if candidate > max_prob:
                                    max_prob = candidate
                                    best_k_prev = k_prev
                                    best_f_prev2 = f_prev2

                        lattice_log_probs[t, f_prev, f_curr, k_curr] = max_prob
                        lattice_backpointers[t, f_prev, f_curr, k_curr, 0] = best_f_prev2
                        lattice_backpointers[t, f_prev, f_curr, k_curr, 1] = f_prev
                        lattice_backpointers[t, f_prev, f_curr, k_curr, 2] = best_k_prev

@nb.njit(cache=True)
def backtracking(
    n_obs: int,
    lattice_log_probs: np.ndarray,
    lattice_backpointers: np.ndarray
):
    """
    Reconstructs the optimal path from the filled lattice.
    """
    opt_fingers = np.zeros(n_obs, dtype=np.int32)
    opt_anchors = np.zeros(n_obs, dtype=np.int32)

    # 1. Find the best ending state
    best_prob = -np.inf
    f_prev, f_curr, k_curr = -1, -1, -1

    reshaped_probs = lattice_log_probs[-1].ravel()
    best_flat_idx = np.argmax(reshaped_probs)

    # Manual unravel_index for Numba compatibility
    k_curr = best_flat_idx % N_ANCHORS
    f_curr = (best_flat_idx // N_ANCHORS) % N_FINGERS
    f_prev = best_flat_idx // (N_FINGERS * N_ANCHORS)

    opt_fingers[-1] = f_curr + 1
    opt_anchors[-1] = k_curr

    # 2. Iterate backwards
    for t in range(n_obs - 1, 0, -1):
        bp = lattice_backpointers[t, f_prev, f_curr, k_curr]
        f_prev2, _, k_prev = bp[0], bp[1], bp[2]

        opt_fingers[t-1] = f_prev + 1
        opt_anchors[t-1] = k_prev

        f_curr = f_prev
        f_prev = f_prev2
        k_curr = k_prev

    return opt_fingers, opt_anchors
