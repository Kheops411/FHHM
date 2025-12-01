import numpy as np
import numba as nb
from .core import compute_emission_score, compute_inertia_cost, ANCHORS
from .structural import N_FINGERS, N_ANCHORS

@nb.njit(cache=True)
def run_forward_pass(
    n_obs: int,
    notes_coord_x: np.ndarray,
    notes_ontime: np.ndarray,
    lattice_log_probs: np.ndarray,
    lattice_backpointers: np.ndarray,
    agility_matrix: np.ndarray,
    inertia_param_slope: float,
    inertia_param_center: float,
    inertia_weight: float,
    rbf_mu: np.ndarray,
    rbf_sigma: np.ndarray,
    smoothing_weight: float,
    finger_base_pos: np.ndarray
):
    if n_obs == 0:
        return

    # A. Initialization (t = 0)
    current_note_coord_x = notes_coord_x[0]
    for f_curr in range(N_FINGERS):
        for k_curr in range(N_ANCHORS):
            hand_pos_center_x_abs = current_note_coord_x + ANCHORS[k_curr]
            emit = compute_emission_score(
                current_note_coord_x,
                hand_pos_center_x_abs,
                f_curr,
                rbf_mu,
                rbf_sigma,
                finger_base_pos
            )
            for f_prev in range(N_FINGERS):
                lattice_log_probs[0, f_prev, f_curr, k_curr] = emit

    if n_obs == 1:
        return

    # B. First Transition (t = 1) - Special Case
    dt = notes_ontime[1] - notes_ontime[0]
    current_note_coord_x = notes_coord_x[1]
    prev_note_coord_x = notes_coord_x[0]
    for f_curr in range(N_FINGERS):
        for k_curr in range(N_ANCHORS):
            hand_pos_center_x_curr_abs = current_note_coord_x + ANCHORS[k_curr]
            emit = compute_emission_score(
                current_note_coord_x,
                hand_pos_center_x_curr_abs,
                f_curr,
                rbf_mu,
                rbf_sigma,
                finger_base_pos
            )
            for f_prev in range(N_FINGERS):
                max_prob = -np.inf
                best_k_prev = -1
                for k_prev in range(N_ANCHORS):
                    hand_center_x_prev_abs = prev_note_coord_x + ANCHORS[k_prev]
                    dist = np.abs(hand_pos_center_x_curr_abs - hand_center_x_prev_abs)
                    inertia = compute_inertia_cost(dist, dt, inertia_param_slope, inertia_param_center, inertia_weight)
                    smoothing = np.abs(ANCHORS[k_curr] - ANCHORS[k_prev]) * smoothing_weight
                    prev_prob = lattice_log_probs[0, 0, f_prev, k_prev]
                    candidate = prev_prob - inertia - smoothing + emit
                    if candidate > max_prob:
                        max_prob = candidate
                        best_k_prev = k_prev
                lattice_log_probs[1, f_prev, f_curr, k_curr] = max_prob
                lattice_backpointers[1, f_prev, f_curr, k_curr, 0] = -1
                lattice_backpointers[1, f_prev, f_curr, k_curr, 1] = f_prev
                lattice_backpointers[1, f_prev, f_curr, k_curr, 2] = best_k_prev

    # C. Main Recursion (t >= 2)
    for t in range(2, n_obs):
        dt = notes_ontime[t] - notes_ontime[t-1]
        current_note_coord_x = notes_coord_x[t]
        prev_note_coord_x = notes_coord_x[t-1]
        for f_curr in range(N_FINGERS):
            for k_curr in range(N_ANCHORS):
                hand_pos_center_x_curr_abs = current_note_coord_x + ANCHORS[k_curr]
                emit = compute_emission_score(
                    current_note_coord_x,
                    hand_pos_center_x_curr_abs,
                    f_curr,
                    rbf_mu,
                    rbf_sigma,
                    finger_base_pos
                )
                for f_prev in range(N_FINGERS):
                    max_prob = -np.inf
                    best_k_prev = -1
                    best_f_prev2 = -1
                    for k_prev in range(N_ANCHORS):
                        hand_center_x_prev_abs = prev_note_coord_x + ANCHORS[k_prev]
                        dist = np.abs(hand_pos_center_x_curr_abs - hand_center_x_prev_abs)
                        inertia = compute_inertia_cost(dist, dt, inertia_param_slope, inertia_param_center, inertia_weight)
                        for f_prev2 in range(N_FINGERS):
                            prev_prob = lattice_log_probs[t-1, f_prev2, f_prev, k_prev]
                            agility = agility_matrix[f_prev2, f_prev, f_curr]
                            smoothing = np.abs(ANCHORS[k_curr] - ANCHORS[k_prev]) * smoothing_weight
                            candidate = prev_prob + agility - inertia - smoothing + emit
                            if candidate > max_prob:
                                max_prob = candidate
                                best_k_prev = k_prev
                                best_f_prev2 = f_prev2
                    lattice_log_probs[t, f_prev, f_curr, k_curr] = max_prob
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 0] = best_f_prev2
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 1] = f_prev
                    lattice_backpointers[t, f_prev, f_curr, k_curr, 2] = best_k_prev

@nb.njit(cache=True)
def run_constrained_forward_pass(
    n_obs: int,
    notes_coord_x: np.ndarray,
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
    smoothing_weight: float,
    finger_base_pos: np.ndarray
):
    if n_obs == 0:
        return

    # A. Initialization (t = 0)
    current_note_coord_x = notes_coord_x[0]
    f_init_range = range(N_FINGERS)
    if true_fingers[0] != -999:
        f_init_range = range(true_fingers[0] - 1, true_fingers[0])
    for f_curr in f_init_range:
        for k_curr in range(N_ANCHORS):
            hand_pos_center_x_abs = current_note_coord_x + ANCHORS[k_curr]
            emit = compute_emission_score(
                current_note_coord_x,
                hand_pos_center_x_abs,
                f_curr,
                rbf_mu,
                rbf_sigma,
                finger_base_pos
            )
            for f_prev in range(N_FINGERS):
                lattice_log_probs[0, f_prev, f_curr, k_curr] = emit

    if n_obs == 1:
        return

    # B. First Transition (t = 1)
    dt = notes_ontime[1] - notes_ontime[0]
    current_note_coord_x = notes_coord_x[1]
    prev_note_coord_x = notes_coord_x[0]
    f_curr_range = range(N_FINGERS)
    if true_fingers[1] != -999:
        f_curr_range = range(true_fingers[1] - 1, true_fingers[1])
    f_prev_range = range(N_FINGERS)
    if true_fingers[0] != -999:
        f_prev_range = range(true_fingers[0] - 1, true_fingers[0])

    for f_curr in f_curr_range:
        for k_curr in range(N_ANCHORS):
            hand_pos_center_x_curr_abs = current_note_coord_x + ANCHORS[k_curr]
            emit = compute_emission_score(
                current_note_coord_x,
                hand_pos_center_x_curr_abs,
                f_curr,
                rbf_mu,
                rbf_sigma,
                finger_base_pos
            )
            for f_prev in f_prev_range:
                max_prob = -np.inf
                best_k_prev = -1
                for k_prev in range(N_ANCHORS):
                    hand_center_x_prev_abs = prev_note_coord_x + ANCHORS[k_prev]
                    dist = np.abs(hand_pos_center_x_curr_abs - hand_center_x_prev_abs)
                    inertia = compute_inertia_cost(dist, dt, inertia_param_slope, inertia_param_center, inertia_weight)
                    smoothing = np.abs(ANCHORS[k_curr] - ANCHORS[k_prev]) * smoothing_weight
                    prev_prob = lattice_log_probs[0, 0, f_prev, k_prev]
                    candidate = prev_prob - inertia - smoothing + emit
                    if candidate > max_prob:
                        max_prob = candidate
                        best_k_prev = k_prev
                lattice_log_probs[1, f_prev, f_curr, k_curr] = max_prob
                lattice_backpointers[1, f_prev, f_curr, k_curr, 0] = -1
                lattice_backpointers[1, f_prev, f_curr, k_curr, 1] = f_prev
                lattice_backpointers[1, f_prev, f_curr, k_curr, 2] = best_k_prev

    # C. Main Recursion (t >= 2)
    for t in range(2, n_obs):
        dt = notes_ontime[t] - notes_ontime[t-1]
        current_note_coord_x = notes_coord_x[t]
        prev_note_coord_x = notes_coord_x[t-1]
        f_curr_range = range(N_FINGERS)
        if true_fingers[t] != -999:
            f_curr_range = range(true_fingers[t] - 1, true_fingers[t])
        f_prev_range = range(N_FINGERS)
        if true_fingers[t-1] != -999:
            f_prev_range = range(true_fingers[t-1] - 1, true_fingers[t-1])

        for f_curr in f_curr_range:
            for k_curr in range(N_ANCHORS):
                hand_pos_center_x_curr_abs = current_note_coord_x + ANCHORS[k_curr]
                emit = compute_emission_score(
                    current_note_coord_x,
                    hand_pos_center_x_curr_abs,
                    f_curr,
                    rbf_mu,
                    rbf_sigma,
                    finger_base_pos
                )
                for f_prev in f_prev_range:
                    max_prob = -np.inf
                    best_k_prev = -1
                    best_f_prev2 = -1
                    for k_prev in range(N_ANCHORS):
                        hand_center_x_prev_abs = prev_note_coord_x + ANCHORS[k_prev]
                        dist = np.abs(hand_pos_center_x_curr_abs - hand_center_x_prev_abs)
                        inertia = compute_inertia_cost(dist, dt, inertia_param_slope, inertia_param_center, inertia_weight)
                        for f_prev2 in range(N_FINGERS):
                            prev_prob = lattice_log_probs[t-1, f_prev2, f_prev, k_prev]
                            agility = agility_matrix[f_prev2, f_prev, f_curr]
                            smoothing = np.abs(ANCHORS[k_curr] - ANCHORS[k_prev]) * smoothing_weight
                            candidate = prev_prob + agility - inertia - smoothing + emit
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
    opt_fingers = np.zeros(n_obs, dtype=np.int32)
    opt_anchors = np.zeros(n_obs, dtype=np.int32)

    # 1. Trouver le meilleur état final
    best_prob = -np.inf
    f_prev, f_curr, k_curr = -1, -1, -1

    reshaped_probs = lattice_log_probs[-1].ravel()
    best_flat_idx = np.argmax(reshaped_probs)

    k_curr = best_flat_idx % N_ANCHORS
    f_curr = (best_flat_idx // N_ANCHORS) % N_FINGERS
    f_prev = best_flat_idx // (N_FINGERS * N_ANCHORS)

    opt_fingers[-1] = f_curr + 1
    opt_anchors[-1] = k_curr

    # 2. Itérer à rebours
    for t in range(n_obs - 1, 0, -1):
        bp = lattice_backpointers[t, f_prev, f_curr, k_curr]
        f_prev2, _, k_prev = bp[0], bp[1], bp[2]

        opt_fingers[t-1] = f_prev + 1
        opt_anchors[t-1] = k_prev

        f_curr = f_prev
        f_prev = f_prev2
        k_curr = k_prev

    return opt_fingers, opt_anchors
