import numpy as np
from .core import SoftPositionModel, ANCHORS, FINGER_BASE_POS
from .structural import ViterbiLattice, N_FINGERS
from .inference import run_constrained_forward_pass, backtracking
from .utils import load_pig_file, apply_time_dep_pitch_order, FINGER_UNKNOWN, PITCH_TO_KEYPOS_LUT

class SoftPositionTrainer:
    def __init__(self):
        self.model = SoftPositionModel()
        self.agility_matrix = np.log(np.full((5, 5, 5), 1.0 / 125, dtype=np.float64))

    def train(self, file_paths: list, n_iterations: int = 5, smoothing_weight: float = 0.0) -> list:
        log_likelihood_history = []

        for it in range(n_iterations):
            print(f"--- Iteration {it + 1}/{n_iterations} ---")

            total_log_likelihood = 0.0
            finger_deltas = [[] for _ in range(N_FINGERS)]
            transition_counts = np.zeros((N_FINGERS, N_FINGERS, N_FINGERS), dtype=np.float64)

            for fpath in file_paths:
                try:
                    all_notes = load_pig_file(fpath)
                except ValueError as e:
                    print(f"Skipping file {fpath}: {e}")
                    continue

                rh_notes = all_notes[all_notes['channel'] == 0]
                lh_notes = all_notes[all_notes['channel'] == 1]
                sequences_to_process = [rh_notes, lh_notes]

                # MODIFIÉ: Utiliser enumerate pour identifier la main (0=RH, 1=LH)
                for hand_idx, hand_notes in enumerate(sequences_to_process):
                    if len(hand_notes) < 3:
                        continue

                    notes_sorted = apply_time_dep_pitch_order(hand_notes)
                    n_obs = len(notes_sorted)
                    notes_pitch = np.array([n['pitch'] for n in notes_sorted], dtype=np.int32)
                    notes_ontime = np.array([n['ontime'] for n in notes_sorted], dtype=np.float64)

                    # MODIFIÉ: Préparer les coordonnées et les inverser pour la main gauche
                    notes_coord_x = PITCH_TO_KEYPOS_LUT[notes_pitch, 0].copy()
                    if hand_idx == 1:  # Si c'est la main gauche
                        notes_coord_x *= -1

                    true_fingers = np.array(
                        [abs(n['finger']) if n['finger'] != FINGER_UNKNOWN else FINGER_UNKNOWN
                         for n in notes_sorted],
                        dtype=np.int32
                    )

                    lattice = ViterbiLattice(n_obs)
                    run_constrained_forward_pass(
                        n_obs=n_obs,
                        notes_coord_x=notes_coord_x, # MODIFIÉ: Passer les coordonnées
                        notes_ontime=notes_ontime,
                        true_fingers=true_fingers,
                        lattice_log_probs=lattice.log_probs,
                        lattice_backpointers=lattice.backpointers,
                        agility_matrix=self.agility_matrix,
                        inertia_param_slope=self.model.time_slope,
                        inertia_param_center=self.model.time_center,
                        inertia_weight=self.model.inertia_weight,
                        rbf_mu=self.model.rbf_mu,
                        rbf_sigma=self.model.rbf_sigma,
                        smoothing_weight=smoothing_weight
                    )

                    final_log_prob = np.max(lattice.log_probs[-1])
                    if np.isfinite(final_log_prob):
                        total_log_likelihood += final_log_prob

                    _, opt_anchors = backtracking(n_obs, lattice.log_probs, lattice.backpointers)

                    for i in range(n_obs):
                        if true_fingers[i] != FINGER_UNKNOWN:
                            finger_idx = true_fingers[i] - 1
                            anchor_idx = opt_anchors[i]
                            
                            delta = -ANCHORS[anchor_idx] - FINGER_BASE_POS[finger_idx]
                            finger_deltas[finger_idx].append(delta)

                    for i in range(2, n_obs):
                        f_prev2 = true_fingers[i - 2] - 1
                        f_prev1 = true_fingers[i - 1] - 1
                        f_curr = true_fingers[i] - 1

                        if 0 <= f_prev2 < 5 and 0 <= f_prev1 < 5 and 0 <= f_curr < 5:
                            transition_counts[f_prev2, f_prev1, f_curr] += 1.0

            self._update_emission_parameters(finger_deltas)
            self._update_agility_parameters(transition_counts)
            log_likelihood_history.append(total_log_likelihood)
            print(f"Total Log Likelihood: {total_log_likelihood}")

        return log_likelihood_history

    def _update_emission_parameters(self, finger_deltas):
        print("Updating RBF parameters (mu and sigma)...")
        for i in range(N_FINGERS):
            if len(finger_deltas[i]) > 1:
                self.model.rbf_mu[i] = np.mean(finger_deltas[i])
                new_sigma = np.std(finger_deltas[i])
                self.model.rbf_sigma[i] = max(5.0, new_sigma) # Plancher pour éviter l'effondrement
        print("Model RBF parameters updated.")
        print(f"  New Mu: {np.round(self.model.rbf_mu, 2)}")
        print(f"  New Sigma: {np.round(self.model.rbf_sigma, 2)}")

    def _update_agility_parameters(self, counts):
        print("Updating Agility Matrix...")
        smoothed_counts = counts + 1e-3
        sums = smoothed_counts.sum(axis=2, keepdims=True)
        # Éviter la division par zéro pour les états non observés
        sums[sums == 0] = 1.0
        probs = smoothed_counts / sums
        self.agility_matrix = np.log(probs + 1e-12)
        non_zeros = np.count_nonzero(counts)
        print(f"Agility Matrix updated. {non_zeros} active transitions learned.")