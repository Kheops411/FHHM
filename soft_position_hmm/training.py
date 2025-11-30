import numpy as np
from .core import SoftPositionModel, ANCHORS
from .structural import ViterbiLattice, N_FINGERS
from .inference import run_constrained_forward_pass, backtracking
from .utils import load_pig_file, apply_time_dep_pitch_order, filter_notes_by_hand

class SoftPositionTrainer:
    def __init__(self):
        self.model = SoftPositionModel()
        # For M4, we initialize agility with zeros (uniform log-prob)
        self.agility_matrix = np.zeros((5, 5, 5), dtype=np.float64)

    def train(self, file_paths: list, n_iterations: int = 5, smoothing_weight: float = 0.0) -> list:
        """
        Runs the EM training loop. Returns list of Total Log Likelihoods per iteration.
        """
        log_likelihood_history = [] # <--- NEW LIST

        for it in range(n_iterations):
            print(f"--- Iteration {it + 1}/{n_iterations} ---")

            total_log_likelihood = 0.0

            # Accumulators for the M-Step
            finger_deltas = [[] for _ in range(N_FINGERS)]

            for fpath in file_paths:
                # 1. Load Data
                notes = load_pig_file(fpath)
                notes_rh = filter_notes_by_hand(notes, 0) # 0 for Right Hand
                notes_sorted = apply_time_dep_pitch_order(notes_rh)

                n_obs = len(notes_sorted)
                if n_obs == 0:
                    continue

                notes_pitch = np.array([n['pitch'] for n in notes_sorted], dtype=np.int32)
                notes_ontime = np.array([n['ontime'] for n in notes_sorted], dtype=np.float64)
                true_fingers = np.array([n['finger'] for n in notes_sorted], dtype=np.int32)

                # 2. E-Step: Guess the Anchors
                lattice = ViterbiLattice(n_obs)
                run_constrained_forward_pass(
                    n_obs=n_obs,
                    notes_pitch=notes_pitch,
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

                # Find the log-likelihood of the best path
                final_log_prob = np.max(lattice.log_probs[-1])
                if np.isfinite(final_log_prob):
                    total_log_likelihood += final_log_prob

                # 3. Collect Statistics
                _, opt_anchors = backtracking(n_obs, lattice.log_probs, lattice.backpointers)

                for i in range(n_obs):
                    finger_idx = true_fingers[i] - 1
                    anchor_idx = opt_anchors[i]

                    # Calculate the absolute hand position
                    hand_pos = notes_pitch[i] + ANCHORS[anchor_idx]

                    # The delta is the observed distance between the note and the hand center
                    delta = notes_pitch[i] - hand_pos
                    finger_deltas[finger_idx].append(delta)

            # 4. M-Step: Update Parameters
            self._update_parameters(finger_deltas)

            log_likelihood_history.append(total_log_likelihood)
            print(f"Total Log Likelihood: {total_log_likelihood}")

        return log_likelihood_history

    def _update_parameters(self, finger_deltas):
        """
        Re-estimates the RBF parameters for each finger based on collected data.
        """
        print("Updating RBF parameters (mu and sigma)...")
        # In the original M1 `compute_emission_score` we used hardcoded offsets/widths
        # Here we update the model's rbf_mu and rbf_sigma which would be used
        # in a more advanced emission score function.
        for i in range(N_FINGERS):
            if len(finger_deltas[i]) > 1:
                self.model.rbf_mu[i] = np.mean(finger_deltas[i])
                # Enforce a minimum sigma to prevent overfitting and divergence
                new_sigma = np.std(finger_deltas[i])
                self.model.rbf_sigma[i] = max(1.0, new_sigma)
        print("Model parameters updated.")
