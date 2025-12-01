import numpy as np
from .core import SoftPositionModel, ANCHORS, FINGER_BASE_POS
from .structural import ViterbiLattice, N_FINGERS
from .inference import run_constrained_forward_pass, backtracking
from .utils import load_pig_file, apply_time_dep_pitch_order, FINGER_UNKNOWN, PITCH_TO_KEYPOS_LUT

class SoftPositionTrainer:
    def __init__(self):
        self.model = SoftPositionModel()
        # Initialize agility with uniform log-probs (effectively zeros if not normalized, 
        # but logically should be log(1/5)). 
        # Here we start with 0.0 as per original design, but it will be overwritten.
        self.agility_matrix = np.log(np.full((5, 5, 5), 1.0 / 125, dtype=np.float64))

    def train(self, file_paths: list, n_iterations: int = 5, smoothing_weight: float = 0.0) -> list:
        """
        Runs the EM training loop. Returns list of Total Log Likelihoods per iteration.
        """
        log_likelihood_history = []

        for it in range(n_iterations):
            print(f"--- Iteration {it + 1}/{n_iterations} ---")

            total_log_likelihood = 0.0

            # Accumulators for the M-Step
            # 1. Emission (RBF)
            finger_deltas = [[] for _ in range(N_FINGERS)]
            
            # 2. Transitions (Agility) - Shape (Prev2, Prev1, Curr)
            transition_counts = np.zeros((N_FINGERS, N_FINGERS, N_FINGERS), dtype=np.float64)

            for fpath in file_paths:
                # 1. Load Data
                try:
                    notes = load_pig_file(fpath)
                except ValueError as e:
                    print(f"Skipping file {fpath}: {e}")
                    continue

                notes_sorted = apply_time_dep_pitch_order(notes)

                n_obs = len(notes_sorted)
                if n_obs == 0:
                    continue

                notes_pitch = np.array([n['pitch'] for n in notes_sorted], dtype=np.int32)
                notes_ontime = np.array([n['ontime'] for n in notes_sorted], dtype=np.float64)
                true_fingers = np.array([n['finger'] for n in notes_sorted], dtype=np.int32)

                # 2. E-Step: Guess the Anchors (Constrained)
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

                # Log-likelihood accumulation
                final_log_prob = np.max(lattice.log_probs[-1])
                if np.isfinite(final_log_prob):
                    total_log_likelihood += final_log_prob

                # 3. Collect Statistics
                # A. Anchors (for Emission)
                _, opt_anchors = backtracking(n_obs, lattice.log_probs, lattice.backpointers)

                for i in range(n_obs):
                    if true_fingers[i] != FINGER_UNKNOWN:
                        finger_idx = true_fingers[i] - 1
                        anchor_idx = opt_anchors[i]

                        # Get physical coordinates
                        note_coord_x = PITCH_TO_KEYPOS_LUT[notes_pitch[i]][0]
                        anchor_pos_x = ANCHORS[anchor_idx] # Anchor is now an absolute offset

                        # This needs to be consistent with the inference step
                        # The anchor is an offset from the note's physical position to the hand's center
                        hand_pos_center_x = note_coord_x + anchor_pos_x

                        # Calculate the target x-position of the finger
                        finger_target_pos = hand_pos_center_x + FINGER_BASE_POS[finger_idx]

                        # The delta is the difference between the finger's actual target and the note's position
                        # This represents the error the RBF learns to model
                        delta = note_coord_x - finger_target_pos

                        finger_deltas[finger_idx].append(delta)
                
                # B. Transitions (for Agility)
                # We iterate from index 2 to account for the 2nd order dependency
                for i in range(2, n_obs):
                    f_prev2 = true_fingers[i-2] - 1
                    f_prev1 = true_fingers[i-1] - 1
                    f_curr  = true_fingers[i] - 1
                    
                    if 0 <= f_prev2 < 5 and 0 <= f_prev1 < 5 and 0 <= f_curr < 5:
                        transition_counts[f_prev2, f_prev1, f_curr] += 1.0

            # 4. M-Step: Update Parameters
            self._update_emission_parameters(finger_deltas)
            self._update_agility_parameters(transition_counts)

            log_likelihood_history.append(total_log_likelihood)
            print(f"Total Log Likelihood: {total_log_likelihood}")

        return log_likelihood_history

    def _update_emission_parameters(self, finger_deltas):
        """
        Re-estimates the RBF parameters (Mu, Sigma) for each finger.
        """
        print("Updating RBF parameters (mu and sigma)...")
        for i in range(N_FINGERS):
            if len(finger_deltas[i]) > 1:
                computed_mean = np.mean(finger_deltas[i])
                new_mu = 0.9 * self.model.rbf_mu[i] + 0.1 * computed_mean
                self.model.rbf_mu[i] = np.clip(new_mu, -100, 100)

                new_sigma = np.std(finger_deltas[i])
                # Ensure sigma doesn't collapse to 0
                self.model.rbf_sigma[i] = max(0.3, new_sigma)
        print("Model RBF parameters updated.")

    def _update_agility_parameters(self, counts):
        """
        Re-estimates the Agility Matrix (Transition Probabilities).
        Calculates P(f_curr | f_prev1, f_prev2)
        Handles unobserved states by defaulting to Uniform Distribution.
        """
        print("Updating Agility Matrix...")
        
        # Apply Laplace Smoothing
        smoothed_counts = counts + 1e-3

        # 1. Sum counts for normalization: sum over the destination finger (axis 2)
        sums = smoothed_counts.sum(axis=2, keepdims=True)
        
        # 2. Calculate Probabilities
        probs = smoothed_counts / sums
        
        # 3. Convert to Log-Probabilities
        self.agility_matrix = np.log(probs + 1e-12)
        
        # Debug: Check non-zero entries
        non_zeros = np.count_nonzero(counts)
        print(f"Agility Matrix updated. {non_zeros} active transitions learned.")