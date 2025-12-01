import numpy as np
from .core import SoftPositionModel, ANCHORS
from .structural import ViterbiLattice, N_FINGERS
from .inference import run_constrained_forward_pass, backtracking
from .utils import load_pig_file, apply_time_dep_pitch_order, filter_notes_by_hand

class SoftPositionTrainer:
    def __init__(self):
        self.model = SoftPositionModel()
        # Initialize agility with uniform log-probs (effectively zeros if not normalized, 
        # but logically should be log(1/5)). 
        # Here we start with 0.0 as per original design, but it will be overwritten.
        self.agility_matrix = np.zeros((5, 5, 5), dtype=np.float64)

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

                notes_rh = filter_notes_by_hand(notes, 0) # 0 for Right Hand
                notes_sorted = apply_time_dep_pitch_order(notes_rh)

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
                    finger_idx = true_fingers[i] - 1
                    anchor_idx = opt_anchors[i]
                    hand_pos = notes_pitch[i] + ANCHORS[anchor_idx]
                    delta = notes_pitch[i] - hand_pos
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
                self.model.rbf_mu[i] = np.mean(finger_deltas[i])
                new_sigma = np.std(finger_deltas[i])
                # Ensure sigma doesn't collapse to 0
                self.model.rbf_sigma[i] = max(0.5, new_sigma)
        print("Model RBF parameters updated.")

    def _update_agility_parameters(self, counts):
        """
        Re-estimates the Agility Matrix (Transition Probabilities).
        Calculates P(f_curr | f_prev1, f_prev2)
        Handles unobserved states by defaulting to Uniform Distribution.
        """
        print("Updating Agility Matrix...")
        
        # 1. Sum counts for normalization: sum over the destination finger (axis 2)
        # Shape: (5, 5, 1)
        sums = counts.sum(axis=2, keepdims=True)
        
        # 2. Identify rows with no data
        # Boolean mask where sum is 0
        missing_data_mask = (sums == 0)
        
        # 3. Avoid division by zero
        # Temporarily set sum to 1.0 so division works (result will be 0, corrected later)
        sums[missing_data_mask] = 1.0
        
        # 4. Calculate Probabilities (ML Estimate)
        probs = counts / sums
        
        # 5. Apply Fallback for Missing Data (Uniform Distribution)
        # If no data observed, probability is 1/5 for all 5 fingers
        # We broadcast the scalar 0.2 into the masked locations
        uniform_prob = 1.0 / 5.0 # 0.2
        
        # We need to broadcast the mask (5,5,1) to (5,5,5) to set values
        # Or simply iterate. Vectorized approach:
        # P[i,j,:] = 0.2 where mask[i,j,0] is True
        
        # Efficient numpy assignment using broadcasting:
        # missing_data_mask is (5,5,1). 
        # We want to affect probs where mask is True. 
        # probs has shape (5,5,5).
        # We can use np.where
        
        probs = np.where(missing_data_mask, uniform_prob, probs)
        
        # 6. Convert to Log-Probabilities
        # Add epsilon solely for numerical stability of the '0' entries in observed rows
        epsilon = 1e-12
        self.agility_matrix = np.log(probs + epsilon)
        
        # Debug: Check non-zero entries
        non_zeros = np.count_nonzero(counts)
        print(f"Agility Matrix updated. {non_zeros} active transitions learned.")