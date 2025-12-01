import numpy as np
from soft_position_hmm.inference import run_forward_pass
from soft_position_hmm.structural import ViterbiLattice

def test_run_forward_pass():
    """
    Validates that run_forward_pass executes without crashing after the
    Euclidean distance update. It uses dummy data.
    """
    print("TASK 4: Running dummy forward pass...")

    # Setup dummy data
    n_obs = 10
    notes_pitch = np.random.randint(21, 108, size=n_obs, dtype=np.int32)
    notes_ontime = np.sort(np.random.rand(n_obs) * 10).astype(np.float64)

    lattice = ViterbiLattice(n_obs)
    agility_matrix = np.log(np.full((5, 5, 5), 1.0/125.0, dtype=np.float64)) # Must be log

    # Dummy model parameters
    inertia_param_slope = 10.0
    inertia_param_center = 0.2
    inertia_weight = 1.0
    rbf_mu = np.array([-4.0, -2.0, 0.0, 2.0, 5.0], dtype=np.float64)
    rbf_sigma = np.array([4.0, 1.5, 1.5, 1.5, 2.5], dtype=np.float64)
    smoothing_weight = 0.5

    try:
        run_forward_pass(
            n_obs,
            notes_pitch,
            notes_ontime,
            lattice.log_probs,
            lattice.backpointers,
            agility_matrix,
            inertia_param_slope,
            inertia_param_center,
            inertia_weight,
            rbf_mu,
            rbf_sigma,
            smoothing_weight
        )
        print("TASK 4 SUCCESS: Forward pass executed without error.")
    except Exception as e:
        print(f"TASK 4 FAILED: An error occurred during forward pass: {e}")
        raise

if __name__ == "__main__":
    test_run_forward_pass()
