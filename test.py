import numpy as np
from soft_position_hmm.core import SoftPositionModel, RH_FINGER_BASE_POS
from soft_position_hmm.interface import predict_fingering
from soft_position_hmm.structural import ViterbiLattice
from soft_position_hmm.inference import run_forward_pass

def test_hand_symmetry():
    print("--- Running Test 1: Right/Left Hand Symmetry Check ---")
    try:
        model = SoftPositionModel()
        notes = np.array([60, 62, 64], dtype=np.int32) # C4, D4, E4
        times = np.array([0.0, 0.1, 0.2], dtype=np.float64)

        # Right hand prediction
        fingers_rh, _ = predict_fingering(notes, times, model, hand_sign=1)
        
        # Left hand prediction
        fingers_lh, _ = predict_fingering(notes, times, model, hand_sign=-1)
        
        # The fingerings should be symmetrical (e.g., 1↔5, 2↔4, 3↔3)
        # Note: LH fingers are negative, so we take the absolute value
        expected_lh_from_rh_map = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
        
        generated_lh_from_rh = np.array([expected_lh_from_rh_map[f] for f in fingers_rh])
        
        print(f"RH Fingering: {fingers_rh}")
        print(f"LH Fingering: {np.abs(fingers_lh)}")
        print(f"Expected LH from RH: {generated_lh_from_rh}")

        assert np.array_equal(generated_lh_from_rh, np.abs(fingers_lh)), "Hand symmetry test failed!"
        print("SUCCESS: Fingering predictions are symmetrical for right and left hands.")
        return True
    except Exception as e:
        print(f"FAILURE: {e}")
        return False

def test_backtracking_short_sequence():
    print("\n--- Running Test 2: Backtracking Validation with t=1 (2 notes) ---")
    try:
        model = SoftPositionModel()
        notes = np.array([60, 62], dtype=np.int32)
        times = np.array([0.0, 0.2], dtype=np.float64)
        
        # This call should complete without crashing
        fingers, _ = predict_fingering(notes, times, model, hand_sign=1)
        
        print(f"Predicted fingers for 2-note sequence: {fingers}")
        assert len(fingers) == 2, "Backtracking did not return 2 fingers."
        print("SUCCESS: Backtracking handles 2-note sequences correctly without crashing.")
        return True
    except Exception as e:
        print(f"FAILURE: {e}")
        return False

def test_probability_consistency():
    print("\n--- Running Test 3: Probability Consistency Check ---")
    try:
        n_obs = 5
        model = SoftPositionModel()
        lattice = ViterbiLattice(n_obs)
        agility_matrix = np.zeros((5, 5, 5), dtype=np.float64)
        
        run_forward_pass(
            n_obs=n_obs,
            notes_coord_x=np.array([500, 520, 540, 550, 570], dtype=np.float64),
            notes_ontime=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
            lattice_log_probs=lattice.log_probs,
            lattice_backpointers=lattice.backpointers,
            agility_matrix=agility_matrix,
            inertia_param_slope=model.time_slope,
            inertia_param_center=model.time_center,
            inertia_weight=model.inertia_weight,
            rbf_mu=model.rbf_mu,
            rbf_sigma=model.rbf_sigma,
            smoothing_weight=0.1,
            finger_base_pos=RH_FINGER_BASE_POS
        )
        
        assert np.all(np.isfinite(lattice.log_probs)), "Lattice contains non-finite values (NaN or inf)."
        assert np.all(lattice.log_probs <= 0), "Lattice contains positive log-probabilities, which is mathematically incorrect."
        print("SUCCESS: Viterbi lattice probabilities are finite and non-positive.")
        return True
    except Exception as e:
        print(f"FAILURE: {e}")
        return False

if __name__ == "__main__":
    results = [
        test_hand_symmetry(),
        test_backtracking_short_sequence(),
        test_probability_consistency()
    ]
    
    if all(results):
        print("\nAll additional tests passed successfully!")
    else:
        print("\nSome additional tests failed.")
