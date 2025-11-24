import numpy as np
from model import HMMModel
from utils import parse_all_scores

def supervised_learning(scores_dir):
    """
    Performs supervised learning on a directory of score files to train an HMM model.

    Args:
        scores_dir (str): The directory containing the training score files.

    Returns:
        HMMModel: A trained HMMModel instance.
    """
    model = HMMModel()

    # Initialize counts to zero
    initial_counts = np.zeros((2, 5))
    transition_counts_1st = np.zeros((2, 5, 5))
    transition_counts_2nd = np.zeros((2, 5, 5, 5))
    output_counts_1st = np.zeros((2, 5, 5, model.nOut))
    output_counts_2nd = np.zeros((2, 5, 5, model.nOut))

    all_scores = parse_all_scores(scores_dir)

    for filename, notes in all_scores.items():
        for hand in [0, 1]: # 0: RH, 1: LH
            hand_notes = notes[notes['hand'] == hand]
            if len(hand_notes) == 0:
                continue

            fingers = hand_notes['finger'] - 1 # Convert to 0-4 index

            # Initial probabilities
            if len(fingers) > 0:
                initial_counts[hand, fingers[0]] += 1

            # 1st-order transitions and emissions
            for n in range(1, len(hand_notes)):
                f_prev, f_curr = fingers[n-1], fingers[n]
                transition_counts_1st[hand, f_prev, f_curr] += 1

                key_int_x = int(hand_notes[n]['lattice_x'] - hand_notes[n-1]['lattice_x'])
                key_int_y = int(hand_notes[n]['lattice_y'] - hand_notes[n-1]['lattice_y'])
                if abs(key_int_x) > model.widthX: key_int_x = np.sign(key_int_x) * model.widthX
                out_idx = 3 * (key_int_x + model.widthX) + key_int_y + 1

                output_counts_1st[hand, f_prev, f_curr, out_idx] += 1

            # 2nd-order transitions and emissions
            for n in range(2, len(hand_notes)):
                f_prev2, f_prev1, f_curr = fingers[n-2], fingers[n-1], fingers[n]
                transition_counts_2nd[hand, f_prev2, f_prev1, f_curr] += 1

                key_int_x = int(hand_notes[n]['lattice_x'] - hand_notes[n-2]['lattice_x'])
                key_int_y = int(hand_notes[n]['lattice_y'] - hand_notes[n-2]['lattice_y'])
                if abs(key_int_x) > model.widthX: key_int_x = np.sign(key_int_x) * model.widthX
                out_idx = 3 * (key_int_x + model.widthX) + key_int_y + 1

                output_counts_2nd[hand, f_prev2, f_curr, out_idx] += 1

    # --- Normalize and convert to log probabilities ---
    epsilon = 1e-10

    model.initial_probabilities = np.log(initial_counts / initial_counts.sum(axis=1, keepdims=True) + epsilon)

    model.transition_matrix_1st = np.log(transition_counts_1st / transition_counts_1st.sum(axis=2, keepdims=True) + epsilon)

    model.transition_matrix_2nd = np.log(transition_counts_2nd / transition_counts_2nd.sum(axis=3, keepdims=True) + epsilon)

    model.output_prob_1st = np.log(output_counts_1st / output_counts_1st.sum(axis=3, keepdims=True) + epsilon)

    model.output_prob_2nd = np.log(output_counts_2nd / output_counts_2nd.sum(axis=3, keepdims=True) + epsilon)

    return model

if __name__ == '__main__':
    # Example usage:
    trained_model = supervised_learning('./scores')
    # You would typically save the trained model parameters here
    print("Training complete.")
    # For example, print the initial probabilities for the right hand
    print("RH Initial Probs (log):", trained_model.initial_probabilities[0])
