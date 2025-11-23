import numpy as np
from numba import jit

class HMMModel:
    def __init__(self, w1=0.5, w2=0.5, lam1=0, short_time_cost=-5):
        self.widthX = 15
        self.nOut = 3 * (2 * self.widthX + 1)
        self.w1, self.w2, self.lam1, self.short_time_cost = w1, w2, lam1, short_time_cost

        self.initial_probabilities = np.zeros((2, 5))
        self.transition_matrix_1st = np.zeros((2, 5, 5))
        self.transition_matrix_2nd = np.zeros((2, 5, 5, 5))
        self.output_prob_1st = np.zeros((2, 5, 5, self.nOut))
        self.output_prob_2nd = np.zeros((2, 5, 5, self.nOut))

    def load_cpp_parameters(self, filepath):
        params = self._parse_param_file(filepath)
        epsilon = 1e-10

        def normalize_and_log(arr, axis=-1):
            arr_sum = arr.sum(axis=axis, keepdims=True)
            # Avoid division by zero for empty rows
            arr_sum[arr_sum == 0] = 1
            return np.log(arr / arr_sum + epsilon)

        def process_output_probs(key):
            raw = np.array(params.get(key, [])).reshape(5, 5, self.nOut)
            return normalize_and_log(raw, axis=2)

        # Initial Probabilities
        self.initial_probabilities[0, :] = normalize_and_log(np.array(params.get('Initial Prob Right', [epsilon]*5)))
        self.initial_probabilities[1, :] = normalize_and_log(np.array(params.get('Initial Prob Left', [epsilon]*5)))

        # 1st-Order Transition Probabilities (raw for interpolation)
        tr_prob_1st_r_raw = np.array(params.get('Transition Prob Right', [epsilon]*25)).reshape(5, 5)
        tr_prob_1st_l_raw = np.array(params.get('Transition Prob Left', [epsilon]*25)).reshape(5, 5)
        tr_prob_1st_r_raw /= tr_prob_1st_r_raw.sum(axis=1, keepdims=True)
        tr_prob_1st_l_raw /= tr_prob_1st_l_raw.sum(axis=1, keepdims=True)

        self.transition_matrix_1st[0, :, :] = np.log(tr_prob_1st_r_raw + epsilon)
        self.transition_matrix_1st[1, :, :] = np.log(tr_prob_1st_l_raw + epsilon)

        # 2nd-Order Transition Probabilities (interpolation and normalization)
        tr_prob_2nd_r_raw = np.array(params.get('Transition Prob 2nd Right', [epsilon]*125)).reshape(5, 5, 5)
        tr_prob_2nd_l_raw = np.array(params.get('Transition Prob 2nd Left', [epsilon]*125)).reshape(5, 5, 5)

        # Interpolation
        interp_r = (1 - self.lam1) * tr_prob_2nd_r_raw + self.lam1 * tr_prob_1st_r_raw[np.newaxis, :, :]
        interp_l = (1 - self.lam1) * tr_prob_2nd_l_raw + self.lam1 * tr_prob_1st_l_raw[np.newaxis, :, :]

        self.transition_matrix_2nd[0, :, :, :] = normalize_and_log(interp_r, axis=2)
        self.transition_matrix_2nd[1, :, :, :] = normalize_and_log(interp_l, axis=2)

        # Output Probabilities
        self.output_prob_1st[0, :, :, :] = process_output_probs('Output Prob Right')
        self.output_prob_1st[1, :, :, :] = process_output_probs('Output Prob Left')
        self.output_prob_2nd[0, :, :, :] = process_output_probs('Output Prob 2nd Right')
        self.output_prob_2nd[1, :, :, :] = process_output_probs('Output Prob 2nd Left')

    def _parse_param_file(self, filepath):
        params = {}
        current_key = ""
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('###'):
                    current_key = line.replace('### ', '').strip()
                    params[current_key] = []
                elif current_key:
                    parts = line.split()
                    start_idx = 2 if 'Output Prob' in current_key else 0
                    try:
                        params[current_key].extend([float(p) for p in parts[start_idx:]])
                    except (ValueError, IndexError):
                        continue
        return params

@jit(nopython=True)
def viterbi_decode(notes, trans_mat_2nd, trans_mat_1st, initial_probs, out_prob_1st, out_prob_2nd, widthX, w1, w2, short_time_cost, hand):
    """
    Finds the most likely sequence of fingerings using the Viterbi algorithm.

    This function is JIT-compiled with Numba for performance.

    Args:
        notes (np.ndarray): A structured NumPy array of notes for a single hand,
                            containing fields like 'pitch', 'onset', 'lattice_x', etc.
        trans_mat_2nd (np.ndarray): The 2nd-order transition probability matrix (log space).
                                    Shape: (5, 5, 5) for (f_{n-2}, f_{n-1}, f_n).
        trans_mat_1st (np.ndarray): The 1st-order transition probability matrix (log space).
                                    Shape: (5, 5) for (f_{n-1}, f_n).
        initial_probs (np.ndarray): The initial state probabilities (log space). Shape: (5,).
        out_prob_1st (np.ndarray): The 1st-order emission probability matrix (log space).
        out_prob_2nd (np.ndarray): The 2nd-order emission probability matrix (log space).
        widthX (int): The maximum horizontal distance for keyboard coordinates.
        w1 (float): Weight for the 1st-order emission probability.
        w2 (float): Weight for the 2nd-order emission probability.
        short_time_cost (float): A penalty applied for awkward fingerings over short time intervals.
        hand (int): The hand being processed (0 for Right, 1 for Left).

    Returns:
        np.ndarray: An array of integers representing the optimal fingering sequence (1-5).
    """
    num_notes = notes.shape[0]
    num_fingers = 5

    if num_notes < 2:
        return np.ones(num_notes, dtype=np.int8)

    trellis = np.full((num_notes, num_fingers, num_fingers), -np.inf, dtype=np.float64)
    backpointers = np.zeros((num_notes, num_fingers, num_fingers), dtype=np.int8)

    key_int_x = int(notes[1]['lattice_x'] - notes[0]['lattice_x'])
    key_int_y = int(notes[1]['lattice_y'] - notes[0]['lattice_y'])
    if abs(key_int_x) > widthX: key_int_x = np.sign(key_int_x) * widthX
    out_idx = 3 * (key_int_x + widthX) + key_int_y + 1

    for f0 in range(num_fingers):
        for f1 in range(num_fingers):
            trellis[1, f0, f1] = initial_probs[f0] + trans_mat_1st[f0, f1] + out_prob_1st[f0, f1, out_idx]

    for n in range(2, num_notes):
        key_int_x1 = int(notes[n]['lattice_x'] - notes[n-1]['lattice_x'])
        key_int_y1 = int(notes[n]['lattice_y'] - notes[n-1]['lattice_y'])
        if abs(key_int_x1) > widthX: key_int_x1 = np.sign(key_int_x1) * widthX
        out_idx1 = 3 * (key_int_x1 + widthX) + key_int_y1 + 1

        key_int_x2 = int(notes[n]['lattice_x'] - notes[n-2]['lattice_x'])
        key_int_y2 = int(notes[n]['lattice_y'] - notes[n-2]['lattice_y'])
        if abs(key_int_x2) > widthX: key_int_x2 = np.sign(key_int_x2) * widthX
        out_idx2 = 3 * (key_int_x2 + widthX) + key_int_y2 + 1

        short_time = abs(notes[n]['onset'] - notes[n-1]['onset']) < 0.03
        del_pitch = notes[n]['pitch'] - notes[n-1]['pitch']
        short_time2 = abs(notes[n]['onset'] - notes[n-2]['onset']) < 0.03
        del_pitch2 = notes[n]['pitch'] - notes[n-2]['pitch']

        for fn_minus_1 in range(num_fingers):
            for fn in range(num_fingers):
                max_prob = -np.inf
                best_prev_finger = -1
                for fn_minus_2 in range(num_fingers):
                    trans_prob = trans_mat_2nd[fn_minus_2, fn_minus_1, fn]
                    emission_prob = w1 * out_prob_1st[fn_minus_1, fn, out_idx1] + \
                                    w2 * out_prob_2nd[fn_minus_2, fn, out_idx2]

                    penalty = 0.0
                    if short_time and ((hand == 0 and (fn - fn_minus_1) * del_pitch < 0) or (hand == 1 and (fn - fn_minus_1) * del_pitch > 0)):
                        penalty += short_time_cost
                    if short_time2 and ((hand == 0 and (fn - fn_minus_2) * del_pitch2 < 0) or (hand == 1 and (fn - fn_minus_2) * del_pitch2 > 0)):
                        penalty += short_time_cost

                    prob = trellis[n - 1, fn_minus_2, fn_minus_1] + trans_prob + emission_prob + penalty
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_finger = fn_minus_2

                trellis[n, fn_minus_1, fn] = max_prob
                backpointers[n, fn_minus_1, fn] = best_prev_finger

    best_path = np.zeros(num_notes, dtype=np.int8)
    if num_notes > 1:
        prob_slice = trellis[num_notes - 1, :, :]
        flat_idx = np.argmax(prob_slice)
        last_f1 = flat_idx // num_fingers
        last_f2 = flat_idx % num_fingers

        best_path[num_notes - 1] = last_f2 + 1
        best_path[num_notes - 2] = last_f1 + 1

        for n in range(num_notes - 3, -1, -1):
            prev_f_idx = backpointers[n + 2, last_f1, last_f2]
            best_path[n] = prev_f_idx + 1
            last_f2 = last_f1
            last_f1 = prev_f_idx

    return best_path
