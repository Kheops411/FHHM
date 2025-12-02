import numpy as np
from structures import ScoreData, HAND_RIGHT, HAND_LEFT
from geometry import get_key_geometry

class HandSOA:
    def __init__(self, x, onsets, durations, is_black, event_ids, is_chord, side="right", hf=0.82):
        self.LR = side
        self.x = x
        self.onsets = onsets
        self.durations = durations
        self.is_black = is_black
        self.event_ids = event_ids
        self.is_chord = is_chord # New array

        self.frest = [None, -7.0, -2.8, 0.0, 2.8, 5.6]
        self.weights = [None, 1.1, 1.0, 1.1, 0.9, 0.8]
        self.bfactor = [None, 0.3, 1.0, 1.1, 0.8, 0.7]

        # Apply Hand Size Factor (Legacy Default 'M' = 0.82)
        self.hf = hf
        self.frest = [None if f is None else f * self.hf for f in self.frest]

    def _is_forbidden_transition(self, fa, fb, i):
        na_x = self.x[i-1]
        nb_x = self.x[i]
        xba = nb_x - na_x

        # Legacy Logic Port:
        # Case 1: Pure Melody (Neither is chord)
        if not self.is_chord[i-1] and not self.is_chord[i]:
            if fa == fb and xba != 0 and self.durations[i-1] < 4:
                return True
            if fa > 1:
                if fb > 1 and (fb - fa) * xba < 0:
                    return True
                if fb == 1 and self.is_black[i] and xba > 0:
                    return True
            elif self.is_black[i-1] and xba < 0 and fb > 1 and self.durations[i-1] < 2:
                return True

        # Case 2: Internal Chord Transition (Both are chord, SAME event ID)
        elif self.is_chord[i-1] and self.is_chord[i] and (self.event_ids[i] == self.event_ids[i-1]):
            axba = abs(xba) * (0.8 / self.hf)
            chord_rules = [
                (fa == fb),
                (fa < fb and self.LR == 'left'),
                (fa > fb and self.LR == 'right'),
                (axba > 5 and {fa, fb} == {3, 4}),
                (axba > 5 and {fa, fb} == {4, 5}),
                (axba > 6 and {fa, fb} == {2, 3}),
                (axba > 7 and {fa, fb} == {2, 4}),
                (axba > 8 and {fa, fb} == {3, 5}),
                (axba > 11 and {fa, fb} == {2, 5}),
                (axba > 12 and {fa, fb} == {1, 2}),
                (axba > 14 and {fa, fb} == {1, 3}),
                (axba > 16 and {fa, fb} == {1, 4})
            ]
            return any(chord_rules)

        # Case 3: Mixed (Melody -> Chord OR Chord -> Melody OR ChordA -> ChordB)
        # Legacy code calculates NO RULES for these transitions.
        # It implicitly returns False.

        return False

    def _compute_transition_cost(self, prev_f, curr_f, i):
        note_prev_x = self.x[i-1]
        note_curr_x = self.x[i]

        anchor = self.frest[prev_f]
        finger_pos = (self.frest[curr_f] - anchor) + note_prev_x
        distance = abs(note_curr_x - finger_pos)
        time_delta = abs(self.onsets[i] - self.onsets[i-1]) + 0.1
        velocity = distance / time_delta
        weight = self.weights[curr_f] * (self.bfactor[curr_f] if self.is_black[i] else 1.0)
        return velocity / weight

    def _run_viterbi(self):
        N = len(self.x)
        if N == 0:
            return []

        dp = [{f: (0.0, None) for f in range(1, 6)}]

        for i in range(1, N):
            curr_layer = {}
            for f_curr in range(1, 6):
                best_cost = float('inf')
                best_prev = None
                for f_prev, (prev_accumulated_cost, _) in dp[i-1].items():
                    if self._is_forbidden_transition(f_prev, f_curr, i):
                        continue

                    move_cost = self._compute_transition_cost(f_prev, f_curr, i)
                    total_cost = prev_accumulated_cost + move_cost

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_prev = f_prev

                if best_prev is not None:
                    curr_layer[f_curr] = (best_cost, best_prev)

            if not curr_layer: # Handle cases where all transitions are forbidden
                prev_best = min(dp[i-1].items(), key=lambda x: x[1][0])
                fallback_prev = prev_best[0]
                prev_cost = prev_best[1][0]
                curr_layer = {f: (prev_cost + 10000.0, fallback_prev) for f in range(1, 6)}

            dp.append(curr_layer)

        return dp

    def _backtrack_path(self, dp):
        N = len(dp)
        if N == 0:
            return []

        last_layer = dp[-1]
        best_end_finger = min(last_layer, key=lambda k: last_layer[k][0])

        path = [best_end_finger]
        curr_f = best_end_finger

        for i in range(N-1, 0, -1):
            _, parent = dp[i][curr_f]
            path.append(parent)
            curr_f = parent

        path.reverse()
        return path

    def generate(self):
        dp = self._run_viterbi()
        path = self._backtrack_path(dp)
        return path


def find_fingerings_soa(soa: ScoreData, hand_side: str) -> np.ndarray:
    """
    Finds fingerings for a given hand using a heuristic algorithm on ScoreData.
    """
    target_hand_int = HAND_RIGHT if hand_side == "right" else HAND_LEFT
    mask = (soa.hand == target_hand_int)

    # Extract data for the target hand
    pitches = soa.pitch[mask]
    onsets = soa.onset[mask]
    offsets = soa.offset[mask]
    event_ids = soa.event_id[mask]

    if len(pitches) == 0:
        return np.zeros(len(soa), dtype=np.int8)

    x, is_black = get_key_geometry(pitches)

    # For left hand, mirror the x coordinates
    if hand_side == 'left':
        x = -x

    # --- NORMALIZATION FIX START ---

    # 1. Space: MM -> CM
    x = x * 0.1

    # 2. Time: Seconds -> Quarter Notes (Approximate)
    # Legacy 'velocity' tuning implies 120BPM logic (1s = 2q).
    time_scale_factor = 2.0
    onsets = onsets * time_scale_factor
    offsets = offsets * time_scale_factor

    # --- NORMALIZATION FIX END ---

    durations = offsets - onsets

    # 3. FIX DURATION SCALING
    # Legacy adapter uses `duration * 4`. Your `durations` are currently `duration * 2`.
    # Multiply by 2.0 again to match legacy "ticks" logic.
    durations = durations * 2.0

    # 4. CALCULATE IS_CHORD
    # We need to know if a specific note is part of a chord to match legacy logic.
    # Count occurrences of each event_id.
    unique_ids, counts = np.unique(event_ids, return_counts=True)
    # Map counts back to the full array.
    # Optimization: Use a lookup array or dictionary if N is small, or just iteration for now.
    # Since N is small in context of piano pieces, a dict map is fast enough for Python layer.
    counts_map = dict(zip(unique_ids, counts))
    is_chord_arr = np.array([counts_map[eid] > 1 for eid in event_ids], dtype=bool)

    # Pass is_chord_arr and hf=0.82 to HandSOA
    hand_engine = HandSOA(x, onsets, durations, is_black, event_ids, is_chord_arr, side=hand_side, hf=0.82)
    finger_path = hand_engine.generate()

    # Create the output array of the same size as the original soa
    output_fingers = np.zeros(len(soa), dtype=np.int8)

    # Get the indices of the notes for the target hand
    hand_indices = np.where(mask)[0]

    # Place the calculated fingerings at the correct indices using vectorized assignment
    if len(finger_path) == len(hand_indices):
        output_fingers[hand_indices] = finger_path

    return output_fingers
