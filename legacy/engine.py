# engineulegacy_2.py

from . import utils as utils


class Hand:
    def __init__(self, noteseq, side="right", size='M'):
        self.LR = side
        self.noteseq = noteseq
        self.fingerseq = []
        self.size = size

        self.frest = [None, -7.0, -2.8, 0.0, 2.8, 5.6]
        self.weights = [None, 1.1, 1.0, 1.1, 0.9, 0.8]
        self.bfactor = [None, 0.3, 1.0, 1.1, 0.8, 0.7]

        hf = utils.handSizeFactor(size)
        self.frest = [None if f is None else f * hf for f in self.frest]
        self.hf = hf
        self.cfps = list(self.frest)

    def set_fingers_positions(self, finger, note_x):
        if finger < 1 or finger > 5:
            return

        current_frest = [-f if f is not None else None for f in self.frest] if self.LR == "left" else list(self.frest)
        anchor = current_frest[finger]
        
        if anchor is not None:
            self.cfps = [None if current_frest[j] is None else (current_frest[j] - anchor) + note_x
                         for j in range(6)]

    def _is_forbidden_transition(self, fa, fb, na, nb):
        xba = nb.x - na.x

        if not na.isChord and not nb.isChord:
            if fa == fb and xba and na.duration < 4:
                return True
            if fa > 1:
                if fb > 1 and (fb - fa) * xba < 0:
                    return True
                if fb == 1 and nb.isBlack and xba > 0:
                    return True
            elif na.isBlack and xba < 0 and fb > 1 and na.duration < 2:
                return True

        elif na.isChord and nb.isChord and na.chordID == nb.chordID:
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

        return False

    def _compute_transition_cost(self, prev_f, curr_f, note_prev, note_curr):
        anchor = self.frest[prev_f]
        finger_pos = (self.frest[curr_f] - anchor) + note_prev.x
        distance = abs(note_curr.x - finger_pos)
        time_delta = abs(note_curr.time - note_prev.time) + 0.1
        velocity = distance / time_delta
        weight = self.weights[curr_f] * (self.bfactor[curr_f] if note_curr.isBlack else 1.0)
        return velocity / weight

    def _filter_active_notes(self, start_measure, nmeasures):
        start_measure = 0 if start_measure == 1 else start_measure
        active_notes = []
        active_indices = []
        
        for i, note in enumerate(self.noteseq):
            if note.measure and not (start_measure <= note.measure <= start_measure + nmeasures):
                continue
            active_notes.append(note)
            active_indices.append(i)
        
        return active_notes, active_indices

    def _run_viterbi(self, active_notes):
        N = len(active_notes)
        dp = [{f: (0.0, None) for f in range(1, 6)}]

        for i in range(1, N):
            prev_note = active_notes[i-1]
            curr_note = active_notes[i]
            curr_layer = {}

            for f_curr in range(1, 6):
                best_cost = float('inf')
                best_prev = None

                for f_prev, (prev_accumulated_cost, _) in dp[i-1].items():
                    if self._is_forbidden_transition(f_prev, f_curr, prev_note, curr_note):
                        continue

                    move_cost = self._compute_transition_cost(f_prev, f_curr, prev_note, curr_note)
                    total_cost = prev_accumulated_cost + move_cost

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_prev = f_prev

                if best_prev is not None:
                    curr_layer[f_curr] = (best_cost, best_prev)

            if not curr_layer:
                prev_best = min(dp[i-1].items(), key=lambda x: x[1][0])
                fallback_prev = prev_best[0]
                prev_cost = prev_best[1][0]
                curr_layer = {f: (prev_cost + 10000.0, fallback_prev) for f in range(1, 6)}

            dp.append(curr_layer)

        return dp

    def _backtrack_path(self, dp, N):
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

    def _apply_fingerings(self, path, active_notes, active_indices):
        for idx_in_path, real_idx in enumerate(active_indices):
            finger = path[idx_in_path]
            note = self.noteseq[real_idx]
            
            note.fingering = finger
            
            if idx_in_path > 0:
                note.cost = self._compute_transition_cost(path[idx_in_path-1], finger, 
                                                          active_notes[idx_in_path-1], note)
            else:
                note.cost = 0
            
            note_x_for_viz = -note.x if self.LR == "left" else note.x
            self.set_fingers_positions(finger, note_x_for_viz)
            self.fingerseq.append(list(self.cfps))

    def generate(self, start_measure=0, nmeasures=1000):
        original_x_values = []
        if self.LR == "left":
            for note in self.noteseq:
                original_x_values.append(note.x)
                note.x = -note.x

        active_notes, active_indices = self._filter_active_notes(start_measure, nmeasures)

        if len(active_notes) == 0:
            if self.LR == "left":
                for i, x in enumerate(original_x_values):
                    self.noteseq[i].x = x
            return self.noteseq

        dp = self._run_viterbi(active_notes)
        path = self._backtrack_path(dp, len(active_notes))
        self._apply_fingerings(path, active_notes, active_indices)

        if self.LR == "left":
            for i, x in enumerate(original_x_values):
                self.noteseq[i].x = x

        return self.noteseq

def find_fingerings(noteseq, side, size, depth=9, autodepth=True, lyrics=False, start_measure=0, nmeasures=1000):
    hand = Hand(noteseq, side, size)
    return hand.generate(start_measure, nmeasures)
