import numpy as np
from python import utils

class HMMTrainer:
    def __init__(self, order=2, epsilon=1e-3, tr_sym=False, rf_sym=False):
        """
        Args:
            order (int): HMM Order (2 or 3).
            epsilon (float): Laplace smoothing factor.
            tr_sym (bool): Time Reversal Symmetry (Augment data by reversing time).
            rf_sym (bool): Reflection Symmetry (Augment data by mirroring hands).
        """
        assert order in [2, 3]
        self.order = order
        self.tr_sym = tr_sym
        self.rf_sym = rf_sym
        
        self.width_x = 15
        self.n_out = 3 * (2 * self.width_x + 1)

        # Initialize counts with epsilon
        self.ini_prob = np.full((2, 5), epsilon)
        self.tr_prob1 = np.full((2, 5, 5), epsilon)
        self.out_prob1 = np.full((2, 5, 5, self.n_out), epsilon)

        self.tr_prob2 = np.full((2, 5, 5, 5), epsilon)
        self.out_prob2 = np.full((2, 5, 5, self.n_out), epsilon)

        if self.order == 3:
            self.tr_prob3 = np.full((2, 5, 5, 5, 5), epsilon)
            self.out_prob3 = np.full((2, 5, 5, self.n_out), epsilon)

    def train(self, score_files: list):
        for filepath in score_files:
            notes = utils.load_pig_file(filepath)
            # CRITICAL: Must use the exact same ordering as C++ to compute valid transitions
            ordered_notes = utils.apply_time_dep_pitch_order(notes)

            for hand in [0, 1]: # 0=RH, 1=LH
                hand_notes = utils.filter_notes_by_hand(ordered_notes, hand)
                n_notes = len(hand_notes)

                if n_notes < self.order:
                    continue

                # --- Initial Probabilities ---
                # C++ implementation typically only counts the very first note of the piece
                # for ini_prob, even with symmetries enabled.
                f0 = hand_notes[0]['finger']
                if 1 <= abs(f0) <= 5:
                    self.ini_prob[hand, abs(f0) - 1] += 1

                # --- Transitions & Emissions ---
                for n in range(1, n_notes):
                    # Current note data
                    fn = abs(hand_notes[n]['finger'])
                    fn1 = abs(hand_notes[n-1]['finger'])
                    
                    if not (1 <= fn <= 5 and 1 <= fn1 <= 5):
                        continue
                    
                    # Indices (0-based)
                    idx_fn = fn - 1
                    idx_fn1 = fn1 - 1

                    # Lattice Delta (n - n1)
                    kp_n = utils.pitch_to_keypos(hand_notes[n]['pitch'])
                    kp_n1 = utils.pitch_to_keypos(hand_notes[n-1]['pitch'])
                    dx1, dy1 = utils.subtract_keypos(kp_n, kp_n1)

                    # 1. Standard Update
                    self._update_order1(hand, idx_fn1, idx_fn, dx1, dy1)
                    
                    # 2. Reflection Symmetry (Mirror Hand)
                    if self.rf_sym:
                        # Hand: Opposite (1-h)
                        # Delta: Invert X only (-dx, dy)
                        self._update_order1(1 - hand, idx_fn1, idx_fn, -dx1, dy1)
                    
                    # 3. Time Reversal Symmetry (Backward)
                    if self.tr_sym:
                        # Direction: Current -> Previous
                        # Delta: Full Inversion (-dx, -dy)
                        self._update_order1(hand, idx_fn, idx_fn1, -dx1, -dy1)
                        
                        # 4. TR + RF Combined
                        if self.rf_sym:
                            # Hand: Opposite
                            # Direction: Current -> Previous
                            # Delta: Mirror X of Reversed Time (-(-dx), -dy) -> (dx, -dy)
                            self._update_order1(1 - hand, idx_fn, idx_fn1, dx1, -dy1)

                    # --- Order 2 Updates ---
                    if n >= 2:
                        fn2 = abs(hand_notes[n-2]['finger'])
                        if 1 <= fn2 <= 5:
                            idx_fn2 = fn2 - 1
                            
                            kp_n2 = utils.pitch_to_keypos(hand_notes[n-2]['pitch'])
                            dx2, dy2 = utils.subtract_keypos(kp_n, kp_n2) # Delta n - n2

                            # Standard
                            self._update_order2(hand, idx_fn2, idx_fn1, idx_fn, dx2, dy2)
                            
                            # Reflection
                            if self.rf_sym:
                                self._update_order2(1 - hand, idx_fn2, idx_fn1, idx_fn, -dx2, dy2)
                            
                            # Time Reversal (n -> n-1 -> n-2)
                            if self.tr_sym:
                                # Note: In C++ TRSym for order 2/3 updates transition: curr -> prev -> prev2
                                # Delta must be reversed: n-2 minus n = -(n - n-2)
                                self._update_order2(hand, idx_fn, idx_fn1, idx_fn2, -dx2, -dy2)

                                if self.rf_sym:
                                     self._update_order2(1 - hand, idx_fn, idx_fn1, idx_fn2, dx2, -dy2)

                    # --- Order 3 Updates ---
                    if self.order == 3 and n >= 3:
                        fn3 = abs(hand_notes[n-3]['finger'])
                        if 1 <= fn3 <= 5:
                            idx_fn3 = fn3 - 1
                            
                            kp_n3 = utils.pitch_to_keypos(hand_notes[n-3]['pitch'])
                            dx3, dy3 = utils.subtract_keypos(kp_n, kp_n3)

                            # Standard
                            self._update_order3(hand, idx_fn3, idx_fn2, idx_fn1, idx_fn, dx3, dy3)
                            
                            # Reflection
                            if self.rf_sym:
                                self._update_order3(1 - hand, idx_fn3, idx_fn2, idx_fn1, idx_fn, -dx3, dy3)
                            
                            # Time Reversal
                            if self.tr_sym:
                                self._update_order3(hand, idx_fn, idx_fn1, idx_fn2, idx_fn3, -dx3, -dy3)
                                
                                if self.rf_sym:
                                    self._update_order3(1 - hand, idx_fn, idx_fn1, idx_fn2, idx_fn3, dx3, -dy3)

        self._normalize_counts()

    def _update_order1(self, h, i_prev, i_curr, dx, dy):
        """Helper to increment Order 1 tables."""
        self.tr_prob1[h, i_prev, i_curr] += 1
        lat_idx = utils.lattice_delta_to_index(dx, dy)
        self.out_prob1[h, i_prev, i_curr, lat_idx] += 1

    def _update_order2(self, h, i_prev2, i_prev1, i_curr, dx, dy):
        """Helper to increment Order 2 tables."""
        self.tr_prob2[h, i_prev2, i_prev1, i_curr] += 1
        lat_idx = utils.lattice_delta_to_index(dx, dy)
        self.out_prob2[h, i_prev2, i_curr, lat_idx] += 1

    def _update_order3(self, h, i_prev3, i_prev2, i_prev1, i_curr, dx, dy):
        """Helper to increment Order 3 tables."""
        self.tr_prob3[h, i_prev3, i_prev2, i_prev1, i_curr] += 1
        lat_idx = utils.lattice_delta_to_index(dx, dy)
        self.out_prob3[h, i_prev3, i_curr, lat_idx] += 1

    def _normalize_counts(self):
        """
        Converts counts to probabilities in place.
        """
        # Initial Prob
        for h in range(2):
            s = np.sum(self.ini_prob[h, :])
            if s > 0: self.ini_prob[h, :] /= s

            # Order 1
            for i in range(5):
                s_tr = np.sum(self.tr_prob1[h, i, :])
                if s_tr > 0: self.tr_prob1[h, i, :] /= s_tr
                
                for j in range(5):
                    # Output 1
                    s_out = np.sum(self.out_prob1[h, i, j, :])
                    if s_out > 0: self.out_prob1[h, i, j, :] /= s_out
                    
                    # Transition 2
                    s_tr2 = np.sum(self.tr_prob2[h, i, j, :])
                    if s_tr2 > 0: self.tr_prob2[h, i, j, :] /= s_tr2
                    
                    # Output 2
                    s_out2 = np.sum(self.out_prob2[h, i, j, :])
                    if s_out2 > 0: self.out_prob2[h, i, j, :] /= s_out2

                    if self.order == 3:
                        for k in range(5):
                            # Transition 3
                            s_tr3 = np.sum(self.tr_prob3[h, i, j, k, :])
                            if s_tr3 > 0: self.tr_prob3[h, i, j, k, :] /= s_tr3
                        
                        # Output 3
                        s_out3 = np.sum(self.out_prob3[h, i, j, :])
                        if s_out3 > 0: self.out_prob3[h, i, j, :] /= s_out3

    def save_parameters(self, filepath):
        # (Le code de sauvegarde reste identique à votre version précédente)
        # Il est juste important que les matrices soient bien remplies avant.
        with open(filepath, 'w') as f:
            f.write("### Initial Prob Right\n")
            f.write('\t'.join(map(str, self.ini_prob[0])) + '\n')
            f.write("### Initial Prob Left\n")
            f.write('\t'.join(map(str, self.ini_prob[1])) + '\n')

            f.write("### Transition Prob Right\n")
            for row in self.tr_prob1[0]:
                f.write('\t'.join(map(str, row)) + '\n')
            f.write("### Transition Prob Left\n")
            for row in self.tr_prob1[1]:
                f.write('\t'.join(map(str, row)) + '\n')

            f.write("### Transition Prob 2nd Right\n")
            for i in range(5):
                for j in range(5):
                    f.write('\t'.join(map(str, self.tr_prob2[0, i, j])) + '\n')
            f.write("### Transition Prob 2nd Left\n")
            for i in range(5):
                for j in range(5):
                    f.write('\t'.join(map(str, self.tr_prob2[1, i, j])) + '\n')

            if self.order == 3:
                f.write("### Transition Prob 3rd Right\n")
                for i in range(5):
                    for j in range(5):
                        for k in range(5):
                            f.write('\t'.join(map(str, self.tr_prob3[0, i, j, k])) + '\n')
                f.write("### Transition Prob 3rd Left\n")
                for i in range(5):
                    for j in range(5):
                        for k in range(5):
                            f.write('\t'.join(map(str, self.tr_prob3[1, i, j, k])) + '\n')

            f.write("### Output Prob Right\n")
            for i in range(5):
                for j in range(5):
                    f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob1[0, i, j])) + '\n')
            f.write("### Output Prob Left\n")
            for i in range(5):
                for j in range(5):
                    f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob1[1, i, j])) + '\n')

            f.write("### Output Prob 2nd Right\n")
            for i in range(5):
                for j in range(5):
                    f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob2[0, i, j])) + '\n')
            f.write("### Output Prob 2nd Left\n")
            for i in range(5):
                for j in range(5):
                    f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob2[1, i, j])) + '\n')

            if self.order == 3:
                f.write("### Output Prob 3rd Right\n")
                for i in range(5):
                    for j in range(5):
                        f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob3[0, i, j])) + '\n')
                f.write("### Output Prob 3rd Left\n")
                for i in range(5):
                    for j in range(5):
                        f.write(f"{i+1} {j+1}\t" + '\t'.join(map(str, self.out_prob3[1, i, j])) + '\n')