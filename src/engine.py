import numpy as np
import numba
from numba import njit
from utils import generate_luts, EPSILON_CHORD, GAP_THRESHOLD
# On suppose que xml_parser est disponible dans votre environnement
from xml_parser import get_valid_mask, Hand

# --- CONSTANTES PHYSIQUES ---
# Géométrie de la main (en cm, relative au centre 'virtuel')
# Correspond aux valeurs : [Unused, Thumb, Index, Middle, Ring, Pinky]
# Note : Ces valeurs sont asymétriques, comme dans le code original.
FREST_R = np.array([0.0, -7.0, -2.8, 0.0, 2.8, 5.6], dtype=np.float64)

# Poids de confort pour chaque doigt
WEIGHTS = np.array([0.0, 1.1, 1.0, 1.1, 0.9, 0.8], dtype=np.float64)

# Facteur de pénalité pour touches noires
BFACTOR = np.array([0.0, 0.3, 1.0, 1.1, 0.8, 0.7], dtype=np.float64)

# Mapping des indices internes (0-4) vers les indices "humains" (1-5) pour l'accès aux tableaux ci-dessus
FINGERS = np.array([1, 2, 3, 4, 5], dtype=np.int8)


def preprocess_data(score):
    """
    Transforme les données brutes du score en segments analysables par Viterbi.
    Gère la géométrie (main gauche/droite) et la segmentation temporelle.
    """
    valid_mask = get_valid_mask(score)

    pitch = score.pitch[valid_mask]
    onset = score.onset[valid_mask]
    duration = score.duration[valid_mask]
    hand = score.hand[valid_mask]

    keypos_lut, is_black_lut = generate_luts()
    x_pos = keypos_lut[pitch]

    # Inversion de l'axe X pour la main gauche (symétrie miroir)
    left_hand_mask = (hand == Hand.LEFT)
    x_pos[left_hand_mask] = -x_pos[left_hand_mask]

    # Détection des accords
    chord_id = np.zeros(len(pitch), dtype=np.int32)
    if len(pitch) > 0:
        time_diffs = np.diff(onset)
        is_new_chord_start = np.concatenate(([True], time_diffs > EPSILON_CHORD))
        chord_id = np.cumsum(is_new_chord_start) - 1

    # Segmentation basée sur les silences (GAP_THRESHOLD)
    offset = onset + duration
    # Calcul du gap entre la fin de note i-1 et le début de note i
    if len(onset) > 1:
        gaps = onset[1:] - offset[:-1]
        segment_breaks = np.where(gaps > GAP_THRESHOLD)[0] + 1
    else:
        segment_breaks = np.array([], dtype=np.int64)

    # Découpage
    pitches = np.split(pitch, segment_breaks)
    onsets = np.split(onset, segment_breaks)
    durations = np.split(duration, segment_breaks)
    hands = np.split(hand, segment_breaks)
    x_poses = np.split(x_pos, segment_breaks)
    chord_ids = np.split(chord_id, segment_breaks)

    is_black = is_black_lut[pitch]
    is_blacks = np.split(is_black, segment_breaks)

    segments = []
    for p, o, d, h, x, cid, b in zip(pitches, onsets, durations, hands, x_poses, chord_ids, is_blacks):
        # On ne traite pas les segments vides
        if len(p) == 0:
            continue

        segment = {
            "pitch": p,
            "is_black": b,
            "onset": o,
            "duration": d,
            "hand": h, # Note: hand array is mostly for debug, LR logic is handled by x_pos sign
            "x_pos": x,
            "chord_id": cid
        }
        segments.append(segment)

    return segments


# --- NUMBA KERNELS ---

@njit(cache=True)
def _get_valid_finger_permutations(n_notes, out_array):
    """Génère les permutations valides de doigts pour un accord de n notes."""
    count = 0
    if n_notes == 1:
        for i0 in range(5):
            out_array[count, 0] = i0
            count += 1
    elif n_notes == 2:
        for i0 in range(5):
            for i1 in range(i0 + 1, 5):
                out_array[count, 0] = i0
                out_array[count, 1] = i1
                count += 1
    elif n_notes == 3:
        for i0 in range(5):
            for i1 in range(i0 + 1, 5):
                for i2 in range(i1 + 1, 5):
                    out_array[count, 0] = i0
                    out_array[count, 1] = i1
                    out_array[count, 2] = i2
                    count += 1
    elif n_notes == 4:
        for i0 in range(5):
            for i1 in range(i0 + 1, 5):
                for i2 in range(i1 + 1, 5):
                    for i3 in range(i2 + 1, 5):
                        out_array[count, 0] = i0
                        out_array[count, 1] = i1
                        out_array[count, 2] = i2
                        out_array[count, 3] = i3
                        count += 1
    elif n_notes == 5:
        out_array[count, :5] = np.array([0, 1, 2, 3, 4], dtype=np.int8)
        count += 1
    return count

@njit(cache=True)
def _count_valid_finger_permutations(n_notes):
    """Renvoie le nombre de combinaisons possibles C(5, n)."""
    if n_notes < 0 or n_notes > 5: return 0
    if n_notes == 0: return 1
    if n_notes == 1: return 5
    if n_notes == 2: return 10
    if n_notes == 3: return 10
    if n_notes == 4: return 5
    if n_notes == 5: return 1
    return 0

@njit(cache=True)
def core_viterbi(
    x_pos: np.ndarray,
    onset: np.ndarray,
    duration: np.ndarray,
    is_black: np.ndarray,
    event_start_indices: np.ndarray,
    n_notes_per_event: np.ndarray,
    state_indices: np.ndarray,
    all_fingerings: np.ndarray,
    frest: np.ndarray
) -> np.ndarray:

    n_events = len(n_notes_per_event)
    if n_events == 0:
        return np.zeros(0, dtype=np.int8)

    total_states = state_indices[-1]

    # DP Tables
    dp = np.full(total_states, np.inf, dtype=np.float64)
    bp = np.full(total_states, -1, dtype=np.int32)

    # Init premier état à 0
    dp[state_indices[0]:state_indices[1]] = 0.0

    # Ajustement de la taille de main
    frest = FREST * hand_factor
    weights = WEIGHTS
    bfactor = BFACTOR

    # Boucle temporelle (Viterbi Forward)
    for t in range(1, n_events):
        prev_start, prev_end = state_indices[t-1], state_indices[t]
        curr_start, curr_end = state_indices[t], state_indices[t+1]

        # Indices des notes dans le tableau plat
        prev_event_note_start = event_start_indices[t-1]
        curr_event_note_start = event_start_indices[t]

        # Données de l'événement précédent
        prev_perms = all_fingerings[prev_start:prev_end, :n_notes_per_event[t-1]]
        prev_event_x = x_pos[prev_event_note_start : prev_event_note_start + n_notes_per_event[t-1]]
        prev_event_d = duration[prev_event_note_start : prev_event_note_start + n_notes_per_event[t-1]]
        prev_event_b = is_black[prev_event_note_start : prev_event_note_start + n_notes_per_event[t-1]]

        # Données de l'événement courant
        curr_perms = all_fingerings[curr_start:curr_end, :n_notes_per_event[t]]
        curr_event_x = x_pos[curr_event_note_start : curr_event_note_start + n_notes_per_event[t]]
        curr_event_b = is_black[curr_event_note_start : curr_event_note_start + n_notes_per_event[t]]

        # Delta temps (avec petit buffer pour éviter div/0)
        time_delta = abs(onset[curr_event_note_start] - onset[prev_event_note_start]) + 0.1

        # Boucle sur les états précédents
        for i in range(prev_perms.shape[0]):
            pf_perm = prev_perms[i, :]
            prev_dp_cost = dp[prev_start + i]

            if prev_dp_cost == np.inf:
                continue

            # Boucle sur les états courants
            for j in range(curr_perms.shape[0]):
                cf_perm = curr_perms[j, :]

                # --- LOGIQUE MÉLODIQUE (Note simple -> Note simple) ---
                is_melodic = (n_notes_per_event[t-1] == 1 and n_notes_per_event[t] == 1)

                if is_melodic:
                    pf = pf_perm[0] # 0=Thumb, 4=Pinky (indices internes)
                    cf = cf_perm[0]

                    x_delta = curr_event_x[0] - prev_event_x[0]
                    prev_dur = prev_event_d[0]

                    # 1. INTERDICTIONS STRICTES (Hard Constraints)

                    # A. Croisement de doigts (sauf pouce)
                    if pf > 0 and cf > 0 and (cf - pf) * x_delta < 0:
                        continue

                    # B. Même doigt sur deux notes différentes (sautillement)
                    if pf == cf and abs(x_delta) > 0.001 and prev_dur < 4.0:
                        continue

                    # C. Pouce sur touche noire en montant
                    if cf == 0 and curr_event_b[0] == 1 and x_delta > 0:
                        continue

                    # D. Cas spécifique legacy : Index sur noire en descendant, venant d'ailleurs
                    if prev_event_b[0] == 1 and x_delta < 0 and cf > 0 and prev_dur < 2.0:
                        continue

                    # 2. CALCUL DU COÛT (Physique Corps Rigide)

                    prev_anchor_x = prev_event_x[0]
                    curr_anchor_x = curr_event_x[0]
                    is_black_key = curr_event_b[0]

                    # Rigid Body Prediction
                    predicted_x = (frest[FINGERS[cf]] - frest[FINGERS[pf]]) + prev_anchor_x
                    distance = abs(curr_anchor_x - predicted_x)

                    # Soft Costs (Croisement pouce)
                    dx = curr_anchor_x - prev_anchor_x
                    df = cf - pf
                    is_cross = (dx * df < 0)

                    crossing_cost = 0.0
                    if is_cross:
                        if cf == 0: # Pouce passe dessous
                            crossing_cost = 2.0
                        elif pf == 0: # Pouce passe dessus
                            crossing_cost = 2.0
                        else:
                            crossing_cost = 100.0 # Pénalité forte sécurité

                    velocity = distance / time_delta

                    # Poids
                    w = weights[FINGERS[cf]]
                    if is_black_key:
                        w *= bfactor[FINGERS[cf]]

                    # Pénalité pouce sur noire
                    if cf == 0 and is_black_key:
                        velocity *= 3.0

                    transition_cost = (velocity / w) + crossing_cost

                else:
                    # --- LOGIQUE ACCORDS ---
                    min_anchor_cost = np.inf
                    for anchor_idx in range(len(pf_perm)):
                        pf_anchor = pf_perm[anchor_idx]
                        anchor_note_x = prev_event_x[anchor_idx]

                        current_anchor_cost = 0.0
                        for k in range(len(cf_perm)):
                            cf = cf_perm[k]
                            target_note_x = curr_event_x[k]
                            is_black_key = curr_event_b[k]

                            predicted_x = (frest[FINGERS[cf]] - frest[FINGERS[pf_anchor]]) + anchor_note_x
                            distance = abs(target_note_x - predicted_x)

                            velocity = distance / time_delta

                            w = weights[FINGERS[cf]]
                            if is_black_key:
                                w *= bfactor[FINGERS[cf]]

                            if cf == 0 and is_black_key:
                                velocity *= 3.0

                            current_anchor_cost += velocity / w

                        if current_anchor_cost < min_anchor_cost:
                            min_anchor_cost = current_anchor_cost

                    transition_cost = min_anchor_cost

                # Mise à jour DP
                new_cost = prev_dp_cost + transition_cost
                if new_cost < dp[curr_start + j]:
                    dp[curr_start + j] = new_cost
                    bp[curr_start + j] = i

    # --- BACKTRACKING ---
    path = np.full(n_events, -1, dtype=np.int32)
    last_start, last_end = state_indices[n_events-1], state_indices[n_events]

    if last_end > last_start:
        best_end_idx = -1
        min_val = np.inf
        for k in range(last_end - last_start):
            val = dp[last_start + k]
            if val < min_val:
                min_val = val
                best_end_idx = k

        path[n_events-1] = best_end_idx

        for t in range(n_events - 2, -1, -1):
            curr_idx_in_dp = state_indices[t+1] + path[t+1]
            path[t] = bp[curr_idx_in_dp]

    # --- RECONSTRUCTION DU RÉSULTAT (0-BASED) ---
    final_fingering = np.zeros(len(x_pos), dtype=np.int8)

    for t in range(n_events):
        if path[t] != -1:
            event_note_start = event_start_indices[t]
            event_idx = state_indices[t] + path[t]
            num_notes = n_notes_per_event[t]

            # Récupération du doigté (indices internes 0..4)
            fingering_perm = all_fingerings[event_idx, :num_notes]

            # --- CORRECTION ICI : On laisse en 0-based pour compatibilité avec test_engine.py ---
            for k in range(num_notes):
                final_fingering[event_note_start + k] = fingering_perm[k]
                # Pas de +1 ici ! Le script de test fait déjà +1 pour l'affichage.

    return final_fingering


def prepare_viterbi_inputs(segment):
    """Prépare les tableaux NumPy plats requis par le kernel Numba."""
    x_pos = segment['x_pos']
    onset = segment['onset']
    duration = segment['duration']
    is_black = segment['is_black']
    chord_id = segment['chord_id']

    unique_chord_ids = np.unique(chord_id)
    n_events = len(unique_chord_ids)

    event_start_indices = np.searchsorted(chord_id, unique_chord_ids)

    # Calcul rapide notes par event
    diff_indices = np.concatenate((event_start_indices, [len(chord_id)]))
    n_notes_per_event = np.diff(diff_indices).astype(np.int32)

    n_states = np.zeros(n_events, dtype=np.int32)
    for i in range(n_events):
        n_states[i] = _count_valid_finger_permutations(n_notes_per_event[i])

    if n_events > 0 and np.any(n_states == 0):
        return None

    state_indices = np.zeros(n_events + 1, dtype=np.int32)
    state_indices[1:] = np.cumsum(n_states)
    total_states = state_indices[-1]

    max_notes_in_chord = np.max(n_notes_per_event) if len(n_notes_per_event) > 0 else 0
    all_fingerings = np.full((total_states, max_notes_in_chord), -1, dtype=np.int8)

    for i in range(n_events):
        n = n_notes_per_event[i]
        start = state_indices[i]
        end = state_indices[i+1]
        _get_valid_finger_permutations(n, all_fingerings[start:end])

    return {
        "x_pos": x_pos,
        "onset": onset,
        "duration": duration,
        "is_black": is_black,
        "event_start_indices": event_start_indices,
        "n_notes_per_event": n_notes_per_event,
        "state_indices": state_indices,
        "all_fingerings": all_fingerings,
    }


def find_fingerings(noteseq, side="right", size='M', start_measure=0, nmeasures=1000):
    """
    Fonction principale.
    Retourne des indices 0-based (0=Pouce, 4=Auriculaire).
    """
    hand_factor = 0.82
    if size == "XXS": hand_factor = 0.33
    elif size == "XS": hand_factor = 0.46
    elif size == "S": hand_factor = 0.64
    elif size == "M": hand_factor = 0.82
    elif size == "L": hand_factor = 1.0
    elif size == "XL": hand_factor = 1.1
    elif size == "XXL": hand_factor = 1.2

    if side == "left":
        FREST = -FREST_R[::-1]
    else:
        FREST = FREST_R

    segments = preprocess_data(noteseq)
    full_fingerings = []

    for seg in segments:
        inputs = prepare_viterbi_inputs(seg)
        if inputs is None:
            full_fingerings.append(np.zeros(len(seg['x_pos']), dtype=np.int8))
            continue

        fingering = core_viterbi(
            inputs['x_pos'],
            inputs['onset'],
            inputs['duration'],
            inputs['is_black'],
            inputs['event_start_indices'],
            inputs['n_notes_per_event'],
            inputs['state_indices'],
            inputs['all_fingerings'],
            FREST * hand_factor
        )
        full_fingerings.append(fingering)

    if len(full_fingerings) > 0:
        result = np.concatenate(full_fingerings)
        if side == "left":
            # Remap fingers for the left hand
            valid_fingers = result != -1
            result[valid_fingers] = np.array([4, 3, 2, 1, 0], dtype=np.int8)[result[valid_fingers]]
        return result
    else:
        return np.array([], dtype=np.int8)
