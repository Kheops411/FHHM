import numpy as np
import numba as nb
from typing import Tuple
import re

# Global LUT: (128 pitches, 2 coordinates [x, y])
PITCH_TO_KEYPOS_LUT = np.zeros((128, 2), dtype=np.int16)

def _compute_pitch_to_keypos_lut():
    for pitch in range(128):
        pc = pitch % 12
        octave = (pitch // 12) - 1
        x = 0
        if pc in (0, 1): x = 0
        elif pc in (2, 3): x = 1
        elif pc == 4: x = 2
        elif pc in (5, 6): x = 3
        elif pc in (7, 8): x = 4
        elif pc in (9, 10): x = 5
        elif pc == 11: x = 6
        x += 7 * (octave - 4)
        y = 1
        if pc in (0, 2, 4, 5, 7, 9, 11): y = 0
        PITCH_TO_KEYPOS_LUT[pitch, 0] = x
        PITCH_TO_KEYPOS_LUT[pitch, 1] = y

_compute_pitch_to_keypos_lut()

def pitch_to_keypos(midi_pitch: int) -> Tuple[int, int]:
    if not (0 <= midi_pitch < 128):
        raise ValueError(f"Pitch {midi_pitch} out of bounds")
    row = PITCH_TO_KEYPOS_LUT[midi_pitch]
    return int(row[0]), int(row[1])

def subtract_keypos(kp1: Tuple[int,int], kp2: Tuple[int,int]) -> Tuple[int,int]:
    return (kp1[0] - kp2[0], kp1[1] - kp2[1])

@nb.njit(cache=True)
def pitch_to_keypos_numba(midi_pitch: int, lut: np.ndarray) -> np.ndarray:
    return lut[midi_pitch]

@nb.njit(cache=True)
def subtract_keypos_numba(x1, y1, x2, y2):
    return x1 - x2, y1 - y2

@nb.njit(cache=True)
def lattice_delta_to_index(dx: int, dy: int, width_x: int = 15) -> int:
    if dx < -width_x: dx = -width_x
    if dx > width_x:  dx = width_x
    return 3 * (dx + width_x) + dy + 1

# --- Data Parsing & Ordering ---

NOTE_DTYPE = np.dtype([
    ('original_idx', np.int32), ('ontime', np.float64), ('offtime', np.float64),
    ('pitch_str', 'U10'), ('pitch', np.int32), ('velocity', np.int32),
    ('channel', np.int32), ('finger_str', 'U20')
])

SITCH_REGEX = re.compile(r'([A-G])([#b+-]*)([0-9])')
COMMENT_REGEX = re.compile(r'(//|#).*')

def sitch_to_pitch(sitch: str) -> int:
    if sitch in ("R", "rest"): return -1
    match = SITCH_REGEX.match(sitch)
    if not match: raise ValueError(f"Invalid pitch string: {sitch}")
    note_name, accidentals, octave_str = match.groups()
    p_rel = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71}[note_name]
    octave = int(octave_str)
    pitch = p_rel + (octave - 4) * 12
    acc_val = 0
    for char in accidentals:
        if char in ('#', '+'): acc_val += 1
        elif char in ('b', '-'): acc_val -= 1
    return pitch + acc_val

def load_pig_file(filepath: str) -> np.ndarray:
    """
    Robust, stream-like parser for PIG files.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    clean_content = COMMENT_REGEX.sub('', content)
    tokens = clean_content.split()
    num_tokens = len(tokens)
    
    # C++ reads 8 tokens: ID, ontime, offtime, sitch, onvel, offvel, channel, fingerNum
    num_notes = num_tokens // 8
    
    notes = np.zeros(num_notes, dtype=NOTE_DTYPE)
    if num_notes == 0:
        return notes

    try:
        for i in range(num_notes):
            base = i * 8
            notes[i]['original_idx'] = int(tokens[base])
            notes[i]['ontime']       = float(tokens[base + 1])
            notes[i]['offtime']      = float(tokens[base + 2])
            notes[i]['pitch_str']    = tokens[base + 3]
            notes[i]['pitch']        = sitch_to_pitch(tokens[base + 3])
            notes[i]['velocity']     = int(tokens[base + 4]) # onvel
            # offvel (tokens[base + 5]) is ignored in our dtype
            notes[i]['channel']      = int(tokens[base + 6])
            notes[i]['finger_str']   = tokens[base + 7]
            
    except (ValueError, IndexError) as e:
        raise ValueError(f"Parsing error at note index {i} (token base {base}): {e}")

    return notes


def apply_time_dep_pitch_order(notes: np.ndarray, time_threshold: float = 0.03) -> np.ndarray:
    """
    Replicates the C++ `TimeDepPitchOrder` logic exactly.
    """
    if len(notes) == 0:
        return notes

    reordered_notes = []

    i = 0
    while i < len(notes):
        cluster_indices = [i]
        j = i + 1
        while j < len(notes) and abs(notes[j]['ontime'] - notes[j-1]['ontime']) < time_threshold:
            cluster_indices.append(j)
            j += 1

        cluster_notes = notes[cluster_indices]
        sorted_indices = np.argsort(cluster_notes['pitch'], kind='stable')[::-1]
        reordered_notes.extend(cluster_notes[sorted_indices])
        i = j

    return np.array(reordered_notes, dtype=notes.dtype)


def filter_notes_by_hand(notes: np.ndarray, hand: int) -> np.ndarray:
    """
    Filters notes based on the C++ `SelectHandByFingerNum` logic.
    """
    if notes.shape[0] == 0:
        return np.array([], dtype=notes.dtype)

    if hand == 0: # Right Hand
        mask = np.array([not f.startswith('-') for f in notes['finger_str']])
        return notes[mask]
    else: # Left Hand
        mask = np.array([f.startswith('-') for f in notes['finger_str']])
        return notes[mask]
