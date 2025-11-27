import numpy as np
import numba as nb
from typing import Tuple
import re
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


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
    ('channel', np.int32), ('finger_str', 'U20'), ('finger', np.int32)
])

SITCH_REGEX = re.compile(r'([A-G])([#b+-]*)([0-9])')
COMMENT_REGEX = re.compile(r'//.*|#.*')

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

def clean_finger_str(finger_str: str) -> int:
    """
    Parses a finger string (e.g., "4_1", "-3") into a single integer.
    This replicates the C++ `GetKeyPressFingerNum` and `ConvertFingerNumberToInt` logic.
    """
    try:
        # Take the part before any substitution marking
        cleaned_str = finger_str.split('_')[0]
        finger_val = int(cleaned_str)
        # Clamp the values to the valid range [-5, 5], excluding 0.
        if 0 < finger_val <= 5:
            return finger_val
        if -5 <= finger_val < 0:
            return finger_val
    except (ValueError, IndexError):
        pass # Fall through to return 0 if parsing fails
    return 0 # Default/invalid


def load_pig_file(filepath: str) -> np.ndarray:
    """
    Robust, line-by-line parser for PIG files that correctly handles the 8-column format.
    """
    notes_list = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = COMMENT_REGEX.sub('', line).strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) != 8:
                logging.warning(f"Skipping malformed line {line_num} in {filepath}: "
                                f"Expected 8 columns, found {len(tokens)}.")
                continue

            try:
                # PIG format: idx, ontime, offtime, pitch_str, onvel, offvel, channel, finger_str
                original_idx = int(float(tokens[0]))
                ontime       = float(tokens[1])
                offtime      = float(tokens[2])
                pitch_str    = tokens[3]
                pitch        = sitch_to_pitch(pitch_str)
                velocity     = int(tokens[4]) # onvel
                channel      = int(tokens[6])
                finger_str   = tokens[7]
                finger       = clean_finger_str(finger_str)

                notes_list.append((original_idx, ontime, offtime, pitch_str, pitch, velocity, channel, finger_str, finger))

            except (ValueError, IndexError) as e:
                logging.warning(f"Skipping malformed note record on line {line_num} in {filepath}: {e}")
                continue

    return np.array(notes_list, dtype=NOTE_DTYPE)


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
        # Corrected logic: compare with previous note in the original sequence
        while j < len(notes) and abs(notes[j]['ontime'] - notes[j-1]['ontime']) < time_threshold:
            cluster_indices.append(j)
            j += 1

        cluster_notes = notes[cluster_indices]
        # C++ sorts by -pitch descending, which is pitch ascending.
        sorted_indices = np.argsort(cluster_notes['pitch'], kind='stable')
        reordered_notes.extend(cluster_notes[sorted_indices])
        i = j

    return np.array(reordered_notes, dtype=notes.dtype)


def filter_notes_by_hand(notes: np.ndarray, hand) -> np.ndarray:
    """
    Filters notes for a specific hand.
    'hand' can be 0 or 'right' for the right hand, 1 or 'left' for the left hand.
    """
    if notes.shape[0] == 0:
        return np.array([], dtype=notes.dtype)

    # In the original data, positive is right, negative is left.
    # We must also handle notes that have no hand assigned (finger == 0),
    # which should be excluded from both.
    if hand == 0 or hand == 'right':
        return notes[notes['finger'] > 0]
    elif hand == 1 or hand == 'left':
        return notes[notes['finger'] < 0]
    else:
        return np.array([], dtype=notes.dtype)
