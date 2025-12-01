import numpy as np
import numba as nb
from typing import Tuple
import re

# Global Constant for unknown/invalid fingers
FINGER_UNKNOWN = -999

# Global LUT: (128 pitches, 2 coordinates [x, y])
PITCH_TO_KEYPOS_LUT = np.zeros((128, 2), dtype=np.float32)

def _compute_pitch_to_keypos_lut():
    """
    Computes a LUT mapping MIDI pitch to a (x, y) coordinate system.
    - X is the lateral position in pseudo-mm.
    - Y is the vertical position (0 for white keys, 1 for black keys).
    Based on standard piano key dimensions.
    """
    # White key properties
    WHITE_KEY_WIDTH = 23.5  # pseudo-mm
    # Pitch class to white key index mapping (0=C, 1=D, etc.)
    PC_TO_WHITE_KEY = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}

    for pitch in range(128):
        octave = pitch // 12
        pc = pitch % 12  # Pitch Class (0-11 for C, C#, ..., B)

        if pc in PC_TO_WHITE_KEY:
            # This is a white key
            white_key_index = PC_TO_WHITE_KEY[pc]
            x_pos = octave * 7 * WHITE_KEY_WIDTH + white_key_index * WHITE_KEY_WIDTH
            y_pos = 0
        else:
            # This is a black key. Position it relative to the previous white key.
            prev_white_key_pc = pc - 1
            white_key_index = PC_TO_WHITE_KEY[prev_white_key_pc]
            # Black keys are positioned halfway between white keys
            x_pos = octave * 7 * WHITE_KEY_WIDTH + white_key_index * WHITE_KEY_WIDTH + (WHITE_KEY_WIDTH / 2.0)
            y_pos = 1

        PITCH_TO_KEYPOS_LUT[pitch, 0] = x_pos
        PITCH_TO_KEYPOS_LUT[pitch, 1] = y_pos

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

# Updated DTYPE to match usage. 
# Note: 'channel' stores the Hand Index (0=RH, 1=LH) from column 6.
NOTE_DTYPE = np.dtype([
    ('original_idx', np.int32), 
    ('ontime', np.float64), 
    ('offtime', np.float64),
    ('pitch_str', 'U10'), 
    ('pitch', np.int32), 
    ('velocity', np.int32),
    ('channel', np.int32), 
    ('finger_str', 'U20'), 
    ('finger', np.int32)
])

SITCH_REGEX = re.compile(r'([A-G])([#b+-]*)([0-9])')

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
    Parses a finger string (e.g., "4_1", "-3", "-5_-1") into a single integer.
    Handles substitution by taking the starting finger (first part).
    """
    try:
        # Take the part before any substitution marking (underscore)
        # "4_1" -> "4", "-5_-1" -> "-5"
        cleaned_str = finger_str.split('_')[0]
        finger_val = int(cleaned_str)
        
        # Clamp/Check validity: [-5, 5] excluding 0
        if 0 < finger_val <= 5:
            return finger_val
        if -5 <= finger_val < 0:
            return finger_val
    except (ValueError, IndexError):
        pass
    return FINGER_UNKNOWN

def load_pig_file(filepath: str) -> np.ndarray:
    """
    Robust line-by-line parser for PIG files.
    Enforces 8 columns per line to prevent data shifting.
    
    Columns:
    0: ID
    1: Onset
    2: Offset
    3: Note Name
    4: Onset Velocity
    5: Offset Velocity (Ignored)
    6: Hand Index (0=RH, 1=LH) -> stored in 'channel'
    7: Finger Index (can include substitution e.g. "1_2")
    """
    raw_notes = []
    
    with open(filepath, 'r') as f:
        line_num = 0
        for line in f:
            line_num += 1
            # 1. Strip comments and whitespace
            content = line.partition('//')[0].strip()
            
            if not content:
                continue
                
            tokens = content.split()
            
            # 2. Strict Structure Validation
            if len(tokens) != 8:
                raise ValueError(f"Line {line_num} malformed: expected 8 columns, got {len(tokens)}. Content: '{content}'")
            
            try:
                # 3. Parse fields
                original_idx = int(tokens[0])
                ontime       = float(tokens[1])
                offtime      = float(tokens[2])
                pitch_str    = tokens[3]
                pitch        = sitch_to_pitch(pitch_str)
                velocity     = int(tokens[4])
                # token[5] is offset velocity, ignored.
                hand_idx     = int(tokens[6]) 
                finger_str   = tokens[7]
                finger       = clean_finger_str(finger_str)
                
                raw_notes.append((
                    original_idx, ontime, offtime, pitch_str, pitch, 
                    velocity, hand_idx, finger_str, finger
                ))
                
            except Exception as e:
                raise ValueError(f"Parsing error on line {line_num}: {e}")

    # Convert to structured numpy array
    if not raw_notes:
        return np.zeros(0, dtype=NOTE_DTYPE)
        
    return np.array(raw_notes, dtype=NOTE_DTYPE)


def apply_time_dep_pitch_order(notes: np.ndarray, time_threshold: float = 0.03) -> np.ndarray:
    """
    Groups notes by onset (within 0.03s tolerance) and sorts them by Pitch ASCENDING.
    """
    if len(notes) == 0:
        return notes

    # Global sort by time
    notes = np.sort(notes, order=['ontime'], kind='stable')

    reordered_notes = []

    i = 0
    while i < len(notes):
        cluster_indices = [i]
        j = i + 1
        # Cluster simultaneous notes
        while j < len(notes) and abs(notes[j]['ontime'] - notes[j-1]['ontime']) < time_threshold:
            cluster_indices.append(j)
            j += 1

        cluster_notes = notes[cluster_indices]

        # Sort by pitch ASCENDING (Low -> High)
        sorted_indices = np.argsort(cluster_notes['pitch'], kind='stable')

        reordered_notes.extend(cluster_notes[sorted_indices])
        i = j

    return np.array(reordered_notes, dtype=notes.dtype)