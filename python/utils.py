import numpy as np
import re
import sys

# Pre-compile the regex for performance
OCTAVE_REGEX = re.compile(r'(-?\d+)$')

def pitch_to_keypos(pitch: int) -> tuple[int, int]:
    """
    Converts a MIDI pitch to a keyboard lattice coordinate.
    """
    pc = pitch % 12
    octave = pitch // 12 - 1
    x = 0
    if pc in [0, 1]: x = 0
    elif pc in [2, 3]: x = 1
    elif pc == 4: x = 2
    elif pc in [5, 6]: x = 3
    elif pc in [7, 8]: x = 4
    elif pc in [9, 10]: x = 5
    elif pc == 11: x = 6
    x += 7 * (octave - 4)
    y = 1 if pc in [1, 3, 6, 8, 10] else 0
    return (x, y)

def subtract_keypos(kp1: tuple[int, int], kp2: tuple[int, int]) -> tuple[int, int]:
    """
    Calculates the interval from kp2 to kp1.
    """
    return (kp1[0] - kp2[0], kp1[1] - kp2[1])

PITCH_TO_KEYPOS_LUT = np.array([pitch_to_keypos(i) for i in range(128)], dtype=np.int16)

def sitch_to_pitch(sitch: str) -> int:
    """
    Converts a spelled pitch string (sitch) to a MIDI pitch number.
    """
    if sitch in ["R", "rest"]:
        return -1

    p_rel = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71}.get(sitch[0])
    
    match = OCTAVE_REGEX.search(sitch)
    if not match:
        raise ValueError(f"Could not parse octave from sitch: {sitch}")
    
    octave = int(match.group(1))
    accidental_str = sitch[1:match.start()]
    
    pitch = p_rel + (octave - 4) * 12
    
    if accidental_str in ["#", "+"]: pitch += 1
    elif accidental_str in ["##", "++"]: pitch += 2
    elif accidental_str in ["b", "-"]: pitch -= 1
    elif accidental_str in ["bb", "--"]: pitch -= 2
        
    return pitch

def parse_pig_file(filepath: str) -> np.ndarray:
    """
    Parses a PIG score file into a structured NumPy array.
    """
    records = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('#'):
                continue
            
            try:
                parts = line.split()
                records.append((int(parts[0]), float(parts[1]), float(parts[2]), 
                                sitch_to_pitch(parts[3]), int(parts[6]), parts[7]))
            except (IndexError, ValueError):
                print(f"Warning: Skipping malformed line {line_num} in {filepath}", file=sys.stderr)
                continue

    dtype = [('original_idx', 'i4'), ('ontime', 'f8'), ('offtime', 'f8'), 
             ('pitch', 'i4'), ('channel', 'i4'), ('finger_str', 'U20')]
    return np.array(records, dtype=dtype)

def sort_notes_by_time(notes: np.ndarray) -> np.ndarray:
    """
    Sorts notes based on the time-dependent clustering logic from the C++ code.
    """
    if len(notes) == 0:
        return notes

    clusters, current_cluster = [], [notes[0]]
    for i in range(1, len(notes)):
        if abs(notes[i]['ontime'] - notes[i-1]['ontime']) >= 0.03:
            clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append(notes[i])
    clusters.append(current_cluster)

    sorted_notes = [note for cluster in clusters for note in sorted(cluster, key=lambda x: x['pitch'])]
    return np.array(sorted_notes, dtype=notes.dtype)

def filter_notes_by_hand(notes: np.ndarray, hand: str) -> np.ndarray:
    """
    Filters the structured array to keep only notes for the specified hand.
    """
    if len(notes) == 0:
        return notes
    is_left = np.char.startswith(notes['finger_str'], '-')
    if hand == 'left': return notes[is_left]
    elif hand == 'right': return notes[~is_left]
    else: raise ValueError("Hand must be 'right' or 'left'")

def clean_finger_str(finger_str: str) -> int:
    """
    Parses the PIG finger string into a simple integer.
    """
    if '_' in finger_str:
        finger_str = finger_str.split('_')[0]
    try:
        val = int(finger_str)
        return val if 1 <= abs(val) <= 5 else 0
    except ValueError:
        return 0
