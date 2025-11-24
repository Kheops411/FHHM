import numpy as np
import re

def pitch_to_keypos(pitch: int) -> tuple[int, int]:
    """
    Converts a MIDI pitch to a keyboard lattice coordinate.

    This is a literal translation of the C++ function `PitchToKeyPos`
    from `KeyPos_v161230.hpp`.

    Args:
        pitch: The MIDI pitch (0-127).

    Returns:
        A tuple (x, y) representing the coordinate.
    """
    pc = pitch % 12
    oct = pitch // 12 - 1

    x = 0
    if pc == 0 or pc == 1:
        x = 0
    elif pc == 2 or pc == 3:
        x = 1
    elif pc == 4:
        x = 2
    elif pc == 5 or pc == 6:
        x = 3
    elif pc == 7 or pc == 8:
        x = 4
    elif pc == 9 or pc == 10:
        x = 5
    elif pc == 11:
        x = 6

    x += 7 * (oct - 4)

    y = 0
    if pc == 0 or pc == 2 or pc == 4 or pc == 5 or pc == 7 or pc == 9 or pc == 11:
        y = 0
    elif pc == 1 or pc == 3 or pc == 6 or pc == 8 or pc == 10:
        y = 1

    return (x, y)

def subtract_keypos(kp1: tuple[int, int], kp2: tuple[int, int]) -> tuple[int, int]:
    """
    Calculates the interval from kp2 to kp1.

    Args:
        kp1: The first key position (x1, y1).
        kp2: The second key position (x2, y2).

    Returns:
        A tuple (dx, dy) representing the interval.
    """
    return (kp1[0] - kp2[0], kp1[1] - kp2[1])

# Pre-calculate the Lookup Table (LUT) for all 128 MIDI pitches.
PITCH_TO_KEYPOS_LUT = np.array([pitch_to_keypos(i) for i in range(128)], dtype=np.int16)


def sitch_to_pitch(sitch: str) -> int:
    """
    Converts a spelled pitch string (sitch) to a MIDI pitch number.

    This is a literal translation of the C++ function `SitchToPitch`
    from `BasicPitchCalculation_v170101.hpp`.

    Args:
        sitch: The spelled pitch string (e.g., "C#4", "Bb-1").

    Returns:
        The MIDI pitch number.
    """
    if sitch == "R" or sitch == "rest":
        return -1

    p_rel = 0
    note_char = sitch[0]

    if note_char == 'C':
        p_rel = 60
    elif note_char == 'D':
        p_rel = 62
    elif note_char == 'E':
        p_rel = 64
    elif note_char == 'F':
        p_rel = 65
    elif note_char == 'G':
        p_rel = 67
    elif note_char == 'A':
        p_rel = 69
    elif note_char == 'B':
        p_rel = 71

    # Use regex to find the octave number, which may be negative
    match = re.search(r'(-?\d+)$', sitch)
    if not match:
        raise ValueError(f"Could not parse octave from sitch: {sitch}")

    oct = int(match.group(1))

    # The accidental is the part between the note and the octave
    accidental_str = sitch[1:match.start()]

    p = p_rel + (oct - 4) * 12

    if accidental_str == "":
        p += 0
    elif accidental_str == "#" or accidental_str == "+":
        p += 1
    elif accidental_str == "##" or accidental_str == "++":
        p += 2
    elif accidental_str == "b" or accidental_str == "-":
        p -= 1
    elif accidental_str == "bb" or accidental_str == "--":
        p -= 2

    return p


def parse_pig_file(filepath: str) -> np.ndarray:
    """
    Parses a PIG score file into a structured NumPy array.

    Args:
        filepath: The path to the .txt score file.

    Returns:
        A structured NumPy array with the note data.
    """
    records = []
    idx_counter = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('#'):
                continue

            try:
                parts = line.split('\t')

                ontime = float(parts[1])
                offtime = float(parts[2])
                sitch = parts[3]
                channel = int(parts[6])
                finger_str = parts[7]

                pitch = sitch_to_pitch(sitch)

                records.append((idx_counter, ontime, offtime, pitch, channel, finger_str))
                idx_counter += 1
            except (IndexError, ValueError):
                # Malformed line, skip it as per the test requirements
                continue

    dtype = [
        ('original_idx', 'i4'),
        ('ontime', 'f8'),
        ('offtime', 'f8'),
        ('pitch', 'i4'),
        ('channel', 'i4'),
        ('finger_str', 'U20') # Increased from U10 to prevent truncation
    ]

    return np.array(records, dtype=dtype)


def sort_notes_by_time(notes: np.ndarray) -> np.ndarray:
    """
    Sorts notes based on the time-dependent clustering logic from the C++ code.

    Args:
        notes: The structured array of notes from parse_pig_file.

    Returns:
        A new structured array of notes, sorted according to the custom logic.
    """
    if len(notes) == 0:
        return notes

    # Ensure notes are sorted by ontime before clustering.
    # The clustering logic implicitly assumes this pre-sorting.
    notes.sort(order='ontime')

    # Cluster events where abs(t_n - t_{n-1}) < 0.03
    clusters = []
    current_cluster = [notes[0]]

    for i in range(1, len(notes)):
        if abs(notes[i]['ontime'] - notes[i-1]['ontime']) >= 0.03:
            clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append(notes[i])
    clusters.append(current_cluster)

    # Sort inside each cluster by Pitch Ascending
    sorted_notes = []
    for cluster in clusters:
        # The C++ code uses stable_sort with a custom comparator on -pitch,
        # which results in a descending sort on -pitch (i.e., an ascending sort on pitch).
        # Our implementation of sorting by 'pitch' in ascending order is equivalent.
        sorted_cluster = sorted(cluster, key=lambda x: x['pitch'], reverse=False)
        sorted_notes.extend(sorted_cluster)

    return np.array(sorted_notes, dtype=notes.dtype)
