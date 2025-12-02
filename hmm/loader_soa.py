import numpy as np
from structures import ScoreData

def _sitch_to_pitch(sitch):
    """
    Converts a pitch string (e.g., 'C#4') to a MIDI pitch number.
    This is a simplified version, assuming standard tuning and no key signatures.
    """
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    sitch = sitch.strip()

    note = sitch[0].upper()
    octave_part = sitch[1:]

    accidental = 0
    if len(octave_part) > 1 and octave_part[0] == '#':
        accidental = 1
        octave_part = octave_part[1:]
    elif len(octave_part) > 1 and octave_part[0] == 'b':
        accidental = -1
        octave_part = octave_part[1:]

    try:
        octave = int(octave_part)
        pitch = 12 * (octave + 1) + note_map[note] + accidental
        return pitch
    except (ValueError, KeyError):
        return 0 # Return 0 for invalid pitch strings

def load_pig_to_soa(filepath: str) -> ScoreData:
    """
    Loads a PIG text file directly into ScoreData.
    """
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('//') and not line.startswith('#') and line.strip():
                lines.append(line.strip().split())

    n_notes = len(lines)
    soa = ScoreData.allocate(n_notes)

    for i, parts in enumerate(lines):
        # Parse columns: ID, onset, offset, pitch(string), velocity... finger.
        if len(parts) < 8:
            continue

        soa.id[i] = int(parts[0])
        soa.onset[i] = float(parts[1])
        soa.offset[i] = float(parts[2])
        soa.pitch[i] = _sitch_to_pitch(parts[3])
        soa.velocity[i] = int(parts[4])
        # Column 5 is unknown, skipping
        soa.hand[i] = int(parts[6])

        # PIG file fingerings can have extra characters like '_1', we take the first digit.
        finger_str = parts[7]
        cleaned_finger_str = ""
        for char in finger_str:
            if char.isdigit() or char == '-':
                cleaned_finger_str += char
            else:
                break
        if cleaned_finger_str:
            soa.finger_gt[i] = int(cleaned_finger_str)
        else:
            soa.finger_gt[i] = 0


    soa.sort_canonical()

    return soa
