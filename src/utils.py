import numpy as np

# Phase 1: Foundations (`src/utils.py`)

# 1. Define Constants:
KEYBOARD_SIZE_CM = 16.5
K_STEP = KEYBOARD_SIZE_CM / 7.0
EPSILON_CHORD = 0.05
GAP_THRESHOLD = 0.5

# Reprendre exactement le _kb_layout fourni dans utils.py
_kb_layout = {
    "C"  : 0.5, "D"  : 1.5, "E"  : 2.5, "F"  : 3.5, "G"  : 4.5, "A"  : 5.5, "B"  : 6.5,
    "B#" : 0.5, "C#" : 1.0, "D#" : 2.0, "E#" : 3.5, "F#" : 4.0, "G#" : 5.0, "A#" : 6.0,
    "C-" : 6.5, "D-" : 1.0, "E-" : 2.0, "F-" : 2.5, "G-" : 4.0, "A-" : 5.0, "B-" : 6.0,
    "C##": 1.5, "D##": 2.5, "F##": 4.5, "G##": 5.5, "A##": 6.5,
    "D--": 0.5, "E--": 1.5, "G--": 3.5, "A--": 4.5, "B--": 5.5,
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
BLACK_SEMITONES = {1, 3, 6, 8, 10}  # indices sémis correspondant aux touches noires

def generate_luts():
    """
    Port the exact logic from legacy/utils.py (do not reinvent the geometry).
    Return: Two Numpy arrays.
        keypos_lut: np.float64[128]. X position (cm) for every MIDI pitch.
        is_black_lut: np.int8[128]. 1 if black key, 0 if white.
    """
    keypos_lut = np.empty(128, dtype=np.float64)
    is_black_lut = np.empty(128, dtype=np.int8)

    keybsize = 16.5
    k = keybsize / 7.0

    for midi in range(128):
        octave = midi // 12
        semitone = midi % 12
        name = NOTE_NAMES[semitone]

        is_black_lut[midi] = 1 if semitone in BLACK_SEMITONES else 0

        step_pos = _kb_layout[name]
        step = step_pos * k

        pos = keybsize * octave + step
        keypos_lut[midi] = pos

    return keypos_lut, is_black_lut
