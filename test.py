import numpy as np

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
    keypos_lut = np.empty(128, dtype=np.float64)
    is_black_lut = np.empty(128, dtype=np.int8)

    keybsize = 16.5
    k = keybsize / 7.0

    for midi in range(128):
        octave = midi // 12         # base octave multiplier (comme utils.keypos_midi)
        semitone = midi % 12
        name = NOTE_NAMES[semitone]

        # is_black strictement défini
        is_black_lut[midi] = 1 if semitone in BLACK_SEMITONES else 0

        # step intra-octave à partir du _kb_layout
        # _kb_layout[name] doit exister pour les 12 noms standards
        step_pos = _kb_layout[name]  # .get(name) serait acceptable mais name doit exister
        step = step_pos * k

        pos = keybsize * octave + step
        keypos_lut[midi] = pos

    return keypos_lut, is_black_lut



if __name__ == "__main__":
    kp, ib = generate_luts()

    # Quelques vérifications rapides
    print(f"MIDI 60 (C) Pos: {kp[60]:.6f} cm, Black: {ib[60]}")
    print(f"MIDI 61 (C#) Pos: {kp[61]:.6f} cm, Black: {ib[61]}")
    print(f"Diff C#-C: {kp[61]-kp[60]:.6f} cm")
    # Vérifier uniformité des pas (devrait être exactement k)
    keybsize = 16.5
    k = keybsize / 7.0
    print(f"Pas semantic attendu (k): {k:.6f} cm")

    print(f"MIDI 11 (B) Pos: {kp[11]:.6f} cm, Black: {ib[12]}")
    print(f"MIDI 12 (C) Pos: {kp[11]:.6f} cm, Black: {ib[12]}")
    print(f"Diff B-C: {kp[12]-kp[11]:.6f} cm")
    # Vérifier uniformité des pas (devrait être exactement k)
    keybsize = 16.5
    k = keybsize / 7.0
    print(f"Pas semantic attendu (k): {k:.6f} cm")

    print(f"MIDI 65 (F4) Pos: {kp[65]:.6f} cm, Black: {ib[65]}")
    print(f"MIDI 77 (F5) Pos: {kp[77]:.6f} cm, Black: {ib[77]}")
    print(f"Diff F4-F5: {kp[77]-kp[65]:.6f} cm")
    # Vérifier uniformité des pas (devrait être exactement k)
    keybsize = 16.5
    k = keybsize / 7.0
    print(f"Pas semantic attendu (k): {k:.6f} cm")