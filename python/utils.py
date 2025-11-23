import numpy as np
import os
import glob

def get_midi_to_lattice_lut():
    """
    Generates a Look-up Table (LUT) for converting MIDI pitches to lattice coordinates,
    matching the C++ implementation's convention where C4 (MIDI 60) is at (0,0).
    """
    lut = np.zeros((128, 2), dtype=np.float32)

    for i in range(128):
        pitch = i
        pc = pitch % 12
        octave = pitch // 12 - 1

        x_pos = 0
        if pc == 0 or pc == 1: x_pos = 0
        elif pc == 2 or pc == 3: x_pos = 1
        elif pc == 4: x_pos = 2
        elif pc == 5 or pc == 6: x_pos = 3
        elif pc == 7 or pc == 8: x_pos = 4
        elif pc == 9 or pc == 10: x_pos = 5
        elif pc == 11: x_pos = 6

        x_pos += 7 * (octave - 4)

        y_pos = 0
        if pc in [0, 2, 4, 5, 7, 9, 11]: y_pos = 0
        elif pc in [1, 3, 6, 8, 10]: y_pos = 1

        lut[i] = [x_pos, y_pos]

    return lut

MIDI_TO_LATTICE_LUT = get_midi_to_lattice_lut()

def parse_fingering_file(filepath):
    """
    Parses a single fingering .txt file and converts it into a structured NumPy array.
    """
    notes = []
    dtype = np.dtype([
        ('pitch', 'i1'), ('onset', 'f4'), ('duration', 'f4'),
        ('lattice_x', 'f4'), ('lattice_y', 'i1'), ('hand', 'i1'), ('finger', 'i1')
    ])

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('//'): continue
            parts = line.strip().split()
            if len(parts) < 8: continue

            onset, offset = float(parts[1]), float(parts[2])
            pitch = int(parts[4])
            hand = int(parts[6])
            finger_str = parts[7].split('_')[0]
            finger = abs(int(finger_str))

            lattice_x, lattice_y = MIDI_TO_LATTICE_LUT[pitch]

            notes.append((
                pitch, onset, offset - onset,
                lattice_x, lattice_y, hand, finger
            ))

    return np.array(notes, dtype=dtype)

def parse_all_scores(scores_dir):
    """
    Parses all .txt files in the given directory.
    """
    all_scores = {}
    for filepath in glob.glob(os.path.join(scores_dir, '*.txt')):
        try:
            filename = os.path.basename(filepath)
            all_scores[filename] = parse_fingering_file(filepath)
        except (ValueError, IndexError) as e:
            print(f"Error parsing file {filepath}: {e}")
            raise
    return all_scores
