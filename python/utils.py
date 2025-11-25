import numpy as np
import numba as nb
from typing import Tuple

# Global LUT: (128 pitches, 2 coordinates [x, y])
PITCH_TO_KEYPOS_LUT = np.zeros((128, 2), dtype=np.int16)

def _compute_pitch_to_keypos_lut():
    """
    Port of C++ PitchToKeyPos from KeyPos_v161230.hpp
    Convention: C4=60=(0,0), D4=62=(1,0), Eb4=63=(1,1)
    """
    for pitch in range(128):
        pc = pitch % 12
        octave = (pitch // 12) - 1
        
        # Base X mapping (White key index 0-6)
        x = 0
        if pc in (0, 1): x = 0
        elif pc in (2, 3): x = 1
        elif pc == 4: x = 2
        elif pc in (5, 6): x = 3
        elif pc in (7, 8): x = 4
        elif pc in (9, 10): x = 5
        elif pc == 11: x = 6

        # Add octave offset (7 white keys per octave)
        # C++: keyPos.x+=7*(oct-4); NOTE: Check if python needs exact match on octave base
        x += 7 * (octave - 4)

        # Y mapping (0=White/Natural, 1=Black/Accidental)
        # C++: if(pc==0||pc==2||pc==4||pc==5||pc==7||pc==9||pc==11){keyPos.y=0;}
        y = 1
        if pc in (0, 2, 4, 5, 7, 9, 11):
            y = 0
        else:
            y = 1
            
        PITCH_TO_KEYPOS_LUT[pitch, 0] = x
        PITCH_TO_KEYPOS_LUT[pitch, 1] = y

# Initialize on import
_compute_pitch_to_keypos_lut()

# --- Public API ---

def pitch_to_keypos(midi_pitch: int) -> Tuple[int, int]:
    """Python-friendly wrapper for tests/non-critical paths."""
    if not (0 <= midi_pitch < 128):
        raise ValueError(f"Pitch {midi_pitch} out of bounds")
    row = PITCH_TO_KEYPOS_LUT[midi_pitch]
    return int(row[0]), int(row[1])

def subtract_keypos(kp1: Tuple[int,int], kp2: Tuple[int,int]) -> Tuple[int,int]:
    """Python-friendly wrapper for tests."""
    return (kp1[0] - kp2[0], kp1[1] - kp2[1])

# --- Numba Optimized API (Hot Path) ---

@nb.njit(cache=True)
def pitch_to_keypos_numba(midi_pitch: int, lut: np.ndarray) -> np.ndarray:
    """
    Numba-optimized lookup.
    Usage: pitch_to_keypos_numba(60, PITCH_TO_KEYPOS_LUT)
    Returns array([x, y])
    """
    # Numba implementation...
    return lut[midi_pitch]

@nb.njit(cache=True)
def subtract_keypos_numba(x1, y1, x2, y2):
    return x1 - x2, y1 - y2

@nb.njit(cache=True)
def lattice_delta_to_index(dx: int, dy: int, width_x: int = 15) -> int:
    # C++: 3*(keyInt.x+widthX)+keyInt.y+1
    # We must clamp dx exactly as C++ does:
    if dx < -width_x: dx = -width_x
    if dx > width_x:  dx = width_x

    return 3 * (dx + width_x) + dy + 1
