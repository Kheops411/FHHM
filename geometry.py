import numpy as np

# Physical X positions (mm) for MIDI pitches 0..127
# Based on linearized piano geometry model.
_KEY_X_POS = np.zeros(128, dtype=np.float64)
_IS_BLACK = np.zeros(128, dtype=bool)

def _init_geometry():
    """Initializes the lookup tables."""
    # Standard white key width approx 23.6mm
    wk_width = 23.6
    
    # 12-tone pattern: 0=White, 1=Black
    # C, C#, D, D#, E, F, F#, G, G#, A, A#, B
    is_black_pattern = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    
    # Mapping pitch class (0-11) to white key index within octave (0-6)
    wk_map = {0:0, 1:0, 2:1, 3:1, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5, 10:5, 11:6}
    
    for pitch in range(128):
        pc = pitch % 12
        octave = (pitch // 12) - 1
        
        is_blk = is_black_pattern[pc]
        
        # Calculate base X based on white keys count
        wk_index_global = wk_map[pc] + (pitch // 12) * 7
        x_base = wk_index_global * wk_width
        
        # Adjust black keys to center them visually
        if is_blk:
            if pc == 1: x_base += wk_width * 0.6   # C#
            elif pc == 3: x_base += wk_width * 0.4 # D#
            elif pc == 6: x_base += wk_width * 0.55 # F#
            elif pc == 8: x_base += wk_width * 0.5  # G#
            elif pc == 10: x_base += wk_width * 0.45 # A#
        
        _KEY_X_POS[pitch] = x_base
        _IS_BLACK[pitch] = bool(is_blk)

_init_geometry()

def get_key_geometry(pitches: np.ndarray):
    """
    Returns (x_coordinates, is_black_status) for an array of MIDI pitches.
    Output arrays are guaranteed C-Contiguous float64 and bool.
    """
    # Clip to valid MIDI range [0, 127] to prevent segfaults
    safe_p = np.clip(pitches, 0, 127)
    
    # Ensure memory contiguity for Numba consumers
    x = np.ascontiguousarray(_KEY_X_POS[safe_p])
    b = np.ascontiguousarray(_IS_BLACK[safe_p])
    
    return x, b