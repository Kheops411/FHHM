# Critical Fixes - Alignment with Legacy Physics & Timing

**Context:**
The SOA implementation failed verification (99% mismatch). The cause is a discrepancy in **units**.
1.  **Timing:** The legacy system uses **Musical Time (Quarters)**, not Seconds.
2.  **Space:** The legacy system uses an arbitrary **16.5cm Octave** model, not realistic mm geometry.

You must strictly replicate the "arbitrary" physics of the legacy system to reproduce its results.

---

### Task 1: Update `structures.py`

Modify the `ScoreData` class to store musical time.

**Action:** Add `onset_quarter` and `duration_quarter` to `__slots__` and `allocate`.

```python
@dataclass
class ScoreData:
    __slots__ = (
        'onset', 'offset', 'pitch', 'velocity', 
        'id', 'source_ref', 'event_id', 'measure', 'hand', 
        'finger_gt', 'finger_out',
        'onset_quarter', 'duration_quarter' # <--- ADD THESE
    )
    
    # ... existing fields ...
    onset_quarter: np.ndarray    # Musical time (Quarters)
    duration_quarter: np.ndarray # Musical duration (Quarters)

    @classmethod
    def allocate(cls, n_notes: int):
        return cls(
            # ... existing fields ...
            onset_quarter=np.zeros(n_notes, dtype=np.float64), # <--- INIT
            duration_quarter=np.zeros(n_notes, dtype=np.float64), # <--- INIT
        )
```

---

### Task 2: Update `geometry.py`

Replace the "Realistic" geometry with the "Legacy" geometry logic. Copy-paste this exact code.

```python
import numpy as np

# Replicating EXACTLY legacy/utils.py logic
_KEY_X_POS = np.zeros(128, dtype=np.float64)
_IS_BLACK = np.zeros(128, dtype=bool)

def _init_geometry():
    keybsize = 16.5  # cm
    k = keybsize / 7.0
    
    # Legacy Layout dictionary
    _kb_layout = {
        "C": 0.5, "D": 1.5, "E": 2.5, "F": 3.5, "G": 4.5, "A": 5.5, "B": 6.5,
        "C#": 1.0, "D#": 2.0, "F#": 4.0, "G#": 5.0, "A#": 6.0
    }
    
    # Simple mapping: 0=C, 1=C#, etc.
    pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
    for pitch in range(128):
        pc = pitch % 12
        octave = (pitch // 12) - 1
        name = pc_names[pc]
        
        step = _kb_layout[name] * k
        # Formula from legacy/utils.py keypos()
        pos = keybsize * octave + step
        
        _KEY_X_POS[pitch] = pos
        _IS_BLACK[pitch] = '#' in name

_init_geometry()

def get_key_geometry(pitches: np.ndarray):
    safe_p = np.clip(pitches, 0, 127)
    x = np.ascontiguousarray(_KEY_X_POS[safe_p])
    b = np.ascontiguousarray(_IS_BLACK[safe_p])
    return x, b
```

---

### Task 3: Update `xml_parser_soa.py`

Populate the new musical time fields.

**Action:** In the loop where you fill `soa` arrays:

```python
    # ... existing assignments ...
    soa.onset_quarter[i] = note.onset      # PlayedNote.onset is in quarters
    soa.duration_quarter[i] = note.duration # PlayedNote.duration is in quarters
```

---

### Task 4: Fix `legacy/engine_soa.py`

Remove the conversion hacks and use the correct data sources.

**Action:** Modify `find_fingerings_soa`.

```python
def find_fingerings_soa(soa: ScoreData, hand_side: str) -> np.ndarray:
    target_hand_int = HAND_RIGHT if hand_side == "right" else HAND_LEFT
    mask = (soa.hand == target_hand_int)
    
    pitches = soa.pitch[mask]
    
    # USE NEW FIELDS: Musical Time
    onsets = soa.onset_quarter[mask]
    # Legacy scaling: main.py adapter multiplies duration by 4. Replicate this.
    durations = soa.duration_quarter[mask] * 4.0 
    
    event_ids = soa.event_id[mask]
    
    if len(pitches) == 0:
        return np.zeros(len(soa), dtype=np.int8)

    # Geometry is now natively in CM (Legacy compatible)
    x, is_black = get_key_geometry(pitches)
    if hand_side == 'left':
        x = -x

    # --- DELETE ALL PREVIOUS NORMALIZATION CODE (x*0.1, time*2.0, etc) ---

    # Calculate Chord logic (Same as before)
    unique_ids, counts = np.unique(event_ids, return_counts=True)
    counts_map = dict(zip(unique_ids, counts))
    is_chord_arr = np.array([counts_map[eid] > 1 for eid in event_ids], dtype=bool)

    hand_engine = HandSOA(x, onsets, durations, is_black, event_ids, is_chord_arr, side=hand_side, hf=0.82)
    finger_path = hand_engine.generate()
    
    # ... Output assignment logic remains same ...
```

---

### Verification

After applying these 4 changes, run `verify_soa_advanced.py`.
Expected result: `PASS` (or extremely high match rate > 99%).