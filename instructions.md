

# Implementation of High-Performance "ScoreData" Engines (SoA Architecture)

**1. Project Context (Read Carefully)**
We are optimizing a piano fingering system. Currently, the system uses slow Python objects. We have introduced a new data structure called `ScoreData` (defined in `structures.py`) which uses contiguous NumPy arrays for maximum performance.

Your task is to implement "Adapters" and "Engines" that work natively with this `ScoreData` structure.

**Resources available:**
*   `structures.py`: Defines the `ScoreData` class. **Do not modify.**
*   `geometry.py`: Defines `get_key_geometry(pitches)`. Returns physical X coordinates for keys. **Do not modify.**
*   `tests/golden_data/`: Contains JSON files representing the "Ground Truth" (correct output).

**2. Strict Development Rules (Zero Tolerance)**
*   **NO PATCHING:** If a calculation is wrong, find the logical error in your array indexing or math. **DO NOT** add `if x == specific_value: return fixed_value`.
*   **NO MAGIC NUMBERS:** Do not insert arbitrary weights or constants to make a test pass.
*   **NO MODIFICATION OF LEGACY FILES:** You must create **NEW** files (`_soa.py`). Do not touch `legacy/engine.py` or `hmm/utils.py`.
*   **VERIFY INTERNALS:** If a result seems wrong, print the intermediate arrays (`x_pos`, `costs`, etc.) to understand *why*.
*   **MATH IS SACRED:** The probability formulas in the legacy engine are correct. Do not change them. Only change **how data is accessed** (Array Index vs Object Attribute).

---

### Task 1: Create `xml_parser_soa.py`

Create this file at the root. It converts the output of the existing XML parser into `ScoreData`.

*   **Function Signature:** `def musicxml_to_soa(xml_path: str) -> Tuple[ScoreData, Dict[int, object]]`
*   **Logic:**
    1.  Use `MusicXMLParser(path).parse()` to get a list of `PlayedNote` objects.
    2.  Filter out notes where `pitch` is `None` (rests).
    3.  Allocate `ScoreData(len(notes))`.
    4.  Fill arrays (`onset`, `offset`, `pitch`, `velocity`, `hand`, `measure`).
    5.  **Crucial:** Create a dictionary `source_map = {}`. Store `id(note.xml_element)` in `ScoreData.source_ref[i]` and put the actual element in `source_map`.
    6.  Call `soa.sort_canonical()`.

### Task 2: Create `hmm/loader_soa.py`

Create this file in `hmm/`. It loads PIG text files directly into `ScoreData`.

*   **Function Signature:** `def load_pig_to_soa(filepath: str) -> ScoreData`
*   **Logic:**
    1.  Read the file line by line (skip comments `//` or `#`).
    2.  Parse columns: ID, onset, offset, pitch(string), velocity... finger.
    3.  Convert pitch string (e.g., "C#4") to MIDI int.
    4.  **Constraint:** Do not use `numpy.genfromtxt` or complex regex if it's slow. Simple string splitting is preferred.
    5.  Populate `ScoreData`.
    6.  Call `soa.sort_canonical()`.

### Task 3: Create `legacy/engine_soa.py`

This is the most complex task. You must port the heuristic algorithm to use arrays.

*   **Function Signature:** `def find_fingerings_soa(soa: ScoreData, hand_side: str) -> np.ndarray`
*   **Logic:**
    1.  **Filter:** Create a boolean mask for the requested hand (0=Right, 1=Left).
    2.  **Geometry:** Call `geometry.get_key_geometry(soa.pitch[mask])` to get `x` and `is_black`.
    3.  **Engine Porting:** Rewrite the `Hand` class from `legacy/engine.py` inside this new file, but:
        *   Remove `self.noteseq` (list of objects).
        *   Accept arrays in `__init__` (`x`, `onsets`, `durations`, `is_black`, `event_ids`).
        *   In `_compute_transition_cost`, use `self.x[i]` instead of `note.x`.
        *   In `_is_forbidden_transition`, use `self.event_ids[i] == self.event_ids[i-1]` to detect chords.
    4.  **Output:** Return an array of fingers (int8) matching the original `soa` size (zeros for the other hand).

---

### Task 4: Mandatory Verification

You cannot just say "it works". You must create and run `verify_soa_implementation.py` with the code below. **If this script fails or prints mismatches, your work is incomplete.**

**Create `verify_soa_implementation.py`:**

```python
import sys
import os
import json
import numpy as np

# Import your new modules
try:
    from xml_parser_soa import musicxml_to_soa
    from hmm.loader_soa import load_pig_to_soa
    from legacy.engine_soa import find_fingerings_soa
    from structures import ScoreData
except ImportError as e:
    print(f"FAIL: Could not import new modules. {e}")
    sys.exit(1)

def check_golden(algo, name, soa_fingers, golden_path):
    with open(golden_path, 'r') as f:
        golden = json.load(f)
    
    # Golden is list of dicts. ScoreData is arrays.
    # We must match them. Golden data is sorted. ScoreData is sorted.
    # However, ScoreData contains ALL notes (both hands). Golden might be partial.
    
    match_count = 0
    mismatch_count = 0
    
    print(f"Verifying {name}...")
    
    # We assume strict alignment because both are sorted canonically
    # But we must filter soa by hand/validity to match golden
    
    # Simple check: Iterate golden, find corresponding note in SOA by time/pitch
    for g_note in golden:
        g_onset = g_note['onset']
        g_pitch = g_note['pitch']
        g_finger = g_note['finger']
        
        # Find in SOA
        # This is slow O(N^2) but fine for verification script
        matches = np.where(
            (np.abs(soa.onset - g_onset) < 0.001) & 
            (soa.pitch == g_pitch)
        )[0]
        
        if len(matches) == 0:
            print(f"  [ERROR] Golden note {g_onset}s pitch {g_pitch} not found in SOA!")
            mismatch_count += 1
            continue
            
        idx = matches[0]
        soa_finger = soa.finger_out[idx]
        
        if soa_finger != g_finger:
            print(f"  [MISMATCH] Time {g_onset:.2f} Pitch {g_pitch}: Expected {g_finger}, Got {soa_finger}")
            # DEBUGGING HINT FOR DEVELOPER:
            # If you see this, check your cost function weights or geometry mapping.
            mismatch_count += 1
        else:
            match_count += 1

    if mismatch_count == 0:
        print(f"  [PASS] {match_count} notes matched perfectly.")
    else:
        print(f"  [FAIL] {mismatch_count} mismatches found.")

def test_legacy_soa():
    print("--- Testing Legacy SOA ---")
    xml_path = "tests/resources/sanity_check.xml"
    if not os.path.exists(xml_path):
        print("Skipping Legacy test (file not found)")
        return

    # 1. Load
    soa, _ = musicxml_to_soa(xml_path)
    
    # 2. Run Algo (Both hands)
    fingers_rh = find_fingerings_soa(soa, "right")
    fingers_lh = find_fingerings_soa(soa, "left")
    
    # Combine
    soa.finger_out = fingers_rh + fingers_lh
    
    # 3. Check
    check_golden("legacy", "sanity_check.xml", soa, "tests/golden_data/legacy_sanity_check.xml.json")

if __name__ == "__main__":
    test_legacy_soa()
```

**Deliverables:**
1.  `xml_parser_soa.py`
2.  `hmm/loader_soa.py`
3.  `legacy/engine_soa.py`
4.  Output log of `verify_soa_implementation.py` showing **[PASS]**.