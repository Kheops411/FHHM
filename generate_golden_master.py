import sys
import os
import json
import numpy as np
from glob import glob
from types import SimpleNamespace

# Ajout du chemin courant pour les imports
sys.path.append(os.getcwd())

# --- Imports Legacy ---
try:
    from xml_parser import MusicXMLParser
    from legacy.engine import find_fingerings
    from legacy.utils import keypos
except ImportError as e:
    print(f"CRITICAL: Failed to import Legacy components: {e}")
    sys.exit(1)

# --- Imports HMM ---
try:
    from hmm.utils import load_pig_file, filter_notes_by_hand
    from hmm.model import HMMParameters, run_viterbi
except ImportError as e:
    # On rend l'import HMM optionnel pour le sanity check XML, 
    # mais on prévient si ça manque pour les tests complets
    print(f"WARN: Failed to import HMM components: {e}")

def serialize_result(seq_name, notes, algo_type):
    """Saves fingering results to JSON for regression testing."""
    output = []
    
    # Sort globally by Onset then Pitch for readable JSON
    # Handling both Object (Legacy) and Dict/Struct (HMM)
    
    # Helper to get sort key
    def get_sort_key(n):
        if hasattr(n, 'onset'): return (n.onset, n.pitch or -1)
        return (n['ontime'], n['pitch'])

    try:
        sorted_notes = sorted(notes, key=get_sort_key)
    except:
        sorted_notes = notes # Fallback if sort fails

    for n in sorted_notes:
        item = {}
        if isinstance(n, dict) or (isinstance(n, np.void) and n.dtype.names):
            # NumPy struct or Dict (HMM)
            item["onset"] = float(n['ontime'])
            item["pitch"] = int(n['pitch'])
            item["finger"] = int(n['finger'])
            # HMM utils uses negative fingers for LH, positive for RH usually.
            # We can infer hand from finger sign if needed, or if separate logic was used.
            # For PIG, we often don't store 'hand' explicitly in the output struct 
            # unless we add it. Let's assume finger sign is enough info for PIG.
        else:
            # Python Object (Legacy/MusicXML)
            item["onset"] = float(n.onset)
            item["pitch"] = int(n.pitch)
            item["finger"] = int(n.finger) if n.finger is not None else 0
            item["hand"] = "right" if n.hand == 0 else "left"
            
        output.append(item)
    
    os.makedirs("tests/golden_data", exist_ok=True)
    filename = f"tests/golden_data/{algo_type}_{seq_name}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"   [OK] Generated {filename}")

def run_legacy_baseline():
    print("1. Running Legacy Baseline...")
    xml_files = glob("tests/resources/*.xml")
    if not xml_files:
        print("   [WARN] No XML files in tests/resources/")
        return

    for xml_file in xml_files:
        try:
            parser = MusicXMLParser(xml_file)
            played_notes = parser.parse()
            
            all_fingered_notes = []

            # Process Both Hands
            hands_config = [(0, "right"), (1, "left")]
            
            for hand_id, side_name in hands_config:
                # Filter notes for this hand, excluding rests
                hand_notes = [n for n in played_notes if n.hand == hand_id and n.pitch is not None]
                hand_notes.sort(key=lambda x: (x.onset, x.pitch))
                
                if not hand_notes: continue

                # --- Legacy Adapter Logic ---
                engine_notes = []
                chord_map = {} 
                
                # Chord detection
                for n in hand_notes:
                    if n.extra.get('chord_size', 1) > 1:
                        chord_map[n.onset] = int(n.onset * 1000)

                for n in hand_notes:
                    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                    note_name = note_names[n.pitch % 12]
                    octave = (n.pitch // 12) - 1
                    
                    kp = SimpleNamespace(name=note_name, octave=octave)
                    x = keypos(kp)
                    
                    engine_notes.append(SimpleNamespace(
                        x=x, 
                        duration=n.duration * 4,
                        isBlack='#' in note_name, 
                        isChord=n.extra.get('chord_size', 1) > 1,
                        time=n.onset, 
                        measure=n.measure_number, 
                        chordID=chord_map.get(n.onset),
                        original_note=n,
                        fingering=None, 
                        cost=0
                    ))
                
                # Run Algorithm for this hand
                result = find_fingerings(engine_notes, side_name, "M")
                
                # Apply Back
                for i, res in enumerate(result):
                    engine_notes[i].original_note.finger = res.fingering
                
                all_fingered_notes.extend(hand_notes)
            
            # Serialize combined results
            if all_fingered_notes:
                serialize_result(os.path.basename(xml_file), all_fingered_notes, "legacy")
            
        except Exception as e:
            print(f"   [FAIL] {xml_file}: {e}")
            import traceback
            traceback.print_exc()

def run_hmm_baseline():
    print("2. Running HMM Baseline...")
    pig_files = glob("scores/*.txt")
    pig_files = pig_files[:3] 
    
    if not pig_files:
        return

    param_file = "hmm/param_FHMM2.txt"
    if not os.path.exists(param_file):
        print(f"   [WARN] Params {param_file} not found")
        return
        
    try:
        params = HMMParameters(param_file)
        
        for p_file in pig_files:
            all_notes = load_pig_file(p_file)
            
            # Run for Right Hand (0)
            rh_indices = np.where(all_notes['finger'] > 0)[0]
            if len(rh_indices) > 0:
                rh_subset = all_notes[rh_indices]
                fingers_rh = run_viterbi(rh_subset, params, hand=0)
                all_notes['finger'][rh_indices] = fingers_rh

            # Run for Left Hand (1)
            lh_indices = np.where(all_notes['finger'] < 0)[0]
            if len(lh_indices) > 0:
                lh_subset = all_notes[lh_indices]
                # Note: HMM code usually expects positive indices for processing 
                # even for left hand, logic is handled by 'hand' param.
                # Assuming filtering/logic inside utils handles it.
                fingers_lh = run_viterbi(lh_subset, params, hand=1)
                # Convert back to negative for storage if needed, or keep as is?
                # PIG format uses negative for LH.
                all_notes['finger'][lh_indices] = -np.abs(fingers_lh)

            serialize_result(os.path.basename(p_file), all_notes, "hmm")
            
    except Exception as e:
        print(f"   [FAIL] HMM Baseline: {e}")

if __name__ == "__main__":
    print("--- GENERATING GOLDEN MASTER DATA ---")
    run_legacy_baseline()
    run_hmm_baseline()
    print("--- DONE ---")