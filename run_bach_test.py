import sys
import numpy as np
import os

# Adjust path to find modules
sys.path.append(os.getcwd())

from xml_parser import MusicXMLParser, Hand
from soft_position_hmm.core import SoftPositionModel, ANCHORS
from soft_position_hmm.interface import predict_fingering

def main():
    xml_path = "Prélude No. 1 en C Majeur-Piano.xml"
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found.")
        return

    print("1. Parsing XML...")
    parser = MusicXMLParser(xml_path)
    all_notes = parser.parse()

    # Filter for LEFT HAND and remove rests
    lh_notes = [n for n in all_notes if n.hand == Hand.LEFT and n.pitch is not None]
    lh_notes.sort(key=lambda x: x.onset)

    print(f"   Found {len(lh_notes)} Left Hand notes.")

    # Extract arrays
    pitches = np.array([n.pitch for n in lh_notes], dtype=np.int32)
    ontimes = np.array([n.onset_seconds for n in lh_notes], dtype=np.float64)

    print("2. Initializing Soft-Position Model...")
    model = SoftPositionModel()

    # --- GEOMETRY TUNING (Simulated LH via Inversion) ---
    inverted_pitches = 128 - pitches

    # Thumb (0): Likes playing to the LEFT of Anchor
    model.rbf_mu[0] = -5.0
    model.rbf_sigma[0] = 3.0

    # Index (1):
    model.rbf_mu[1] = -2.0

    # Middle (2):
    model.rbf_mu[2] = 0.0

    # Ring (3):
    model.rbf_mu[3] = 2.0

    # Pinky (4): Likes playing to the RIGHT of Anchor
    model.rbf_mu[4] = 5.0
    model.rbf_sigma[4] = 3.0

    # --- PHYSICS TUNING ---
    # Force the hand to be VERY lazy (High Inertia Cost)
    # With Step=1 grid, we can afford very high inertia because
    # exact matches are possible.
    model.inertia_weight = 5.0
    model.time_slope = 15.0

    print("3. Running Inference...")
    fingers, anchors_indices = predict_fingering(
        inverted_pitches,
        ontimes,
        model
    )

    real_fingers = fingers

    print("4. Analyzing First 4 Measures...")
    print(f"{'Time':<8} | {'Note':<6} | {'Finger':<6} | {'Anchor':<6} | {'Status'}")
    print("-" * 60)

    previous_hand_center = -999

    for i in range(min(32, len(real_fingers))):
        pitch_val = pitches[i]
        inv_pitch = inverted_pitches[i]
        f = real_fingers[i]
        a_idx = anchors_indices[i]
        a_val = ANCHORS[a_idx]

        # Hand Center = Note + Anchor
        hand_center = inv_pitch + a_val

        status = ""
        if i > 0:
            dist = abs(hand_center - previous_hand_center)
            if dist == 0:
                status = "(Stable)"
            else:
                status = f"--> MOVE ({dist})"

        print(f"{ontimes[i]:<8.2f} | {pitch_val:<6} | {f:<6} | {a_val:<6} | {status}")

        previous_hand_center = hand_center

if __name__ == "__main__":
    main()
