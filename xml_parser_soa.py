from typing import Tuple, Dict
import numpy as np
from structures import ScoreData
from xml_parser import MusicXMLParser, PlayedNote

def musicxml_to_soa(xml_path: str) -> Tuple[ScoreData, Dict[int, object]]:
    """
    Converts the output of the existing XML parser into ScoreData.
    """
    parser = MusicXMLParser(xml_path)
    played_notes = parser.parse()

    # Filter out notes where pitch is None (rests).
    notes = [note for note in played_notes if note.pitch is not None]

    # Allocate ScoreData
    n_notes = len(notes)
    soa = ScoreData.allocate(n_notes)

    source_map = {}

    for i, note in enumerate(notes):
        # Fill arrays
        soa.onset[i] = note.onset_seconds
        soa.offset[i] = note.offset_seconds
        soa.pitch[i] = note.pitch
        soa.velocity[i] = note.velocity
        soa.hand[i] = note.hand
        soa.measure[i] = note.measure_number

        # Crucial: Create a dictionary `source_map`.
        if note.xml_element is not None:
            element_id = id(note.xml_element)
            soa.source_ref[i] = element_id
            source_map[element_id] = note.xml_element

    # Generate event_id for chords based on onset time
    if n_notes > 0:
        # Sort by onset to group chords correctly
        sorted_indices = np.argsort(soa.onset)

        event_id_counter = 0
        soa.event_id[sorted_indices[0]] = event_id_counter
        for i in range(1, n_notes):
            idx_curr = sorted_indices[i]
            idx_prev = sorted_indices[i-1]
            if abs(soa.onset[idx_curr] - soa.onset[idx_prev]) > 1e-3:
                event_id_counter += 1
            soa.event_id[idx_curr] = event_id_counter

    return soa, source_map
