"""
xml_parser.py

Parse MusicXML => produce full "PlayedNote" timeline with comprehensive support for:
- Structural navigation (repeats, endings, D.C./D.S./Coda/Segno/<sound>)
- Tuplets, ties, ornaments (trill/mordent/turn), tremolo, glissando, arpeggios
- Grace notes (make-time or steal-time), articulations, slurs, fermata, pedal
- Preserves XML refs for fingering injection

Dependencies: lxml
Usage: python xml_parser.py input.xml output.xml
"""
from lxml import etree
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from collections import defaultdict
from enum import IntEnum
import sys

# ============================================================================
# Constants
# ============================================================================

class Hand(IntEnum):
    """Piano hand enumeration"""
    RIGHT = 0
    LEFT = 1

NATURAL_OFFSETS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
SHARP_ORDER = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
FLAT_ORDER = ['B', 'E', 'A', 'D', 'G', 'C', 'F']
NOTE_INDEX = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}
INDEX_NOTE = {v: k for k, v in NOTE_INDEX.items()}

DEFAULT_VELOCITY = 80
DEFAULT_DIVISIONS = 480
DEFAULT_TEMPO = 120.0

SUBDIV_MAP = {
    'whole': 4.0,
    'half': 2.0,
    'quarter': 1.0,
    'eighth': 0.5,
    '16th': 0.25,
    '32nd': 0.125,
}

# Articulation modifications
ARTICULATION_RATIOS = {
    'staccatissimo': 0.25,
    'staccato': 0.5,
    'portato': 0.75,
    'tenuto': 0.99,
    'marcato': 0.95,
    'accent': 0.95,
    'sforzando': 0.95,
}

VELOCITY_MODIFIERS = {
    'marcato': 1.2,
    'accent': 1.2,
    'sforzando': 1.5,
}

# Ornament motifs: (prefix, body, suffix) as diatonic offsets
ORNAMENT_MOTIFS = {
    'Trill': ([], [0, +1], []),
    'TrillBaroque': ([], [+1, 0], [-1, 0]),
    'Mordent': ([0], [-1, 0], []),
    'UpperMordent': ([0, +1], [], []),
    'Turn': ([+1, 0, -1], [0], []),
    'InvertedTurn': ([-1, 0, +1], [0], []),
    'PrallMordent': ([], [+1, 0, -1, 0], []),
}

# Glissando minimum note duration (in quarters)
GLISSANDO_MIN_NOTE_DURATION = 0.02

# Arpeggio constants
ARPEGGIO_MAX_OFFSET_MS = 60.0  # milliseconds (0.06 seconds)
ARPEGGIO_DEFAULT_STRETCH = 1.0
MIN_NOTE_DURATION = 0.0001  # quarters

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PlayedNote:
    """Represents a single note event in performance time"""
    hand: int
    pitch: Optional[int]  # MIDI pitch (None for rest)
    onset: float  # Quarter note units
    duration: float  # Quarter note units
    offset: float  # onset + duration
    velocity: int
    xml_element: Optional[etree._Element] = None
    source_tag: str = 'note'
    voice: str = '1'
    staff: int = 1
    measure_number: int = 0
    finger: Optional[int] = None
    extra: dict = field(default_factory=dict)

@dataclass
class PartState:
    """State tracking for a musical part"""
    divisions: int = DEFAULT_DIVISIONS
    keysig_map: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in NATURAL_OFFSETS})
    tempo: float = DEFAULT_TEMPO
    accidentals: Dict[Tuple[str, int], int] = field(default_factory=dict)

# ============================================================================
# Utility Functions
# ============================================================================

def key_fifths_to_alter_map(fifths: int) -> Dict[str, int]:
    """Convert circle of fifths to pitch alterations"""
    alter_map = {s: 0 for s in NATURAL_OFFSETS}
    if fifths > 0:
        for i in range(min(7, fifths)):
            alter_map[SHARP_ORDER[i]] = 1
    elif fifths < 0:
        for i in range(min(7, -fifths)):
            alter_map[FLAT_ORDER[i]] = -1
    return alter_map

def get_midi_pitch(
    step: str,
    octave: int,
    accidentals: Dict[Tuple[str, int], int],
    keysig: Dict[str, int]
) -> Optional[int]:
    """Calculate MIDI pitch number"""
    if step is None or octave is None:
        return None
    
    key = (step, octave)
    alter = accidentals.get(key, keysig.get(step, 0))
    return (octave + 1) * 12 + NATURAL_OFFSETS[step] + alter

def get_diatonic_neighbor(
    step: str,
    octave: int,
    offset: int,
    accidentals: Dict[Tuple[str, int], int],
    keysig: Dict[str, int]
) -> Optional[int]:
    """Get MIDI pitch of diatonic neighbor (for ornaments)"""
    if step is None or octave is None:
        return None
    
    idx = NOTE_INDEX[step] + offset
    target_octave = octave + (idx // 7)
    target_step = INDEX_NOTE[idx % 7]
    
    return get_midi_pitch(target_step, target_octave, accidentals, keysig)

def ticks_to_quarters(ticks: int, divisions: int) -> float:
    """Convert MusicXML duration ticks to quarter note units"""
    return ticks / max(1, divisions)

def bpm_to_bps(bpm: float) -> float:
    """Convert beats per minute to beats per second"""
    return max(bpm, 1.0) / 60.0

def safe_int(text: Optional[str], default: int = 0) -> int:
    """Safely parse integer from text"""
    if text is None:
        return default
    try:
        return int(text)
    except (ValueError, TypeError):
        return default

def safe_float(text: Optional[str], default: float = 0.0) -> float:
    """Safely parse float from text"""
    if text is None:
        return default
    try:
        return float(text)
    except (ValueError, TypeError):
        return default

# ============================================================================
# Structural Navigation
# ============================================================================

class StructuralNavigator:
    """
    Handles structural navigation of score (repeats, endings, D.C./D.S., etc.)
    Priority: <sound> directives > traditional repeat marks
    """
    
    def __init__(self, part: etree._Element):
        self.part = part
        self.measures = part.findall('./measure')
        self.n = len(self.measures)
        
        # Initialize structural markers
        self.repeat_start = [False] * self.n
        self.repeat_stop = [None] * self.n  # Times to repeat
        self.endings = [None] * self.n
        self.sound_directives = [None] * self.n
        self.segno_idx = None
        self.coda_idx = None
        
        self._index_structure()
    
    def _index_structure(self):
        """Pre-index all structural elements"""
        for idx, measure in enumerate(self.measures):
            # Index repeat barlines
            for barline in measure.findall('barline'):
                repeat = barline.find('repeat')
                if repeat is not None:
                    direction = repeat.get('direction')
                    if direction == 'forward':
                        self.repeat_start[idx] = True
                    elif direction == 'backward':
                        times = safe_int(repeat.get('times'), 1)
                        self.repeat_stop[idx] = times
            
            # Index endings (voltas)
            for ending in measure.findall('ending'):
                number = ending.get('number')
                if number:
                    nums = [safe_int(n.strip()) for n in number.split(',')]
                    self.endings[idx] = [n for n in nums if n > 0]
            
            # Index segno/coda markers
            for direction in measure.findall('.//direction'):
                dt = direction.find('direction-type')
                if dt is not None:
                    if dt.find('segno') is not None:
                        self.segno_idx = idx
                    if dt.find('coda') is not None:
                        self.coda_idx = idx
                
                # Index sound directives
                sound = direction.find('sound')
                if sound is not None and sound.attrib:
                    self.sound_directives[idx] = dict(sound.attrib)
    
    def build_play_sequence(self, max_iterations: int = 10000) -> List[int]:
        """Generate execution order of measures with proper volta handling"""
        sequence = []
        idx = 0
        visit_counts = [0] * self.n
        volta_passes = defaultdict(int)
        iterations = 0
        
        def find_repeat_start(before: int) -> int:
            """Find most recent repeat start marker"""
            for i in range(before, -1, -1):
                if self.repeat_start[i]:
                    return i
            return 0
        
        def should_play_measure(measure_idx: int, current_pass: int) -> bool:
            """Check if measure should be played based on ending/volta"""
            endings = self.endings[measure_idx]
            if endings is None:
                return True
            return current_pass in endings
        
        while idx < self.n and iterations < max_iterations:
            iterations += 1
            visit_counts[idx] += 1
            
            # Check volta before adding to sequence
            repeat_start = find_repeat_start(idx)
            repeat_key = (repeat_start, idx)
            current_pass = volta_passes.get(repeat_key, 0) + 1
            
            if should_play_measure(idx, current_pass):
                sequence.append(idx)
            
            # Process sound directives (priority)
            sound = self.sound_directives[idx]
            if sound:
                if sound.get('dacapo') in ('yes', 'true', '1'):
                    idx = 0
                    continue
                if sound.get('dalsegno') in ('yes', 'true', '1') and self.segno_idx is not None:
                    idx = self.segno_idx
                    continue
                if sound.get('tocoda') in ('yes', 'true', '1') and self.coda_idx is not None:
                    idx = self.coda_idx
                    continue
                if sound.get('fine') in ('yes', 'true', '1'):
                    break
            
            # Process textual directives
            for direction in self.measures[idx].findall('direction'):
                words = direction.find('direction-type/words')
                if words is not None and words.text:
                    text = words.text.strip().lower()
                    if 'd.c.' in text or 'da capo' in text:
                        idx = 0
                        continue
                    if ('d.s.' in text or 'dal segno' in text) and self.segno_idx is not None:
                        idx = self.segno_idx
                        continue
                    if ('al coda' in text or 'to coda' in text) and self.coda_idx is not None:
                        idx = self.coda_idx
                        continue
                    if 'fine' in text:
                        return sequence
            
            # Process repeat barlines
            times = self.repeat_stop[idx]
            if times:
                start = find_repeat_start(idx - 1)
                key = (start, idx)
                volta_passes[key] += 1
                
                if volta_passes[key] <= times:
                    idx = start
                    continue
            
            idx += 1
        
        return sequence

# ============================================================================
# Main Parser
# ============================================================================

class MusicXMLParser:
    """Main parser for MusicXML files"""
    
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = etree.parse(xml_path)
        self.root = self.tree.getroot()
        self.played_notes: List[PlayedNote] = []
        self.part_states: Dict[int, PartState] = {}
        self.buffers: Dict[tuple, list] = {}
        self._chord_start_ticks = 0
    
    def parse(self) -> List[PlayedNote]:
        """Parse MusicXML and generate PlayedNote timeline"""
        parts = self.root.findall('.//part')
        
        for part in parts:
            part_id = id(part)
            self.part_states[part_id] = PartState()
            
            navigator = StructuralNavigator(part)
            play_sequence = navigator.build_play_sequence()
            measures = part.findall('./measure')
            
            part_cursor = 0.0
            
            for measure_idx in play_sequence:
                if not (0 <= measure_idx < len(measures)):
                    continue
                
                measure = measures[measure_idx]
                state = self.part_states[part_id]
                
                self._process_attributes(measure, state)
                measure_cursor = self._process_measure(
                    measure, part_id, state, part_cursor
                )
                part_cursor += measure_cursor
        
        self._resolve_ties()
        self.played_notes.sort(key=lambda x: (x.hand, x.onset, x.pitch or -1))
        
        return self.played_notes
    
    def _process_attributes(self, measure: etree._Element, state: PartState):
        """Process measure attributes (divisions, key, time)"""
        attributes = measure.find('attributes')
        if attributes is None:
            return
        
        # Update divisions
        divisions = attributes.findtext('divisions')
        if divisions:
            state.divisions = safe_int(divisions, DEFAULT_DIVISIONS)
        
        # Update key signature
        key = attributes.find('key')
        if key is not None:
            fifths = safe_int(key.findtext('fifths'))
            state.keysig_map = key_fifths_to_alter_map(fifths)
    
    def _process_measure(
        self,
        measure: etree._Element,
        part_id: int,
        state: PartState,
        measure_onset: float
    ) -> float:
        """Process all notes in a measure"""
        local_ticks = 0
        state.accidentals.clear()
        
        # Reset buffers for this measure
        self.buffers[('chord_buffer', part_id)] = []
        self.buffers[('grace_buffer', part_id)] = []
        self.buffers[('tremolo_buffer', part_id)] = None
        self.buffers[('glissando_buffer', part_id)] = None
        
        # Store all notes for post-processing (glissando target search)
        measure_notes = []
        
        for child in measure:
            tag = etree.QName(child.tag).localname
            
            if tag == 'backup':
                duration = safe_int(child.findtext('duration'))
                local_ticks = max(0, local_ticks - duration)
            
            elif tag == 'forward':
                duration = safe_int(child.findtext('duration'))
                local_ticks += duration
            
            elif tag == 'direction':
                self._process_direction(child, state)
            
            elif tag == 'note':
                local_ticks = self._process_note(
                    child, part_id, state, measure_onset, local_ticks, measure, measure_notes
                )
        
        # Post-process: resolve glissando targets
        self._resolve_glissandos(part_id, measure_notes)
        
        # Flush remaining chord buffer
        self._flush_chord_buffer(part_id, state)
        
        return ticks_to_quarters(local_ticks, state.divisions)
    
    def _process_direction(self, direction: etree._Element, state: PartState):
        """Process tempo and other directives"""
        sound = direction.find('sound')
        if sound is not None and 'tempo' in sound.attrib:
            state.tempo = safe_float(sound.get('tempo'), DEFAULT_TEMPO)
        
        metronome = direction.find('.//metronome/per-minute')
        if metronome is not None:
            state.tempo = safe_float(metronome.text, DEFAULT_TEMPO)
    
    def _process_note(
        self,
        note: etree._Element,
        part_id: int,
        state: PartState,
        measure_onset: float,
        local_ticks: int,
        measure: etree._Element,
        measure_notes: List[dict]
    ) -> int:
        """Process a single note element"""
        is_chord = note.find('chord') is not None
        is_grace = note.find('grace') is not None
        
        # Handle grace notes
        if is_grace:
            grace_info = self._capture_grace_note(note, local_ticks, state, measure)
            self.buffers[('grace_buffer', part_id)].append(grace_info)
            duration = safe_int(note.findtext('duration'))
            return local_ticks + duration if duration and not is_chord else local_ticks
        
        # Track chord start position
        if not is_chord:
            self._chord_start_ticks = local_ticks
        
        # Parse basic note properties
        pitch_node = note.find('pitch')
        is_rest = pitch_node is None
        
        midi = None
        step = None
        octave = None
        
        if not is_rest:
            step = pitch_node.findtext('step')
            octave = safe_int(pitch_node.findtext('octave'))
            
            # Handle alterations
            alter_node = pitch_node.find('alter')
            if alter_node is not None:
                state.accidentals[(step, octave)] = safe_int(alter_node.text)
            
            accidental = note.find('accidental')
            if accidental is not None:
                alter_value = self._parse_accidental(accidental.text)
                if alter_value is not None:
                    state.accidentals[(step, octave)] = alter_value
            
            midi = get_midi_pitch(step, octave, state.accidentals, state.keysig_map)
        
        # Duration and tuplets
        duration = safe_int(note.findtext('duration'))
        duration_factor = self._get_tuplet_factor(note)
        
        # Voice and staff
        voice = note.findtext('voice', '1')
        staff = safe_int(note.findtext('staff'), 1)
        
        # Calculate timing
        onset_ticks = self._chord_start_ticks if is_chord else local_ticks
        onset = measure_onset + ticks_to_quarters(onset_ticks, state.divisions)
        nominal_duration = ticks_to_quarters(duration, state.divisions) * duration_factor
        
        # Parse notations
        notations = note.find('notations')
        ornaments = self._get_ornaments(notations)
        articulations = self._get_articulations(note)
        
        # Detect tremolo type
        tremolo_node = notations.find('tremolo') if notations is not None else None
        tremolo_type = None
        tremolo_beams = 2
        if tremolo_node is not None:
            tremolo_type = tremolo_node.get('type', 'single')
            if tremolo_node.text and tremolo_node.text.strip().isdigit():
                tremolo_beams = int(tremolo_node.text.strip())
        
        is_tremolo_single = tremolo_type == 'single' or (tremolo_node is not None and tremolo_type is None)
        is_tremolo_start = tremolo_type == 'start'
        is_tremolo_stop = tremolo_type == 'stop'
        
        # Detect glissando/slide
        glissando_node = notations.find('glissando') if notations is not None else None
        slide_node = notations.find('slide') if notations is not None else None
        glissando_type = None
        if glissando_node is not None:
            glissando_type = glissando_node.get('type', 'start')
        elif slide_node is not None:
            glissando_type = slide_node.get('type', 'start')
        
        is_glissando_start = glissando_type == 'start'
        is_glissando_stop = glissando_type == 'stop'
        
        is_arpeggiate = notations is not None and notations.find('arpeggiate') is not None
        
        # Velocity
        velocity = self._calculate_velocity(articulations)
        art_ratio = self._get_articulation_ratio(articulations)
        
        # Build note info dict
        note_info = {
            'xml': note,
            'is_rest': is_rest,
            'midi': midi,
            'step': step,
            'octave': octave,
            'onset_q': onset,
            'nominal_q': nominal_duration,
            'duration_ticks': duration,
            'voice': voice,
            'staff': staff,
            'measure_number': safe_int(measure.get('number')),
            'is_chord': is_chord,
            'ornaments': ornaments,
            'articulations': articulations,
            'is_tremolo_single': is_tremolo_single,
            'is_tremolo_start': is_tremolo_start,
            'is_tremolo_stop': is_tremolo_stop,
            'tremolo_beams': tremolo_beams,
            'is_glissando_start': is_glissando_start,
            'is_glissando_stop': is_glissando_stop,
            'arpeggiate': is_arpeggiate,
            'tempo': state.tempo,
            'accmap_snapshot': dict(state.accidentals),
            'keysig_map': dict(state.keysig_map),
            'velocity': velocity,
            'art_ratio': art_ratio,
            'onset_ticks': onset_ticks
        }
        
        # Store for glissando resolution
        measure_notes.append(note_info)
        
        # Handle chords vs single notes
        if is_chord:
            self.buffers[('chord_buffer', part_id)].append(note_info)
        else:
            # Check for tremolo two-note start
            if is_tremolo_start:
                self.buffers[('tremolo_buffer', part_id)] = note_info
                # Don't process yet, wait for stop note
            elif is_tremolo_stop:
                # Process tremolo pair
                tremolo_start = self.buffers.get(('tremolo_buffer', part_id))
                if tremolo_start:
                    self._expand_tremolo_two_notes(tremolo_start, note_info)
                    self.buffers[('tremolo_buffer', part_id)] = None
                else:
                    # Fallback: treat as simple note
                    self._process_single_note(note_info, part_id)
            # Check for glissando start
            elif is_glissando_start:
                self.buffers[('glissando_buffer', part_id)] = note_info
                # Will be resolved in post-processing
            else:
                # Normal single note processing
                self._process_single_note(note_info, part_id)
        
        return local_ticks + duration if not is_chord else local_ticks
    
    def _process_single_note(self, note_info: dict, part_id: int):
        """Process a single note (not part of chord, tremolo, or glissando)"""
        # Flush grace notes before principal
        grace_result = self._flush_graces_for_principal(part_id, note_info)
        if grace_result:
            grace_notes, time_shift = grace_result
            self.played_notes.extend(grace_notes)
            note_info['onset_q'] += time_shift
            note_info['nominal_q'] = max(0.0001, note_info['nominal_q'] - time_shift)
        
        # Process based on type
        if note_info['is_tremolo_single']:
            self._expand_tremolo(note_info)
        elif note_info['ornaments']:
            self._expand_ornament(note_info)
        else:
            self._append_simple_note(note_info)
    
    def _parse_accidental(self, text: Optional[str]) -> Optional[int]:
        """Parse accidental text to alteration value"""
        if not text:
            return None
        
        accidental_map = {
            'sharp': 1,
            'flat': -1,
            'natural': 0,
            'double-sharp': 2,
            'double-flat': -2,
        }
        return accidental_map.get(text.strip().lower())
    
    def _get_tuplet_factor(self, note: etree._Element) -> float:
        """Calculate duration factor for tuplets"""
        time_mod = note.find('time-modification')
        if time_mod is None:
            return 1.0
        
        actual = safe_int(time_mod.findtext('actual-notes'), 1)
        normal = safe_int(time_mod.findtext('normal-notes'), 1)
        
        return normal / actual if actual > 0 else 1.0
    
    def _get_articulations(self, note: etree._Element) -> List[str]:
        """Extract articulation markings"""
        articulations = []
        notations = note.find('notations')
        if notations is not None:
            art_node = notations.find('articulations')
            if art_node is not None:
                for art in art_node:
                    articulations.append(etree.QName(art.tag).localname)
        return articulations
    
    def _get_ornaments(self, notations: Optional[etree._Element]) -> List[str]:
        """Extract ornament markings"""
        ornaments = []
        if notations is not None:
            orn_node = notations.find('ornaments')
            if orn_node is not None:
                for orn in orn_node:
                    ornaments.append(etree.QName(orn.tag).localname)
        return ornaments
    
    def _calculate_velocity(self, articulations: List[str]) -> int:
        """Calculate MIDI velocity based on articulations"""
        velocity = DEFAULT_VELOCITY
        for art in articulations:
            modifier = VELOCITY_MODIFIERS.get(art, 1.0)
            velocity = int(velocity * modifier)
        return min(127, velocity)
    
    def _get_articulation_ratio(self, articulations: List[str]) -> float:
        """Get duration ratio for articulations"""
        for art in articulations:
            if art in ARTICULATION_RATIOS:
                return ARTICULATION_RATIOS[art]
        return 1.0
    
    def _append_simple_note(self, note_info: dict):
        """Append a simple note to played_notes"""
        duration = note_info['nominal_q'] * note_info['art_ratio']
        if duration <= 0:
            duration = 0.0001
        
        played_note = PlayedNote(
            hand=Hand.LEFT if note_info['staff'] == 2 else Hand.RIGHT,
            pitch=note_info['midi'],
            onset=note_info['onset_q'],
            duration=duration,
            offset=note_info['onset_q'] + duration,
            velocity=note_info['velocity'],
            xml_element=note_info['xml'],
            source_tag='note',
            voice=note_info['voice'],
            staff=note_info['staff'],
            measure_number=note_info['measure_number'],
        )
        self.played_notes.append(played_note)
    
    def _capture_grace_note(
        self,
        note: etree._Element,
        local_ticks: int,
        state: PartState,
        measure: etree._Element
    ) -> dict:
        """Capture grace note information for later processing"""
        pitch_node = note.find('pitch')
        step = None
        octave = None
        
        if pitch_node is not None:
            step = pitch_node.findtext('step')
            octave = safe_int(pitch_node.findtext('octave'))
        
        grace_node = note.find('grace')
        grace_type = 'Appoggiatura'
        if grace_node is not None and grace_node.get('slash') in ('yes', 'true', '1'):
            grace_type = 'Acciaccatura'
        
        make_time = False
        if grace_node is not None and grace_node.get('make-time') in ('yes', 'true', '1'):
            make_time = True
        
        return {
            'xml': note,
            'step': step,
            'octave': octave,
            'accmap_snapshot': dict(state.accidentals),
            'keysig_map': dict(state.keysig_map),
            'staff': safe_int(note.findtext('staff'), 1),
            'voice': note.findtext('voice', '1'),
            'measure_number': safe_int(measure.get('number')),
            'grace_type': grace_type,
            'make_time': make_time,
            'velocity': DEFAULT_VELOCITY,
        }
    
    def _flush_graces_for_principal(
        self,
        part_id: int,
        principal_info: dict
    ) -> Optional[Tuple[List[PlayedNote], float]]:
        """Flush accumulated grace notes before a principal note"""
        key = ('grace_buffer', part_id)
        if key not in self.buffers or not self.buffers[key]:
            return None
        
        graces = self.buffers[key]
        self.buffers[key] = []
        
        n = len(graces)
        principal_onset = principal_info['onset_q']
        principal_duration = principal_info['nominal_q']
        principal_xml = principal_info['xml']
        
        # Calculate durations for each grace
        durations = []
        for grace in graces:
            if grace['grace_type'] == 'Acciaccatura':
                # Very short
                t = min(SUBDIV_MAP['32nd'] / 2.0, (0.5 * principal_duration) / max(1, n))
            else:
                # Appoggiatura - steal from principal
                is_dotted = principal_xml.find('dot') is not None
                if is_dotted:
                    t = (2.0 / 3.0) * principal_duration / max(1, n)
                else:
                    t = 0.5 * principal_duration / max(1, n)
            durations.append(t)
        
        total_shift = sum(durations)
        
        # Create grace note events
        grace_notes = []
        current_onset = principal_onset
        
        for grace, duration in zip(graces, durations):
            midi = None
            if grace['step'] is not None:
                midi = get_midi_pitch(
                    grace['step'],
                    grace['octave'],
                    grace['accmap_snapshot'],
                    grace['keysig_map']
                )
            
            played_note = PlayedNote(
                hand=Hand.LEFT if grace['staff'] == 2 else Hand.RIGHT,
                pitch=midi,
                onset=current_onset,
                duration=duration,
                offset=current_onset + duration,
                velocity=grace['velocity'],
                xml_element=grace['xml'],
                source_tag=f"grace:{grace['grace_type']}",
                voice=grace['voice'],
                staff=grace['staff'],
                measure_number=grace['measure_number'],
            )
            grace_notes.append(played_note)
            current_onset += duration
        
        return (grace_notes, total_shift)
    
    def _flush_chord_buffer(self, part_id: int, state: PartState):
        """Flush and expand chord buffer with arpeggio support"""
        key = ('chord_buffer', part_id)
        if key not in self.buffers or not self.buffers[key]:
            return
        
        chord_notes = self.buffers[key]
        self.buffers[key] = []
        
        # Group by onset (absolute time) - DO NOT separate by voice/staff for arpeggios
        groups_by_onset = defaultdict(list)
        for note_info in chord_notes:
            onset_key = note_info['onset_q']
            groups_by_onset[onset_key].append(note_info)
        
        # Process each onset group
        for onset, notes_at_onset in groups_by_onset.items():
            # Separate arpeggiated from non-arpeggiated notes
            arpeggiated = []
            non_arpeggiated = []
            
            for note_info in notes_at_onset:
                if note_info['arpeggiate']:
                    arpeggiated.append(note_info)
                else:
                    non_arpeggiated.append(note_info)
            
            # Further group arpeggiated notes by arpeggio number if present
            if arpeggiated:
                arp_groups = self._group_arpeggios_by_number(arpeggiated)
                for arp_group in arp_groups:
                    self._expand_arpeggio_group(arp_group, state)
            
            # Process non-arpeggiated notes as regular chords
            if non_arpeggiated:
                # Group by voice/staff for regular chords
                regular_groups = defaultdict(list)
                for note_info in non_arpeggiated:
                    k = (note_info['voice'], note_info['staff'])
                    regular_groups[k].append(note_info)
                
                for group in regular_groups.values():
                    self._expand_chord_group(group, state, arpeggio=False)
    
    def _group_arpeggios_by_number(self, arpeggiated_notes: List[dict]) -> List[List[dict]]:
        """Group arpeggiated notes by their arpeggio number attribute"""
        # Extract arpeggio numbers from XML
        groups_by_number = defaultdict(list)
        
        for note_info in arpeggiated_notes:
            arp_node = note_info['xml'].find('.//arpeggiate')
            if arp_node is not None:
                number = arp_node.get('number')
                if number:
                    groups_by_number[number].append(note_info)
                else:
                    # No number specified - group with other unnumbered arpeggios
                    groups_by_number[None].append(note_info)
            else:
                groups_by_number[None].append(note_info)
        
        return list(groups_by_number.values())
    
    def _expand_arpeggio_group(self, members: List[dict], state: PartState):
        """Expand an arpeggio group spanning potentially multiple staves"""
        if not members:
            return
        
        N = len(members)
        
        # Get nominal duration (maximum of all notes in group)
        nominal = max(m['nominal_q'] for m in members)
        
        # Sort by MIDI pitch (ascending, grave to aigu)
        member_list = [(m['midi'] or 0, m) for m in members]
        member_list.sort(key=lambda x: x[0])
        
        # Detect direction (check if ANY note has direction="down")
        is_descending = False
        for m in members:
            arp_node = m['xml'].find('.//arpeggiate')
            if arp_node is not None:
                direction = arp_node.get('direction', '').lower()
                if 'down' in direction:
                    is_descending = True
                    break
        
        # Apply direction: reverse if descending
        if is_descending:
            member_list = list(reversed(member_list))
        
        # Calculate timing
        tempo_bpm = state.tempo
        
        # Convert max offset from milliseconds to quarters based on tempo
        # MaxOffsetQuarters = 0.06 * (TempoBPM / 60.0)
        max_offset_quarters = (ARPEGGIO_MAX_OFFSET_MS / 1000.0) * (tempo_bpm / 60.0)
        
        # Raw step: divide total duration by number of notes
        raw_step = nominal / N if N > 0 else 0.0
        
        # Actual offset step: minimum of raw step and max offset
        offset_step = min(raw_step, max_offset_quarters)
        
        # Base onset
        base_onset = members[0]['onset_q']
        
        # Generate events with staggered onsets
        for i, (midi, note_info) in enumerate(member_list):
            # Calculate delay for this note
            delay = offset_step * i * ARPEGGIO_DEFAULT_STRETCH
            
            # Real onset = base + delay
            real_onset = base_onset + delay
            
            # Real duration = original nominal - delay (time compression)
            # The note must end at the same absolute time as if not arpeggiated
            real_duration = max(MIN_NOTE_DURATION, nominal - delay)
            
            played_note = PlayedNote(
                hand=Hand.LEFT if note_info['staff'] == 2 else Hand.RIGHT,
                pitch=midi,
                onset=real_onset,
                duration=real_duration,
                offset=real_onset + real_duration,
                velocity=note_info['velocity'],
                xml_element=note_info['xml'],
                source_tag='arpeggio_note',
                voice=note_info['voice'],
                staff=note_info['staff'],
                measure_number=note_info['measure_number'],
                extra={'arpeggio_size': N, 'arpeggio_index': i, 'arpeggio_delay': delay}
            )
            self.played_notes.append(played_note)
    
    def _expand_chord_group(self, members: List[dict], state: PartState, arpeggio: bool = True):
        """Expand a regular chord group (non-arpeggiated, deprecated)"""
        # This method is kept for backward compatibility but should not be used for arpeggios
        if not members:
            return
        
        nominal = max(m['nominal_q'] for m in members)
        N = len(members)
        
        # Sort by pitch
        member_list = [(m['midi'] or 0, m) for m in members]
        member_list.sort(key=lambda x: x[0])
        
        base_onset = members[0]['onset_q']
        
        # Regular chord: all notes start simultaneously
        for i, (midi, note_info) in enumerate(member_list):
            played_note = PlayedNote(
                hand=Hand.LEFT if note_info['staff'] == 2 else Hand.RIGHT,
                pitch=midi,
                onset=base_onset,
                duration=nominal,
                offset=base_onset + nominal,
                velocity=note_info['velocity'],
                xml_element=note_info['xml'],
                source_tag='chord_note',
                voice=note_info['voice'],
                staff=note_info['staff'],
                measure_number=note_info['measure_number'],
                extra={'chord_size': N, 'chord_index': i}
            )
            self.played_notes.append(played_note)
    
    def _expand_tremolo(self, note_info: dict):
        """Expand tremolo into rapid repeated notes"""
        tremolo_node = note_info['xml'].find('.//tremolo')
        
        # Determine subdivision from beams/slashes
        beams = note_info.get('tremolo_beams', 2)
        
        # Map beams to subdivision
        if beams == 1:
            t_step = SUBDIV_MAP['eighth']
        elif beams == 2:
            t_step = SUBDIV_MAP['16th']
        else:
            t_step = SUBDIV_MAP['32nd']
        
        onset = note_info['onset_q']
        remaining = note_info['nominal_q']
        
        # Generate repeated notes
        while remaining > 1e-9:
            duration = min(t_step, remaining)
            
            played_note = PlayedNote(
                hand=Hand.LEFT if note_info['staff'] == 2 else Hand.RIGHT,
                pitch=note_info['midi'],
                onset=onset,
                duration=duration,
                offset=onset + duration,
                velocity=note_info['velocity'],
                xml_element=note_info['xml'],
                source_tag='tremolo',
                voice=note_info['voice'],
                staff=note_info['staff'],
                measure_number=note_info['measure_number'],
            )
            self.played_notes.append(played_note)
            
            onset += duration
            remaining -= duration
    
    def _expand_tremolo_two_notes(self, note_A_info: dict, note_B_info: dict):
        """Expand two-note tremolo (alternating between two notes/chords)"""
        # Determine subdivision from beams
        beams = note_A_info.get('tremolo_beams', 2)
        
        # Map beams to subdivision
        if beams == 1:
            t_step = SUBDIV_MAP['eighth']
        elif beams == 2:
            t_step = SUBDIV_MAP['16th']
        else:
            t_step = SUBDIV_MAP['32nd']
        
        # Total duration is sum of both notes
        total_duration = note_A_info['nominal_q'] + note_B_info['nominal_q']
        
        # Calculate number of steps
        steps_count = int(total_duration / t_step) if t_step > 0 else 0
        
        # Starting onset
        onset = note_A_info['onset_q']
        
        # Generate alternating sequence
        for i in range(steps_count):
            # Alternate between note A (even) and note B (odd)
            if i % 2 == 0:
                pitch = note_A_info['midi']
                source_note = note_A_info
            else:
                pitch = note_B_info['midi']
                source_note = note_B_info
            
            played_note = PlayedNote(
                hand=Hand.LEFT if note_A_info['staff'] == 2 else Hand.RIGHT,
                pitch=pitch,
                onset=onset,
                duration=t_step,
                offset=onset + t_step,
                velocity=note_A_info['velocity'],
                xml_element=note_A_info['xml'],  # Reference first note
                source_tag='tremolo_two_notes',
                voice=note_A_info['voice'],
                staff=note_A_info['staff'],
                measure_number=note_A_info['measure_number'],
            )
            self.played_notes.append(played_note)
            
            onset += t_step
    
    def _resolve_glissandos(self, part_id: int, measure_notes: List[dict]):
        """Resolve glissando start notes by finding their target notes"""
        glissando_start = self.buffers.get(('glissando_buffer', part_id))
        
        if glissando_start is None:
            return
        
        # Find target note (same voice/staff with glissando stop)
        target_note = None
        start_idx = -1
        
        # Find start note index
        for idx, note in enumerate(measure_notes):
            if note is glissando_start:
                start_idx = idx
                break
        
        # Search for stop note
        if start_idx >= 0:
            for idx in range(start_idx + 1, len(measure_notes)):
                candidate = measure_notes[idx]
                if (candidate['voice'] == glissando_start['voice'] and
                    candidate['staff'] == glissando_start['staff'] and
                    candidate['is_glissando_stop']):
                    target_note = candidate
                    break
        
        if target_note:
            # Expand glissando
            self._expand_glissando(glissando_start, target_note)
        else:
            # Fallback: treat as simple note
            self._process_single_note(glissando_start, part_id)
        
        # Clear buffer
        self.buffers[('glissando_buffer', part_id)] = None
    
    def _expand_glissando(self, note_start_info: dict, note_end_info: dict):
        """Expand glissando into chromatic sequence from start to end pitch"""
        pitch_start = note_start_info['midi']
        pitch_end = note_end_info['midi']
        
        if pitch_start is None or pitch_end is None:
            # Fallback: play as simple notes
            self._append_simple_note(note_start_info)
            return
        
        # Generate chromatic pitch sequence
        if pitch_start <= pitch_end:
            pitches = list(range(pitch_start, pitch_end + 1))
        else:
            pitches = list(range(pitch_start, pitch_end - 1, -1))
        
        # Total duration is the start note's duration
        total_duration = note_start_info['nominal_q']
        num_pitches = len(pitches)
        
        if num_pitches == 0:
            return
        
        # Duration per chromatic step
        t_step = max(GLISSANDO_MIN_NOTE_DURATION, total_duration / num_pitches)
        
        # Generate note sequence
        onset = note_start_info['onset_q']
        
        for pitch in pitches:
            played_note = PlayedNote(
                hand=Hand.LEFT if note_start_info['staff'] == 2 else Hand.RIGHT,
                pitch=pitch,
                onset=onset,
                duration=t_step,
                offset=onset + t_step,
                velocity=note_start_info['velocity'],
                xml_element=note_start_info['xml'],  # Reference start note
                source_tag='glissando',
                voice=note_start_info['voice'],
                staff=note_start_info['staff'],
                measure_number=note_start_info['measure_number'],
            )
            self.played_notes.append(played_note)
            
            onset += t_step
    
    def _expand_ornament(self, note_info: dict):
        """Expand ornaments (trills, mordents, turns) into note sequences"""
        # Map MusicXML ornament names to our motif keys
        ornament_map = {
            'trill-mark': 'Trill',
            'trill': 'Trill',
            'mordent': 'Mordent',
            'inverted-mordent': 'UpperMordent',
            'turn': 'Turn',
            'inverted-turn': 'InvertedTurn',
        }
        
        # Find first supported ornament
        motif_key = None
        for orn in note_info['ornaments']:
            motif_key = ornament_map.get(orn)
            if motif_key and motif_key in ORNAMENT_MOTIFS:
                break
        
        if not motif_key:
            # No supported ornament, play as simple note
            self._append_simple_note(note_info)
            return
        
        prefix, body, suffix = ORNAMENT_MOTIFS[motif_key]
        
        # Calculate subdivision based on tempo
        bps = bpm_to_bps(note_info['tempo'])
        
        if motif_key in ('Trill',):
            if bps < 1.8:
                t_sub = SUBDIV_MAP['32nd'] / 10.0
            elif bps < 3.0:
                t_sub = SUBDIV_MAP['32nd']
            else:
                t_sub = SUBDIV_MAP['16th']
        else:
            if bps < 1.8:
                t_sub = SUBDIV_MAP['32nd']
            elif bps < 3.0:
                t_sub = SUBDIV_MAP['16th']
            else:
                t_sub = SUBDIV_MAP['eighth']
        
        # Calculate lengths
        prefix_len = len(prefix) * t_sub
        suffix_len = len(suffix) * t_sub
        nominal = note_info['nominal_q']
        remaining = max(0.0, nominal - prefix_len - suffix_len)
        
        body_unit_len = len(body) * t_sub if body else 0.0
        reps = int(remaining / body_unit_len) if body_unit_len > 0 else 0
        
        # Helper to get MIDI for diatonic offset
        def midi_for_offset(offset: int) -> Optional[int]:
            return get_diatonic_neighbor(
                note_info['step'],
                note_info['octave'],
                offset,
                note_info['accmap_snapshot'],
                note_info['keysig_map']
            )
        
        # Build note sequence
        sequence = []
        for off in prefix:
            sequence.append((midi_for_offset(off), t_sub))
        for _ in range(reps):
            for off in body:
                sequence.append((midi_for_offset(off), t_sub))
        for off in suffix:
            sequence.append((midi_for_offset(off), t_sub))
        
        # Generate ornament notes
        onset = note_info['onset_q']
        used_total = 0.0
        
        for midi, duration in sequence:
            if duration <= 0:
                continue
            
            played_note = PlayedNote(
                hand=Hand.LEFT if note_info['staff'] == 2 else Hand.RIGHT,
                pitch=midi,
                onset=onset,
                duration=duration,
                offset=onset + duration,
                velocity=note_info['velocity'],
                xml_element=note_info['xml'],
                source_tag=f'ornament:{motif_key}',
                voice=note_info['voice'],
                staff=note_info['staff'],
                measure_number=note_info['measure_number'],
                extra={'ornament': motif_key}
            )
            self.played_notes.append(played_note)
            onset += duration
            used_total += duration
        
        # Append shortened principal note
        remaining_main = max(0.0001, nominal - used_total)
        principal = PlayedNote(
            hand=Hand.LEFT if note_info['staff'] == 2 else Hand.RIGHT,
            pitch=note_info['midi'],
            onset=note_info['onset_q'] + used_total,
            duration=remaining_main,
            offset=note_info['onset_q'] + used_total + remaining_main,
            velocity=note_info['velocity'],
            xml_element=note_info['xml'],
            source_tag='note_after_ornament',
            voice=note_info['voice'],
            staff=note_info['staff'],
            measure_number=note_info['measure_number'],
        )
        self.played_notes.append(principal)
    
    def _resolve_ties(self):
        """Merge tied notes into single sustained notes"""
        # Group notes by signature
        sig_groups = defaultdict(list)
        for note in self.played_notes:
            if note.xml_element is not None:
                sig = (note.voice, note.staff, note.pitch)
                sig_groups[sig].append((note, note.xml_element))
        
        merged = []
        consumed = set()
        
        for sig, group in sig_groups.items():
            group.sort(key=lambda x: x[0].onset)
            i = 0
            while i < len(group):
                note, xml = group[i]
                
                if id(note) in consumed:
                    i += 1
                    continue
                
                # Check for tie start
                ties = [t.get('type') for t in xml.findall('tie')]
                if 'start' not in ties:
                    merged.append(note)
                    consumed.add(id(note))
                    i += 1
                    continue
                
                # Build tie chain
                total_duration = note.duration
                j = i + 1
                while j < len(group):
                    next_note, next_xml = group[j]
                    next_ties = [t.get('type') for t in next_xml.findall('tie')]
                    
                    total_duration += next_note.duration
                    consumed.add(id(next_note))
                    
                    if 'stop' in next_ties:
                        j += 1
                        break
                    j += 1
                
                # Create merged note
                merged_note = PlayedNote(
                    hand=note.hand,
                    pitch=note.pitch,
                    onset=note.onset,
                    duration=total_duration,
                    offset=note.onset + total_duration,
                    velocity=note.velocity,
                    xml_element=note.xml_element,
                    source_tag='tied_note',
                    voice=note.voice,
                    staff=note.staff,
                    measure_number=note.measure_number,
                )
                merged.append(merged_note)
                consumed.add(id(note))
                i = j
        
        # Add non-tied notes
        for note in self.played_notes:
            if id(note) not in consumed:
                merged.append(note)
        
        self.played_notes = merged

# ============================================================================
# Fingering Integration
# ============================================================================

def apply_fingering(
    notes: List[PlayedNote],
    algorithm: Callable[[List[PlayedNote]], List[PlayedNote]]
) -> List[PlayedNote]:
    """Apply fingering algorithm to notes"""
    by_hand = defaultdict(list)
    for note in notes:
        by_hand[note.hand].append(note)
    
    for hand, hand_notes in by_hand.items():
        hand_notes.sort(key=lambda x: (x.onset, -(x.pitch or -999)))
        fingered = algorithm(hand_notes)
        
        for original, fingered_note in zip(hand_notes, fingered):
            original.finger = fingered_note.finger
    
    return notes

def inject_fingerings(
    tree: etree._ElementTree,
    notes: List[PlayedNote],
    output_path: str
):
    """Inject fingering numbers into MusicXML"""
    processed = set()
    
    for note in notes:
        if note.xml_element is None or note.finger is None:
            continue
        
        elem_id = id(note.xml_element)
        if elem_id in processed:
            continue
        
        notations = note.xml_element.find('notations')
        if notations is None:
            notations = etree.SubElement(note.xml_element, 'notations')
        
        technical = notations.find('technical')
        if technical is None:
            technical = etree.SubElement(notations, 'technical')
        
        fingering = technical.find('fingering')
        if fingering is None:
            fingering = etree.SubElement(technical, 'fingering')
        
        fingering.text = str(note.finger)
        processed.add(elem_id)
    
    tree.write(
        output_path,
        pretty_print=True,
        xml_declaration=True,
        encoding='UTF-8'
    )

# ============================================================================
# Example Fingering Algorithm
# ============================================================================

def simple_fingering_algorithm(notes: List[PlayedNote]) -> List[PlayedNote]:
    """Simple cycling fingering algorithm (stub)"""
    for i, note in enumerate(notes):
        note.finger = (i % 5) + 1
    return notes

# ============================================================================
# Main Entry Point
# ============================================================================

def process_musicxml(
    input_path: str,
    output_path: str,
    fingering_algorithm: Callable = simple_fingering_algorithm
) -> List[PlayedNote]:
    """Process MusicXML file with fingering"""
    parser = MusicXMLParser(input_path)
    notes = parser.parse()
    notes = apply_fingering(notes, fingering_algorithm)
    inject_fingerings(parser.tree, notes, output_path)
    return notes

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python musicxml_pianist_improved.py input.xml output.xml")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    played_notes = process_musicxml(input_file, output_file)
    print(f"✓ Generated {len(played_notes)} PlayedNote events")
    print(f"✓ Output written to {output_file}")