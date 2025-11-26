#!/usr/bin/env python3
# musicxml_pianist_parser.py
# Dependencies: lxml
from lxml import etree
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from collections import defaultdict
import math
import sys

# ------------------------------
# Data classes
# ------------------------------
@dataclass
class PlayedNote:
    hand: int                     # 0 = right, 1 = left
    pitch: int                    # MIDI number
    onset: float                  # in fraction of quarter note (quarter = 1.0)
    duration: float               # in fraction of quarter note
    offset: float                 # onset + duration
    velocity: int                 # MIDI velocity (default)
    xml_element: Optional[etree._Element]  # source <note> if present (None for derived but often present)
    source_tag: str               # 'note','acciaccatura','ornament','trill', etc.
    voice: str
    staff: int
    measure_number: int
    extra: dict = field(default_factory=dict)
    finger: Optional[int] = None  # to be filled by fingering algorithm

# ------------------------------
# Constants & helpers tonal
# ------------------------------
NATURAL_OFFSETS = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
SHARP_ORDER = ['F','C','G','D','A','E','B']
FLAT_ORDER  = ['B','E','A','D','G','C','F']

DEFAULT_VELOCITY = 80

def key_fifths_to_alter_map(fifths:int)->Dict[str,int]:
    m = {s:0 for s in NATURAL_OFFSETS}
    if fifths>0:
        for i in range(min(7,fifths)):
            m[SHARP_ORDER[i]] = 1
    elif fifths<0:
        for i in range(min(7,-fifths)):
            m[FLAT_ORDER[i]] = -1
    return m

# Return midi using AccMap + KeySig rules
def get_midi(step:str, octave:int, accmap:Dict[Tuple[str,int],int], keysig_map:Dict[str,int]) -> int:
    if step is None or octave is None:
        return None
    key = (step, octave)
    if key in accmap:
        alter = accmap[key]
    else:
        alter = keysig_map.get(step, 0)
    return (octave + 1) * 12 + NATURAL_OFFSETS[step] + alter

# Diatonic neighbor (k steps)
NOTE_INDEX = {'C':0,'D':1,'E':2,'F':3,'G':4,'A':5,'B':6}
INDEX_NOTE = {v:k for k,v in NOTE_INDEX.items()}
def get_diatonic_neighbor(step:str, octave:int, k:int, accmap:Dict[Tuple[str,int],int], keysig_map:Dict[str,int]) -> int:
    if step is None or octave is None:
        return None
    idx = NOTE_INDEX[step] + k
    oct_delta = idx // 7
    idx_mod = idx % 7
    target_step = INDEX_NOTE[idx_mod]
    target_oct = octave + oct_delta
    return get_midi(target_step, target_oct, accmap, keysig_map)

# convert xml duration (ticks) + divisions -> quarter fraction
def ticks_to_quarters(ticks:int, divisions:int) -> float:
    if divisions <= 0: divisions = 1
    return ticks / divisions

# tempo BPM -> BPS (beats per second)
def bpm_to_bps(bpm:float) -> float:
    if bpm is None or bpm <= 0: return 1.0
    return bpm / 60.0

# ------------------------------
# Ornaments motif table (diatonic offsets sequences)
# motifs are lists: (prefix list, body list, suffix list)
# offset values are diatonic offsets relative to main note
# ------------------------------
ORNAMENT_MOTIFS = {
    'Trill': ([], [0, +1], []),
    'TrillBaroque': ([], [+1,0], [-1,0]),
    'Mordent': ([0], [ -1, 0], []),  # normalized
    'UpperMordent': ([0, +1], [], []),
    'UpMordent': ([-1,0],[+1,0],[-1,0]),
    'Turn': ([+1,0,-1], [0], []),
    'InvertedTurn': ([-1,0,+1],[0],[]),
    'PrallMordent': ([], [+1,0,-1,0], []),
    'LinePrall': ([+2,+2,+2], [+1,0], [+1,0]),
    # add more as needed from user's list...
}

# Minimum supported smallest subdivision preference (in ticks/quaver units)
# We'll map types to default minimal note duration types for T_sub selection
ORNAMENT_MIN_DURATION_PREFERENCE = {
    'Trill': 'semiquaver_div10',    # triple-croche/10 special
    'TrillBaroque': 'semiquaver_div10',
    'Mordent': 'semiquaver',        # triple-croche
    'MordentUp': 'semiquaver',
    'Turn': 'semiquaver',
    'Prall': 'semiquaver',
    # default -> semiquaver
}

# Map textual subdivision names -> fraction of quarter
SUBDIV_MAP = {
    'semibreve': 4.0,
    'minim': 2.0,
    'quarter': 1.0,
    'quaver': 0.5,          # croche
    'semiquaver': 0.25,     # double-croche
    'demisemiquaver': 0.125,# triple-croche
}

# ------------------------------
# Parser + Expander class
# ------------------------------
class PianistMusicXMLParser:
    def __init__(self, xml_path:str):
        self.xml_path = xml_path
        self.tree = etree.parse(xml_path)
        self.root = self.tree.getroot()
        self.played_notes: List[PlayedNote] = []
        # per part states, keyed by part id
        self.part_states = defaultdict(lambda: {
            'divisions': 1,
            'keysig_map': {s:0 for s in NATURAL_OFFSETS},
            'tempo': 120.0,
            'accmap': defaultdict(int),  # reset each measure
        })

    # main entry
    def parse_and_expand(self):
        parts = self.root.findall('.//part')
        for part in parts:
            part_id = part.get('id') or 'P'
            # iterate measures sequentially
            absolute_measure_index = 0
            for measure in part.findall('./measure'):
                absolute_measure_index += 1
                self._process_measure(part_id, measure, absolute_measure_index)
        # sort final timeline
        self.played_notes.sort(key=lambda x:(x.hand, x.onset, x.pitch))
        return self.played_notes

    def _process_measure(self, part_id:str, measure:etree._Element, measure_number:int):
        state = self.part_states[part_id]
        # update attributes (divisions/key/clef) if any
        attr = measure.find('attributes')
        if attr is not None:
            div = attr.findtext('divisions')
            if div is not None:
                state['divisions'] = int(div)
            key = attr.find('key')
            if key is not None:
                fifths = key.findtext('fifths')
                if fifths is not None:
                    state['keysig_map'] = key_fifths_to_alter_map(int(fifths))
            # clef ignored for hand heuristics in this version (staff used)
        # reset measure-level accidental map
        accmap = defaultdict(int)
        state['accmap'] = accmap

        # prepare a cursor in ticks (relative to measure). We'll convert to quarters by dividing by divisions.
        cursor_ticks = 0
        last_chord_start_ticks = None

        # scan children in measure order (forward/backup supported)
        for child in measure:
            tag = etree.QName(child.tag).localname
            if tag == 'forward':
                dur_node = child.find('duration')
                if dur_node is not None:
                    cursor_ticks += int(dur_node.text)
                continue
            if tag == 'backup':
                dur_node = child.find('duration')
                if dur_node is not None:
                    cursor_ticks = max(0, cursor_ticks - int(dur_node.text))
                continue
            if tag == 'direction':
                self._apply_direction_to_state(child, state)
                continue
            if tag != 'note':
                continue
            note_elem = child
            # chord?
            is_chord = note_elem.find('chord') is not None
            if not is_chord:
                last_chord_start_ticks = cursor_ticks
            start_ticks = last_chord_start_ticks if is_chord else cursor_ticks

            # parse pitch or rest
            pitch_node = note_elem.find('pitch')
            is_rest = pitch_node is None
            step = None; octave = None; alter = None
            if not is_rest:
                step = pitch_node.findtext('step')
                oct_txt = pitch_node.findtext('octave')
                octave = int(oct_txt) if oct_txt is not None else None
                alt_node = pitch_node.find('alter')
                if alt_node is not None:
                    alter = int(float(alt_node.text))
                # explicit accidental overrides measure accidental rules
                acc_node = note_elem.find('accidental')
                if acc_node is not None and acc_node.text:
                    txt = acc_node.text.strip().lower()
                    if txt == 'sharp': alter = 1
                    elif txt == 'flat': alter = -1
                    elif txt == 'natural': alter = 0
                    elif txt == 'double-sharp': alter = 2
                    elif txt == 'double-flat': alter = -2
                if alter is not None:
                    accmap[(step,octave)] = alter
            # duration ticks (grace may omit)
            dur_node = note_elem.find('duration')
            duration_ticks = int(dur_node.text) if dur_node is not None else 0
            # voice/staff
            voice = note_elem.findtext('voice','1')
            staff = int(note_elem.findtext('staff','1'))
            # ornament detection & articulations
            notations = note_elem.find('notations')
            ornaments = []
            articulations = []
            if notations is not None:
                orn_node = notations.find('ornaments')
                if orn_node is not None:
                    for o in orn_node:
                        ornaments.append(etree.QName(o.tag).localname)
                art_node = notations.find('articulations')
                if art_node is not None:
                    for a in art_node:
                        articulations.append(etree.QName(a.tag).localname)
                # staccato might also be under articulations
            # grace?
            grace_node = note_elem.find('grace')
            is_grace = grace_node is not None
            grace_type = None
            make_time = False
            if is_grace:
                # detect acciaccatura slash
                if 'slash' in grace_node.attrib and grace_node.get('slash') in ('yes','true','1'):
                    grace_type = 'Acciaccatura'
                else:
                    grace_type = 'Appoggiatura'
                if 'make-time' in grace_node.attrib and grace_node.get('make-time') in ('yes','true','1'):
                    make_time = True

            # compute midi
            midi = None
            if not is_rest:
                midi = get_midi(step, octave, accmap, state['keysig_map'])

            # build a basic note record (not yet expanded)
            nominal_quarters = ticks_to_quarters(duration_ticks, state['divisions']) if duration_ticks>0 else 0.0
            onset_quarters = ticks_to_quarters(start_ticks, state['divisions'])
            tempo = state.get('tempo', 120.0)
            # default velocity, may be changed by articulations
            vel = DEFAULT_VELOCITY

            # handle articulations simple ratio changes
            art_ratio = 1.0
            if 'staccatissimo' in articulations:
                art_ratio = 0.25
            elif 'staccato' in articulations:
                art_ratio = 0.5
            elif 'portato' in articulations:
                art_ratio = 0.75
            elif 'tenuto' in articulations:
                art_ratio = 0.99
            elif 'marcato' in articulations or 'accent' in articulations:
                art_ratio = 0.95
                vel = int(vel * 1.2)
            # sforzando or subito
            if 'sforzando' in articulations or 'subito' in articulations:
                art_ratio = 0.95
                vel = int(vel * 1.5)

            # if it's a tremolo (notations->tremolo) or note has <tremolo> children, handle in expand_tremolo
            is_tremolo = note_elem.find('tremolo') is not None

            # Save a compact record for later expansion
            base_note = {
                'xml': note_elem,
                'is_rest': is_rest,
                'midi': midi,
                'onset_q': onset_quarters,
                'nominal_q': nominal_quarters,
                'duration_ticks': duration_ticks,
                'voice': voice,
                'staff': staff,
                'measure_number': measure_number,
                'is_chord': is_chord,
                'is_grace': is_grace,
                'grace_type': grace_type,
                'make_time': make_time,
                'ornaments': ornaments,
                'articulations': articulations,
                'is_tremolo': is_tremolo,
                'tempo': tempo,
                'accmap_snapshot': dict(accmap),  # for ornament pitch resolution
                'keysig_map': dict(state['keysig_map']),
                'velocity': vel
            }

            # Expand according to type: grace notes / ornaments / tremolo / glissando / normal / chord handled later
            # If grace: we emit a PlayedNote immediately but defer time-stealing application until all graces for the principal are known.
            if base_note['is_grace']:
                # convert to PlayedNote with duration per rules later in grace processing stage
                # store base note to a buffer attached to measure for post-processing
                # We'll annotate the xml node with a temporary attribute to link later
                self._append_grace_candidate(part_id, base_note)
                # chord cursor: do not advance measure cursor
                if not is_chord:
                    cursor_ticks += duration_ticks
                continue

            # Tremolo handling
            if base_note['is_tremolo']:
                self._expand_tremolo_and_append(part_id, base_note)
                if not is_chord:
                    cursor_ticks += duration_ticks
                continue

            # Glissando detection (look for 'glissando' in notations or a <slide> element)
            if note_elem.find('slide') is not None or note_elem.find('.//glissando') is not None:
                # For simplicity: look for notation end target via <slide> or tied sequence; best-effort
                self._expand_glissando_and_append(part_id, base_note)
                if not is_chord:
                    cursor_ticks += duration_ticks
                continue

            # Ornaments handling (trill/mordent/turn/etc.)
            if base_note['ornaments']:
                self._expand_ornament_and_append(part_id, base_note)
                if not is_chord:
                    cursor_ticks += duration_ticks
                continue

            # Arpeggio detection: if notations contain 'arpeggiate' then expand chord on flush
            if base_note['is_chord']:
                # accumulate to temporary chord buffer in measure-level structure
                self._append_chord_candidate(part_id, base_note)
            else:
                # normal standalone note -> append PlayedNote directly applying articulation ratio
                duration_real = nominal_quarters * art_ratio
                pn = PlayedNote(
                    hand = 1 if staff==2 else 0,
                    pitch = base_note['midi'],
                    onset = onset_quarters,
                    duration = duration_real,
                    offset = onset_quarters + duration_real,
                    velocity = base_note['velocity'],
                    xml_element = base_note['xml'],
                    source_tag = 'note',
                    voice = base_note['voice'],
                    staff = base_note['staff'],
                    measure_number = base_note['measure_number'],
                    extra = {'orig_nominal': nominal_quarters, 'art_ratio': art_ratio}
                )
                self.played_notes.append(pn)
                if not is_chord:
                    cursor_ticks += duration_ticks

        # end measure children
        # flush any chord buffer present in this measure (we kept chord buffers in an attribute)
        self._flush_chord_buffers_for_measure(part_id, measure_number, state)
        # after finishing measure, reset measure-level accidental rules accmap (as per MusicXML)
        state['accmap'] = defaultdict(int)

    # ------------------------------------------------------------------
    # Buffers & expansions utilities
    # The implementation uses simple ephemeral buffers stored on self keyed by part+measure to collect
    # chord members and grace candidates for post-processing. For simplicity these buffers are dictionaries.
    # ------------------------------------------------------------------
    def _append_chord_candidate(self, part_id, base_note):
        key = ('chord_buffer', part_id)
        if not hasattr(self, '_buffers'):
            self._buffers = {}
        if key not in self._buffers:
            self._buffers[key] = []
        self._buffers[key].append(base_note)

    def _flush_chord_buffers_for_measure(self, part_id, measure_number, state):
        key = ('chord_buffer', part_id)
        if not hasattr(self, '_buffers') or key not in self._buffers:
            return
        chord_buffer = self._buffers.pop(key)
        # partition buffer into sequential chord groups using same start tick (we assume they were added in order)
        # group by (onset_q, voice, staff)
        groups = defaultdict(list)
        for n in chord_buffer:
            k = (n['onset_q'], n['voice'], n['staff'])
            groups[k].append(n)
        for k, members in groups.items():
            # detect arpeggiation flag among members
            has_arpeggio = any(m['xml'].find('.//notations/arpeggiate') is not None or m['xml'].find('arpeggiate') is not None for m in members)
            self._expand_chord_members_and_append(members, has_arpeggio, state)

    def _append_grace_candidate(self, part_id, base_note):
        key = ('grace_buffer', part_id)
        if not hasattr(self, '_buffers'):
            self._buffers = {}
        if key not in self._buffers:
            self._buffers[key] = []
        self._buffers[key].append(base_note)

    def _flush_graces_for_principal(self, part_id, principal_start_q, principal_nominal_q, principal_xml):
        # find grace buffer for this part and principal onset
        key = ('grace_buffer', part_id)
        if not hasattr(self, '_buffers') or key not in self._buffers:
            return []
        buf = self._buffers[key]
        # select those graces that precede this principal (we used rule: grace notes are attached directly before principal in measure order)
        # For robustness, pick all graces currently in buffer and apply them sequentially at principal_onset
        graces = buf[:]  # copy
        self._buffers[key] = []  # clear buffer after consuming
        # compute durations per rules
        n = len(graces)
        res = []
        if n == 0:
            return res
        # detect types and compute T_grace per rule
        # convert nominal to quarter fraction
        T_nom = principal_nominal_q
        # for each grace compute duration
        durations = []
        for g in graces:
            if g['grace_type'] == 'Acciaccatura':
                # triple-croche = demisemiquaver = 0.125 (quarter fractions is 1/8? careful)
                # Here triple-croche = demisemiquaver? we use demisemiquaver = 0.125 (quarter fraction)
                # rule: min(durée_triple_croche / 2, (0.5 * T_nom) / n)
                triple = SUBDIV_MAP.get('demisemiquaver', 0.125)
                t = min(triple/2.0, (0.5 * T_nom) / max(1,n))
            else:
                # Appoggiatura
                # Check composite meter? For now use the rule: if principal dotted (we inspect xml dot) and meter composite detection omitted -> use 2/3
                dotted = principal_xml.find('dot') is not None
                if dotted:
                    t = (2.0/3.0) * T_nom / max(1,n)
                else:
                    t = 0.5 * T_nom / max(1,n)
            durations.append(t)
        # assign sequential onsets starting at principal onset (on-beat)
        current = principal_start_q
        for i,g in enumerate(graces):
            midi = get_midi(g.get('xml').findtext('pitch/step') if g.get('xml').find('pitch') is not None else None,
                            int(g.get('xml').findtext('pitch/octave')) if g.get('xml').find('pitch') is not None else None,
                            g['accmap_snapshot'], g['keysig_map']) if g.get('xml').find('pitch') is not None else None
            pn = PlayedNote(
                hand = 1 if g['staff']==2 else 0,
                pitch = midi,
                onset = current,
                duration = durations[i],
                offset = current + durations[i],
                velocity = g['velocity'],
                xml_element = g['xml'],
                source_tag = 'grace',
                voice = g['voice'],
                staff = g['staff'],
                measure_number = g['measure_number'],
                extra = {'grace_type': g['grace_type']}
            )
            res.append(pn)
            current += durations[i]
        # main note will be shifted by sum(durations)
        return res, sum(durations)

    # ------------------------------------------------------------------
    # Expansion routines (ornaments, tremolo, glissando, chord arpeggios)
    # ------------------------------------------------------------------
    def _expand_ornament_and_append(self, part_id, base_note):
        # For each ornament in base_note['ornaments'] expand according to motif table
        for orn in base_note['ornaments']:
            orn_name = orn  # string like 'trill-mark' etc. translate to motif keys
            # normalize common names
            keymap = {
                'trill-mark':'Trill', 'trill':'Trill','mordent':'Mordent','inverted-mordent':'UpperMordent',
                'turn':'Turn','inverted-turn':'InvertedTurn','prall':'Prall','prall-mordent':'PrallMordent'
            }
            motif_key = keymap.get(orn_name, None)
            if motif_key is None:
                # fallback: treat as simple note
                self._append_simple_note_from_base(base_note)
                return
            motif = ORNAMENT_MOTIFS.get(motif_key)
            if motif is None:
                # fallback
                self._append_simple_note_from_base(base_note)
                return
            # compute T_sub based on tempo and type (respect Correction 1)
            bpm = base_note.get('tempo', 120.0)
            bps = bpm_to_bps(bpm)
            # default prefer triple-croche
            if motif_key in ('Trill','TrillBaroque'):
                # Trill uses special fast slow rule: demisemiquaver / 10 for very slow only for Trill
                if bps < 1.8:
                    # use demisemiquaver / 10 -> triple-croche / 10 for very slow
                    T_sub = SUBDIV_MAP.get('demisemiquaver', 0.125) / 10.0
                elif bps < 3.0:
                    T_sub = SUBDIV_MAP.get('demisemiquaver', 0.125)
                else:
                    T_sub = SUBDIV_MAP.get('semiquaver', 0.25)
            else:
                # Mordents/turns prefer triple-croche baseline
                if bps < 1.8:
                    T_sub = SUBDIV_MAP.get('demisemiquaver', 0.125)
                elif bps < 3.0:
                    T_sub = SUBDIV_MAP.get('semiquaver', 0.25)
                else:
                    T_sub = SUBDIV_MAP.get('quaver', 0.5)

            # build prefix/body/suffix sequences translated to midis using accmap snapshot + keysig_map
            prefix, body, suffix = motif
            # compute prefix duration
            prefix_len = len(prefix) * T_sub
            suffix_len = len(suffix) * T_sub
            nominal = base_note['nominal_q']
            remaining = max(0.0, nominal - prefix_len - suffix_len)
            body_unit_len = T_sub * max(1, len(body))
            n_reps = int(remaining // body_unit_len) if body_unit_len>0 else 0

            sequence = []
            # prefix
            for off in prefix:
                midi = get_diatonic_neighbor(base_note.get('xml').findtext('pitch/step') if base_note.get('xml').find('pitch') is not None else None,
                                             int(base_note.get('xml').findtext('pitch/octave')) if base_note.get('xml').find('pitch') is not None else None,
                                             off, base_note['accmap_snapshot'], base_note['keysig_map']) if base_note.get('xml').find('pitch') is not None else None
                sequence.append((midi, T_sub))
            # body repeated
            for _ in range(n_reps):
                for off in body:
                    midi = get_diatonic_neighbor(base_note.get('xml').findtext('pitch/step') if base_note.get('xml').find('pitch') is not None else None,
                                                 int(base_note.get('xml').findtext('pitch/octave')) if base_note.get('xml').find('pitch') is not None else None,
                                                 off, base_note['accmap_snapshot'], base_note['keysig_map']) if base_note.get('xml').find('pitch') is not None else None
                    sequence.append((midi, T_sub))
            # suffix
            for off in suffix:
                midi = get_diatonic_neighbor(base_note.get('xml').findtext('pitch/step') if base_note.get('xml').find('pitch') is not None else None,
                                             int(base_note.get('xml').findtext('pitch/octave')) if base_note.get('xml').find('pitch') is not None else None,
                                             off, base_note['accmap_snapshot'], base_note['keysig_map']) if base_note.get('xml').find('pitch') is not None else None
                sequence.append((midi, T_sub))
            # build PlayedNotes sequence starting at base onset
            onset = base_note['onset_q']
            for midi, dur in sequence:
                pn = PlayedNote(
                    hand = 1 if base_note['staff']==2 else 0,
                    pitch = midi,
                    onset = onset,
                    duration = dur,
                    offset = onset + dur,
                    velocity = base_note['velocity'],
                    xml_element = base_note['xml'],
                    source_tag = f'ornament:{motif_key}',
                    voice = base_note['voice'],
                    staff = base_note['staff'],
                    measure_number = base_note['measure_number'],
                    extra = {'ornament': motif_key}
                )
                self.played_notes.append(pn)
                onset += dur
            # principal note will be shortened or handled by caller (we keep original main note but shorten below)
            # For simplicity: we do not implicitly append the main note here; caller (original flow) should ensure main note handling
            return

    def _expand_tremolo_and_append(self, part_id, base_note):
        # read <tremolo> element and number of strokes
        tnode = base_note['xml'].find('tremolo')
        if tnode is None:
            self._append_simple_note_from_base(base_note)
            return
        # determine stroke count by number of <tremolo> inner text or attributes; fallback to typical
        # in MusicXML, <tremolo> contains <tremolo>child-level/#text = number of strokes? we check 'type' attribute or child value
        # For safety: if <tremolo> has a 'type' or integer value, use mapping; else treat as 16th repetitions by default
        # We'll map number of beams: 1 -> 8th, 2->16th,3->32th
        # Default step = semiquaver (0.25)
        beams = 2
        try:
            beams = int(tnode.text) if tnode.text is not None else beams
        except:
            beams = beams
        if beams == 1:
            T_step = SUBDIV_MAP.get('quaver',0.5)
        elif beams == 2:
            T_step = SUBDIV_MAP.get('semiquaver',0.25)
        else:
            T_step = SUBDIV_MAP.get('demisemiquaver',0.125)
        # repeat main note every T_step until nominal filled
        onset = base_note['onset_q']
        remaining = base_note['nominal_q']
        while remaining > 1e-9:
            dur = min(T_step, remaining)
            pn = PlayedNote(
                hand = 1 if base_note['staff']==2 else 0,
                pitch = base_note['midi'],
                onset = onset,
                duration = dur,
                offset = onset + dur,
                velocity = base_note['velocity'],
                xml_element = base_note['xml'],
                source_tag = 'tremolo',
                voice = base_note['voice'],
                staff = base_note['staff'],
                measure_number = base_note['measure_number'],
                extra = {'tremolo_beams': beams}
            )
            self.played_notes.append(pn)
            onset += dur
            remaining -= dur

    def _expand_glissando_and_append(self, part_id, base_note):
        # naive best-effort: find next tied or adjacent note in same voice that has slide/gliss target; otherwise skip
        # For demonstration we'll skip complex detection and append the base note as-is
        self._append_simple_note_from_base(base_note)

    def _append_simple_note_from_base(self, base_note):
        # apply articulation ratio if any
        art_ratio = 1.0
        if base_note['articulations']:
            if 'staccatissimo' in base_note['articulations']:
                art_ratio = 0.25
            elif 'staccato' in base_note['articulations']:
                art_ratio = 0.5
            elif 'portato' in base_note['articulations']:
                art_ratio = 0.75
            elif 'tenuto' in base_note['articulations']:
                art_ratio = 0.99
            elif 'marcato' in base_note['articulations'] or 'accent' in base_note['articulations']:
                art_ratio = 0.95
                base_note['velocity'] = int(base_note['velocity'] * 1.2)
        duration_real = base_note['nominal_q'] * art_ratio
        pn = PlayedNote(
            hand = 1 if base_note['staff']==2 else 0,
            pitch = base_note['midi'],
            onset = base_note['onset_q'],
            duration = duration_real,
            offset = base_note['onset_q'] + duration_real,
            velocity = base_note['velocity'],
            xml_element = base_note['xml'],
            source_tag = 'note',
            voice = base_note['voice'],
            staff = base_note['staff'],
            measure_number = base_note['measure_number'],
            extra = {'orig_nominal': base_note['nominal_q']}
        )
        self.played_notes.append(pn)

    def _expand_chord_members_and_append(self, members:list, has_arpeggio:bool, state:dict):
        # members: list of base_note dicts sharing same onset_q and staff/voice
        # compute nominal (take max nominal among members for safety)
        if not members:
            return
        nominal = max(m['nominal_q'] for m in members)
        N = len(members)
        # tempo -> BPS for MaxStepQuarters
        bpm = members[0].get('tempo', 120.0)
        bps = bpm_to_bps(bpm)
        max_step_quarters = 0.06 * bps  # Correction 2 applied
        if has_arpeggio:
            step_time = min(nominal / N if N>0 else 0.0, max_step_quarters)
        else:
            step_time = 0.0
        # sort according to arpeggiate direction; default ascending by pitch
        # retrieve midi for each member
        member_infos = []
        for m in members:
            member_infos.append((m['midi'], m))
        member_infos.sort(key=lambda x:x[0])  # asc
        # if some member xml indicates arpeggiate-down, invert
        # simple heuristic: if any member has <arpeggiate type="down"> then reverse
        arpeggiate_down = any(m['xml'].find('.//arpeggiate') is not None and m['xml'].find('.//arpeggiate').get('direction')=='down' for _,m in member_infos)
        ordered = [mi for mi,_ in (reversed(member_infos) if arpeggiate_down else member_infos)]
        # expand each with onset shift and adjusted duration (end same absolute time)
        base_onset = members[0]['onset_q']
        for i, (midi, m) in enumerate(member_infos if not arpeggiate_down else list(reversed(member_infos))):
            # compute onset_i
            onset_i = base_onset + i * step_time
            duration_i = max(0.0001, nominal - i * step_time)
            pn = PlayedNote(
                hand = 1 if m['staff']==2 else 0,
                pitch = midi,
                onset = onset_i,
                duration = duration_i,
                offset = onset_i + duration_i,
                velocity = m['velocity'],
                xml_element = m['xml'],
                source_tag = 'arp_chord' if has_arpeggio else 'chord_note',
                voice = m['voice'],
                staff = m['staff'],
                measure_number = m['measure_number'],
                extra = {'chord_size': N, 'chord_index': i}
            )
            self.played_notes.append(pn)

# ------------------------------
# Fingering integration point
# ------------------------------
def apply_fingering_algorithm(play_notes:List[PlayedNote], fingering_fn:Callable[[List[PlayedNote]],List[PlayedNote]]):
    """
    fingering_fn : function that receives list of PlayedNote for ONE hand at a time,
                   returns same list with .finger set (1..5 integers).
    We'll group by hand and call fingering_fn independently (your algorithm requirement).
    """
    # group by hand
    by_hand = defaultdict(list)
    for pn in play_notes:
        by_hand[pn.hand].append(pn)
    for hand, notes in by_hand.items():
        # ensure sorted by onset then pitch
        notes_sorted = sorted(notes, key=lambda x:(x.onset, -x.pitch))
        # call user algorithm
        res = fingering_fn(notes_sorted)
        # copy finger values back to master list (match by id: onset+pitch)
        # we assume returned list items correspond 1:1 to input notes_sorted
        for in_note, out_note in zip(notes_sorted, res):
            in_note.finger = out_note.finger

# ------------------------------
# Inject fingerings into XML
# ------------------------------
def inject_fingerings_to_xml(tree:etree._ElementTree, played_notes:List[PlayedNote], output_path:str):
    """
    For each PlayedNote that has xml_element not None, write <notations><technical><fingering>value</fingering></technical></notations>
    If multiple PlayedNotes map to same xml element (e.g. chord members), we write the finger of the first occurrence.
    """
    written = set()
    for pn in played_notes:
        elem = pn.xml_element
        if elem is None or pn.finger is None:
            continue
        elem_id = id(elem)
        if elem_id in written:
            continue
        # find or create notations/technical/fingering
        notations = elem.find('notations')
        if notations is None:
            notations = etree.SubElement(elem, 'notations')
        technical = notations.find('technical')
        if technical is None:
            technical = etree.SubElement(notations, 'technical')
        fing = technical.find('fingering')
        if fing is None:
            fing = etree.SubElement(technical, 'fingering')
        fing.text = str(pn.finger)
        written.add(elem_id)
    # write file
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')

# ------------------------------
# Example dummy fingering algorithm (must be replaced by your actual algorithm)
# ------------------------------
def dummy_fingering_algorithm(notes:List[PlayedNote]) -> List[PlayedNote]:
    # naive: assign fingers cycling 1..5
    i=0
    for n in notes:
        n.finger = (i % 5) + 1
        i += 1
    return notes

# ------------------------------
# Runner / API
# ------------------------------
def process_file(input_path:str, output_path:str, fingering_fn:Callable[[List[PlayedNote]],List[PlayedNote]]):
    parser = PianistMusicXMLParser(input_path)
    played = parser.parse_and_expand()
    # Call fingering per hand
    apply_fingering_algorithm(played, fingering_fn)
    # inject into XML and write
    inject_fingerings_to_xml(parser.tree, played, output_path)
    return played

# ------------------------------
# CLI usage example
# ------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python musicxml_pianist_parser.py input.xml output.xml")
        sys.exit(1)
    inp = sys.argv[1]; outp = sys.argv[2]
    played = process_file(inp, outp, dummy_fingering_algorithm)
    print(f"Processed {len(played)} played notes. Output written to {outp}")
