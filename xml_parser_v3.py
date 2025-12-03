# xml_parser.py
import numpy as np
from lxml import etree
from collections import defaultdict
from enum import IntEnum
import sys

# ============================================================================
# 1. Core Data Structure (ScoreData)
# ============================================================================

# Tolérance temporelle
EPSILON_ONSET = 1e-3

class Hand(IntEnum):
    RIGHT = 0
    LEFT = 1
    UNKNOWN = -1

class ScoreData:
    """
    Conteneur SoA (Structure of Arrays) unifié pour partition.
    Optimisé pour NumPy/Numba et les algorithmes de ML.
    """
    __slots__ = (
        'pitch', 'onset', 'offset', 'duration', 'velocity',
        'hand', 'source_id', 'event_id', 'finger_gt', 'finger_out',
        '_size'
    )

    def __init__(self, n_notes: int):
        # Données musicales
        self.pitch = np.zeros(n_notes, dtype=np.int16)      # MIDI 0-127
        self.onset = np.zeros(n_notes, dtype=np.float64)    # Secondes
        self.offset = np.zeros(n_notes, dtype=np.float64)   # Secondes
        self.duration = np.zeros(n_notes, dtype=np.float64) # Secondes
        self.velocity = np.zeros(n_notes, dtype=np.uint8)   # 0-127
        
        # Métadonnées
        self.hand = np.full(n_notes, Hand.UNKNOWN, dtype=np.int8)
        self.source_id = np.zeros(n_notes, dtype=np.int64)  # ID unique (ex: pointeur mémoire node XML)
        self.event_id = np.arange(n_notes, dtype=np.int64)  # ID séquentiel unique
        
        # Doigtés (0 = unset, 1-5 = doigt)
        self.finger_gt = np.zeros(n_notes, dtype=np.int8)   # Ground Truth (XML)
        self.finger_out = np.zeros(n_notes, dtype=np.int8)  # Résultat algo

        self._size = n_notes

    @property
    def size(self) -> int:
        return self._size

    def compute_duration(self):
        self.duration = self.offset - self.onset

    def validate(self):
        if np.any(self.duration < -EPSILON_ONSET):
            raise ValueError("Erreur critique : Durées négatives détectées.")
    
    def get_hand_view(self, hand_side: int) -> dict:
        """
        Retourne des vues (pas de copie) filtrées par main et triées temporellement.
        Utilisable directement par les algos.
        """
        mask = (self.hand == hand_side)
        indices = np.where(mask)[0]
        
        # Tri par onset puis pitch
        sorted_idx = indices[np.lexsort((self.pitch[indices], self.onset[indices]))]
        
        return {
            'indices': sorted_idx, # Pour mapper le retour
            'pitch': self.pitch[sorted_idx],
            'onset': self.onset[sorted_idx],
            'duration': self.duration[sorted_idx],
            'velocity': self.velocity[sorted_idx],
            'finger_gt': self.finger_gt[sorted_idx],
            'finger_out': self.finger_out[sorted_idx] # Vue inscriptible
        }

# ============================================================================
# 2. Parsing Constants & Utils
# ============================================================================

NATURAL_OFFSETS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
SHARP_ORDER = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
FLAT_ORDER = ['B', 'E', 'A', 'D', 'G', 'C', 'F']
NOTE_INDEX = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}
INDEX_NOTE = {v: k for k, v in NOTE_INDEX.items()}

DEFAULT_VELOCITY = 80
DEFAULT_DIVISIONS = 480
DEFAULT_TEMPO = 120.0

SUBDIV_MAP = {
    'whole': 4.0, 'half': 2.0, 'quarter': 1.0, 'eighth': 0.5,
    '16th': 0.25, '32nd': 0.125,
}

ARTICULATION_RATIOS = {
    'staccatissimo': 0.25, 'staccato': 0.5, 'portato': 0.75,
    'tenuto': 0.99, 'marcato': 0.95, 'accent': 0.95, 'sforzando': 0.95,
}

VELOCITY_MODIFIERS = {
    'marcato': 1.2, 'accent': 1.2, 'sforzando': 1.5,
}

ORNAMENT_MOTIFS = {
    'Trill': ([], [0, +1], []),
    'TrillBaroque': ([], [+1, 0], [-1, 0]),
    'Mordent': ([0], [-1, 0], []),
    'UpperMordent': ([0, +1], [], []),
    'Turn': ([+1, 0, -1], [0], []),
    'InvertedTurn': ([-1, 0, +1], [0], []),
    'PrallMordent': ([], [+1, 0, -1, 0], []),
}

GLISSANDO_MIN_NOTE_DURATION = 0.02
ARPEGGIO_MAX_OFFSET_MS = 60.0
ARPEGGIO_DEFAULT_STRETCH = 1.0
MIN_NOTE_DURATION = 0.0001

def key_fifths_to_alter_map(fifths: int):
    alter_map = {s: 0 for s in NATURAL_OFFSETS}
    if fifths > 0:
        for i in range(min(7, fifths)): alter_map[SHARP_ORDER[i]] = 1
    elif fifths < 0:
        for i in range(min(7, -fifths)): alter_map[FLAT_ORDER[i]] = -1
    return alter_map

def get_midi_pitch(step, octave, accidentals, keysig):
    if step is None or octave is None: return None
    key = (step, octave)
    alter = accidentals.get(key, keysig.get(step, 0))
    return (octave + 1) * 12 + NATURAL_OFFSETS[step] + alter

def get_diatonic_neighbor(step, octave, offset, accidentals, keysig):
    if step is None or octave is None: return None
    idx = NOTE_INDEX[step] + offset
    target_octave = octave + (idx // 7)
    target_step = INDEX_NOTE[idx % 7]
    return get_midi_pitch(target_step, target_octave, accidentals, keysig)

def ticks_to_quarters(ticks, divisions): return ticks / max(1, divisions)
def quarters_to_seconds(quarters, tempo): return quarters * (60.0 / max(tempo, 1.0))
def bpm_to_bps(bpm): return max(bpm, 1.0) / 60.0
def safe_int(text, default=0): 
    try: return int(text) if text is not None else default
    except: return default
def safe_float(text, default=0.0):
    try: return float(text) if text is not None else default
    except: return default

# ============================================================================
# 3. Structural Navigation
# ============================================================================

class StructuralNavigator:
    def __init__(self, part):
        self.part = part
        self.measures = part.findall('./measure')
        self.n = len(self.measures)
        self.repeat_start = [False] * self.n
        self.repeat_stop = [None] * self.n
        self.endings = [None] * self.n
        self.sound_directives = [None] * self.n
        self.segno_idx = None
        self.coda_idx = None
        self._index_structure()

    def _index_structure(self):
        for idx, measure in enumerate(self.measures):
            for barline in measure.findall('barline'):
                repeat = barline.find('repeat')
                if repeat is not None:
                    d = repeat.get('direction')
                    if d == 'forward': self.repeat_start[idx] = True
                    elif d == 'backward': self.repeat_stop[idx] = safe_int(repeat.get('times'), 1)
            
            for ending in measure.findall('ending'):
                num = ending.get('number')
                if num: self.endings[idx] = [safe_int(n) for n in num.split(',') if safe_int(n) > 0]
            
            for direction in measure.findall('.//direction'):
                dt = direction.find('direction-type')
                if dt is not None:
                    if dt.find('segno') is not None: self.segno_idx = idx
                    if dt.find('coda') is not None: self.coda_idx = idx
                sound = direction.find('sound')
                if sound is not None and sound.attrib: self.sound_directives[idx] = dict(sound.attrib)

    def build_play_sequence(self, max_iter=10000):
        seq = []
        idx = 0
        volta_passes = defaultdict(int)
        iterations = 0
        
        def find_start(before):
            for i in range(before, -1, -1):
                if self.repeat_start[i]: return i
            return 0
            
        while idx < self.n and iterations < max_iter:
            iterations += 1
            start = find_start(idx)
            current_pass = volta_passes.get((start, idx), 0) + 1
            
            if self.endings[idx] is None or current_pass in self.endings[idx]:
                seq.append(idx)
            
            sound = self.sound_directives[idx]
            if sound:
                if sound.get('dacapo') in ('yes','1'): idx = 0; continue
                if sound.get('dalsegno') in ('yes','1') and self.segno_idx: idx = self.segno_idx; continue
                if sound.get('tocoda') in ('yes','1') and self.coda_idx: idx = self.coda_idx; continue
                if sound.get('fine') in ('yes','1'): break

            times = self.repeat_stop[idx]
            if times:
                key = (start, idx)
                volta_passes[key] += 1
                if volta_passes[key] <= times:
                    idx = start
                    continue
            idx += 1
        return seq

# ============================================================================
# 4. Main Parser
# ============================================================================

class PartState:
    def __init__(self):
        self.divisions = DEFAULT_DIVISIONS
        self.keysig_map = {s: 0 for s in NATURAL_OFFSETS}
        self.tempo = DEFAULT_TEMPO
        self.accidentals = {}

class MusicXMLParser:
    def __init__(self, xml_path):
        self.tree = etree.parse(xml_path)
        self.root = self.tree.getroot()
        # Mapping ID mémoire -> Element XML pour l'injection
        self.xml_map = {} 
        # Liste temporaire pour accumuler les données avant conversion en ScoreData
        self._raw_notes = [] 
        self.part_states = {}
        self.buffers = {}
        self._chord_start_ticks = 0

    def parse(self) -> ScoreData:
        parts = self.root.findall('.//part')
        for part in parts:
            part_id = id(part)
            self.part_states[part_id] = PartState()
            nav = StructuralNavigator(part)
            seq = nav.build_play_sequence()
            measures = part.findall('./measure')
            cursor = 0.0
            
            for m_idx in seq:
                if not (0 <= m_idx < len(measures)): continue
                measure = measures[m_idx]
                state = self.part_states[part_id]
                self._process_attributes(measure, state)
                cursor += self._process_measure(measure, part_id, state, cursor)
        
        self._resolve_ties()
        
        # --- Conversion finale vers ScoreData (Bulk Load) ---
        n = len(self._raw_notes)
        score = ScoreData(n)
        
        # Tri initial pour assurer la cohérence temporelle
        # Tri: Hand -> Onset -> Pitch
        self._raw_notes.sort(key=lambda x: (x['hand'], x['onset'], x['pitch'] or -1))
        
        for i, note in enumerate(self._raw_notes):
            pitch = note.get('pitch')
            score.pitch[i] = pitch if pitch is not None else 0
            score.onset[i] = note['onset']
            score.offset[i] = note['offset']
            score.velocity[i] = note['velocity']
            score.hand[i] = note['hand']
            score.finger_gt[i] = note.get('finger_gt', 0) or 0
            
            # Gestion du lien vers le XML
            xml_elem = note.get('xml_element')
            if xml_elem is not None:
                eid = id(xml_elem)
                score.source_id[i] = eid
                self.xml_map[eid] = xml_elem # Sauvegarde pour l'injection
            else:
                score.source_id[i] = 0
                
        score.compute_duration()
        score.validate()
        return score

    # --- Internal Parsing Logic (identique à l'original mais remplit _raw_notes) ---

    def _process_attributes(self, measure, state):
        attr = measure.find('attributes')
        if attr is None: return
        div = attr.findtext('divisions')
        if div: state.divisions = safe_int(div, DEFAULT_DIVISIONS)
        key = attr.find('key')
        if key is not None: state.keysig_map = key_fifths_to_alter_map(safe_int(key.findtext('fifths')))

    def _process_measure(self, measure, part_id, state, measure_onset):
        local_ticks = 0
        state.accidentals.clear()
        self.buffers[('chord', part_id)] = []
        self.buffers[('grace', part_id)] = []
        self.buffers[('tremolo', part_id)] = None
        self.buffers[('glissando', part_id)] = None
        measure_notes = []

        for child in measure:
            tag = etree.QName(child.tag).localname
            if tag == 'backup': local_ticks = max(0, local_ticks - safe_int(child.findtext('duration')))
            elif tag == 'forward': local_ticks += safe_int(child.findtext('duration'))
            elif tag == 'direction': self._process_direction(child, state)
            elif tag == 'note':
                local_ticks = self._process_note(child, part_id, state, measure_onset, local_ticks, measure, measure_notes)
        
        self._resolve_glissandos(part_id, measure_notes)
        self._flush_chord(part_id, state)
        return ticks_to_quarters(local_ticks, state.divisions)

    def _process_direction(self, d, state):
        sound = d.find('sound')
        if sound is not None and 'tempo' in sound.attrib: state.tempo = safe_float(sound.get('tempo'), DEFAULT_TEMPO)
        metro = d.find('.//metronome/per-minute')
        if metro is not None: state.tempo = safe_float(metro.text, DEFAULT_TEMPO)

    def _process_note(self, note, part_id, state, measure_onset, local_ticks, measure, measure_notes):
        is_chord = note.find('chord') is not None
        if note.find('grace') is not None:
            self.buffers[('grace', part_id)].append(self._capture_grace(note, state, measure))
            dur = safe_int(note.findtext('duration'))
            return local_ticks + dur if dur and not is_chord else local_ticks
        
        if not is_chord: self._chord_start_ticks = local_ticks
        
        pitch_node = note.find('pitch')
        midi = None
        step = octave = None
        if pitch_node is not None:
            step = pitch_node.findtext('step')
            octave = safe_int(pitch_node.findtext('octave'))
            if pitch_node.find('alter') is not None: state.accidentals[(step, octave)] = safe_int(pitch_node.find('alter').text)
            acc = note.find('accidental')
            if acc is not None:
                val = {'sharp':1,'flat':-1,'natural':0,'double-sharp':2,'double-flat':-2}.get(acc.text.strip().lower())
                if val is not None: state.accidentals[(step, octave)] = val
            midi = get_midi_pitch(step, octave, state.accidentals, state.keysig_map)

        dur_ticks = safe_int(note.findtext('duration'))
        t_mod = note.find('time-modification')
        factor = (safe_int(t_mod.findtext('normal-notes'),1)/safe_int(t_mod.findtext('actual-notes'),1)) if t_mod is not None else 1.0
        
        onset = measure_onset + ticks_to_quarters(self._chord_start_ticks if is_chord else local_ticks, state.divisions)
        nom_q = ticks_to_quarters(dur_ticks, state.divisions) * factor
        
        # --- CORRECTION DES BUGS ---
        notations = note.find('notations')
        
        # Bug 1 Fix: Vérification explicite des noeuds avant itération
        orn = []
        arts = []
        if notations is not None:
            orn_node = notations.find('ornaments')
            if orn_node is not None:
                orn = [etree.QName(x).localname for x in orn_node]
            
            art_node = notations.find('articulations')
            if art_node is not None:
                arts = [etree.QName(x).localname for x in art_node]
        
        trem_node = notations.find('tremolo') if notations is not None else None
        trem_type = trem_node.get('type','single') if trem_node is not None else None
        
        # Bug 2 Fix: Sécurisation de la recherche de glissando/slide
        gliss_node = None
        if notations is not None:
            gliss_node = notations.find('glissando')
            if gliss_node is None:
                gliss_node = notations.find('slide')

        info = {
            'xml': note, 'midi': midi, 'step': step, 'octave': octave,
            'onset_q': onset, 'nominal_q': nom_q, 'staff': safe_int(note.findtext('staff'),1),
            'voice': note.findtext('voice','1'), 'ornaments': orn, 'articulations': arts,
            'tempo': state.tempo, 'accs': dict(state.accidentals), 'keys': dict(state.keysig_map),
            'velocity': self._calc_vel(arts), 'art_ratio': self._get_art_ratio(arts),
            'finger_gt': self._get_xml_finger(note),
            'is_trem_start': trem_type=='start', 'is_trem_stop': trem_type=='stop',
            'is_trem_single': trem_type=='single' or (trem_node is not None and not trem_type),
            'trem_beams': int(trem_node.text) if trem_node is not None and trem_node.text and trem_node.text.isdigit() else 2,
            'is_gliss_start': gliss_node is not None and gliss_node.get('type')=='start',
            'is_gliss_stop': gliss_node is not None and gliss_node.get('type')=='stop',
            'arp': notations.find('arpeggiate') is not None if notations is not None else False
        }
        
        measure_notes.append(info)
        
        if is_chord: self.buffers[('chord', part_id)].append(info)
        elif info['is_trem_start']: self.buffers[('tremolo', part_id)] = info
        elif info['is_trem_stop']:
            start = self.buffers[('tremolo', part_id)]
            if start: self._expand_trem_pair(start, info); self.buffers[('tremolo', part_id)] = None
            else: self._proc_note(info, part_id)
        elif info['is_gliss_start']: self.buffers[('glissando', part_id)] = info
        else: self._proc_note(info, part_id)
            
        return local_ticks + dur_ticks if not is_chord else local_ticks
    
    def _proc_note(self, info, part_id):
            # Flush grace
            k = ('grace', part_id)
            if self.buffers[k]:
                graces = self.buffers[k]; self.buffers[k] = []
                princ_on = info['onset_q']
                shift = 0
                for g in graces:
                    d = 0.0
                    if g['type'] == 'Acciaccatura': d = min(SUBDIV_MAP['32nd']/2, (0.5*info['nominal_q'])/len(graces))
                    else: d = 0.5 * info['nominal_q'] / len(graces)
                    
                    gm = get_midi_pitch(g['step'],g['oct'],g['accs'],g['keys'])
                    # Correction Tempo ici aussi au cas où
                    self._add_raw(g['xml'], gm, princ_on+shift, d, g['staff'], g['vel'], None, info['tempo'])
                    shift += d
                info['onset_q'] += shift
                info['nominal_q'] = max(MIN_NOTE_DURATION, info['nominal_q'] - shift)

            if info['is_trem_single']: self._expand_trem(info)
            elif info['ornaments']: self._expand_orn(info)
            else: 
                # --- CORRECTION CRITIQUE ICI ---
                # Ajout de l'argument info['tempo'] à la fin
                self._add_raw(
                    info['xml'], 
                    info['midi'], 
                    info['onset_q'], 
                    info['nominal_q']*info['art_ratio'], 
                    info['staff'], 
                    info['velocity'], 
                    info['finger_gt'],
                    info['tempo']  # <--- C'était l'oubli !
                )
    def _add_raw(self, xml, pitch, onset_q, dur_q, staff, vel, finger, tempo=DEFAULT_TEMPO):
        # Conversion finale en secondes et ajout à la liste brute
        sec_on = quarters_to_seconds(onset_q, tempo)
        sec_dur = quarters_to_seconds(max(MIN_NOTE_DURATION, dur_q), tempo)
        self._raw_notes.append({
            'pitch': pitch,
            'onset': sec_on,
            'offset': sec_on + sec_dur,
            'velocity': vel,
            'hand': Hand.LEFT if staff == 2 else Hand.RIGHT,
            'xml_element': xml,
            'finger_gt': finger
        })

    def _capture_grace(self, n, s, m):
        pn = n.find('pitch')
        return {
            'xml': n, 'step': pn.findtext('step') if pn is not None else None,
            'oct': safe_int(pn.findtext('octave')) if pn is not None else None,
            'accs': dict(s.accidentals), 'keys': dict(s.keysig_map),
            'staff': safe_int(n.findtext('staff'), 1), 'vel': DEFAULT_VELOCITY,
            'type': 'Acciaccatura' if n.find('grace').get('slash')=='yes' else 'Appoggiatura'
        }

    def _calc_vel(self, arts):
        v = DEFAULT_VELOCITY
        for a in arts: v = int(v * VELOCITY_MODIFIERS.get(a, 1.0))
        return min(127, v)
        
    def _get_art_ratio(self, arts):
        for a in arts:
            if a in ARTICULATION_RATIOS: return ARTICULATION_RATIOS[a]
        return 1.0

    def _get_xml_finger(self, note):
        try:
            t = note.find('.//fingering')
            if t is not None and t.text:
                return int(t.text.replace('(','').replace(')','').split('-')[-1].split()[0])
        except: pass
        return 0

    def _expand_arpeggio_group(self, members, state):
        """
        Version V2 de l'expansion d'arpège.
        Répartit les notes de l'accord dans le temps (strumming).
        """
        if not members: return
        
        N = len(members)
        nominal = max(m['nominal_q'] for m in members)
        
        # Tri par hauteur MIDI
        # Note : members est une liste de dicts (info)
        member_list = sorted(members, key=lambda x: x['midi'] or 0)
        
        # Détection direction
        is_descending = False
        for m in members:
            # Recherche de la direction dans le XML stocké
            arp = m['xml'].find('.//arpeggiate')
            if arp is not None and 'down' in arp.get('direction', '').lower():
                is_descending = True
                break
        
        if is_descending:
            member_list.reverse()
        
        # Calcul timing (similaire V1)
        # ARPEGGIO_MAX_OFFSET_MS = 60.0
        # ARPEGGIO_DEFAULT_STRETCH = 1.0
        bpm = state.tempo
        max_off_q = (ARPEGGIO_MAX_OFFSET_MS / 1000.0) * (bpm / 60.0)
        
        # Pas temporel
        raw_step = nominal / N if N > 0 else 0.0
        offset_step = min(raw_step, max_off_q)
        
        base_onset = members[0]['onset_q']
        
        for i, info in enumerate(member_list):
            delay = offset_step * i * ARPEGGIO_DEFAULT_STRETCH
            real_onset = base_onset + delay
            real_dur = max(MIN_NOTE_DURATION, nominal - delay)
            
            # Ajout via _add_raw
            self._add_raw(
                info['xml'], 
                info['midi'], 
                real_onset, 
                real_dur, 
                info['staff'], 
                info['velocity'], 
                info['finger_gt'], 
                state.tempo
            )
    def _flush_chord(self, pid, state):
        """
        Gestion complète des accords et arpèges (étalement temporel).
        """
        k = ('chord', pid)
        if k not in self.buffers or not self.buffers[k]: return
        
        chord_notes = self.buffers[k]
        self.buffers[k] = []
        
        # 1. Grouper par onset (cas où plusieurs voix ont des accords au même moment)
        groups_by_onset = defaultdict(list)
        for n in chord_notes: groups_by_onset[n['onset_q']].append(n)
        
        for onset, notes in groups_by_onset.items():
            # Séparer arpégés / non-arpégés
            arp_notes = [n for n in notes if n['arp']]
            reg_notes = [n for n in notes if not n['arp']]
            
            # Traitement des arpèges (groupés par numéro d'arpège XML si présent)
            if arp_notes:
                # Sous-groupes par attribut 'number' de <arpeggiate>
                sub_groups = defaultdict(list)
                for n in arp_notes:
                    arp_tag = n['xml'].find('.//notations/arpeggiate')
                    num = arp_tag.get('number') if arp_tag is not None else 'default'
                    sub_groups[num].append(n)
                
                for grp in sub_groups.values():
                    self._expand_arpeggio_group(grp, state)
            
            # Traitement des accords normaux (plaqués)
            if reg_notes:
                base_onset = reg_notes[0]['onset_q']
                # On s'assure que la durée est celle de la note la plus longue pour la tenue visuelle, 
                # ou on respecte la durée individuelle (MusicXML permet des durées différentes dans un accord)
                for n in reg_notes:
                    self._add_raw(n['xml'], n['midi'], base_onset, n['nominal_q']*n['art_ratio'], 
                                n['staff'], n['velocity'], n['finger_gt'], state.tempo)
    # ---------- nouvelle implémentation _expand_orn ----------
    def _expand_orn(self, i):
        """
        Expansion des ornements avec subdivisions proportionnelles.
        i : info dict (comme précédemment)
        """
        # Map XML tag -> motif key
        orn_map = {
            'trill-mark': 'Trill', 'trill': 'Trill', 'mordent': 'Mordent',
            'inverted-mordent': 'UpperMordent', 'turn': 'Turn', 'inverted-turn': 'InvertedTurn',
        }

        motif_key = next((orn_map.get(o) for o in i['ornaments'] if orn_map.get(o) in ORNAMENT_MOTIFS), None)

        if not motif_key:
            # fallback : note simple
            self._add_raw(i['xml'], i['midi'], i['onset_q'], i['nominal_q'] * i['art_ratio'], i['staff'], i['velocity'], i['finger_gt'], i['tempo'])
            return

        prefix, body, suffix = ORNAMENT_MOTIFS[motif_key]

        # Choix de paramétrage musical par type
        tempo = max(i.get('tempo', DEFAULT_TEMPO), 1.0)
        if motif_key == 'Trill':
            frac = 0.85            # 85% de la note pour le trille
            notes_per_sec = 8     # cible ≈ 8 micro-événements / sec (ajustable)
            N_min, N_max = 3, 30  # cycles min/max (un cycle = len(body) events)
        elif motif_key in ('Mordent', 'UpperMordent', 'PrallMordent'):
            frac = 0.3
            notes_per_sec = 10
            N_min, N_max = 1, 8
        elif motif_key in ('Turn', 'InvertedTurn'):
            frac = 0.4
            notes_per_sec = 9
            N_min, N_max = 1, 8
        else:
            frac = 0.5
            notes_per_sec = 8
            N_min, N_max = 1, 12

        # Longueurs des parties
        prefix_len = len(prefix)
        body_len = len(body)
        suffix_len = len(suffix)

        # Calcul subdivision en quarters
        t_sub_q, total_events, cycles = self._ornament_subdivide(i['nominal_q'], tempo, frac, N_min, N_max, notes_per_sec, body_len, prefix_len, suffix_len)

        # Générer séquence d'événements (pitch, dur)
        def get_neighbor(offset):
            return get_diatonic_neighbor(i['step'], i['octave'], offset, i['accs'], i['keys'])

        seq_events = []
        # prefix
        for off in prefix:
            seq_events.append((get_neighbor(off), t_sub_q))
        # body répété cycles fois
        for _ in range(cycles):
            for off in body:
                seq_events.append((get_neighbor(off), t_sub_q))
        # suffix
        for off in suffix:
            seq_events.append((get_neighbor(off), t_sub_q))

        # Si pour une raison quelconque seq_events ne remplit pas la fraction (arrondi), ajuster :
        used_total_q = sum(d for (_, d) in seq_events) if seq_events else 0.0
        frac_total_q = frac * i['nominal_q']
        # Si on a moins, raccourcir légèrement chaque event pour coller ; si on a plus, tronquer la séquence
        if used_total_q > frac_total_q + 1e-12:
            # trop long → supprimer événements excédentaires (à la fin)
            while seq_events and sum(d for (_, d) in seq_events) - seq_events[-1][1] >= frac_total_q - 1e-12:
                seq_events.pop()
        elif used_total_q < frac_total_q - 1e-12 and seq_events:
            # redistribuer proportionnellement (rare)
            factor = frac_total_q / used_total_q
            seq_events = [(m, d * factor) for (m, d) in seq_events]

        # Ajouter les événements de l'ornement
        onset = i['onset_q']
        used_total = 0.0
        for midi, dur_q in seq_events:
            if midi is None:
                onset += dur_q
                used_total += dur_q
                continue
            # GT finger = 0 pour ornements (indique micro-note)
            self._add_raw(i['xml'], midi, onset, max(MIN_NOTE_DURATION, dur_q), i['staff'], i['velocity'], 0, i['tempo'])
            onset += dur_q
            used_total += dur_q

        # Note principale restante (sustain)
        remaining_main = max(MIN_NOTE_DURATION, i['nominal_q'] - used_total)
        # Appliquer art_ratio à la partie principale
        self._add_raw(i['xml'], i['midi'], onset, remaining_main * i['art_ratio'], i['staff'], i['velocity'], i['finger_gt'], i['tempo'])

    # ---------- nouvelle implémentation _expand_trem ----------
    def _expand_trem(self, i):
        """
        Tremolo d'une seule note (répétitions d'une même hauteur).
        On répartit la durée en N événements calculés selon tempo et une vitesse max raisonnable.
        """
        tempo = max(i.get('tempo', DEFAULT_TEMPO), 1.0)
        nominal_q = max(MIN_NOTE_DURATION, i['nominal_q'])
        # Paramètres
        frac = 1.0               # utiliser la totalité de la note pour le tremolo
        notes_per_sec = 12       # cible max micro-notes par seconde
        N_min, N_max = 4, 80     # bornes

        # Estimer nombre total d'événements sur la durée nominale
        t_sub_q, total_events, _ = self._ornament_subdivide(nominal_q, tempo, frac, N_min, N_max, notes_per_sec, body_len=1, prefix_len=0, suffix_len=0)

        # Répartir total_events sur la durée nominale
        onset = i['onset_q']
        remaining = nominal_q
        for k in range(total_events):
            dur_q = min(t_sub_q, remaining)
            self._add_raw(i['xml'], i['midi'], onset, dur_q, i['staff'], i['velocity'], i['finger_gt'], i['tempo'])
            onset += dur_q
            remaining -= dur_q
            if remaining <= 0:
                break

    # ---------- nouvelle implémentation _expand_trem_pair ----------
    def _expand_trem_pair(self, a, b):
        """
        Tremolo alterné entre a et b. On prend la durée de a comme référence (musical).
        """
        tempo = max(a.get('tempo', DEFAULT_TEMPO), 1.0)
        nominal_q = max(MIN_NOTE_DURATION, a['nominal_q'])  # durée de référence = a
        # Paramètres raisonnables
        frac = 1.0
        notes_per_sec = 12
        N_min, N_max = 4, 80

        # Si b a une nominale plus courte, on ignore pour la durée du tremolo (convention)
        t_sub_q, total_events, _ = self._ornament_subdivide(nominal_q, tempo, frac, N_min, N_max, notes_per_sec, body_len=1, prefix_len=0, suffix_len=0)

        on = a['onset_q']
        remaining = nominal_q
        # Alterne entre a et b
        for k in range(total_events):
            src = a if (k % 2 == 0) else b
            dur_q = min(t_sub_q, remaining)
            self._add_raw(a['xml'], src['midi'], on, dur_q, a['staff'], a['velocity'], a['finger_gt'], a['tempo'])
            on += dur_q
            remaining -= dur_q
            if remaining <= 0:
                break
    def _resolve_glissandos(self, pid, notes):
        start = self.buffers[('glissando', pid)]
        if not start: return
        self.buffers[('glissando', pid)] = None
        # Find target
        tgt = None
        try:
            idx = notes.index(start)
            for k in range(idx+1, len(notes)):
                if notes[k]['staff'] == start['staff'] and notes[k]['is_gliss_stop']:
                    tgt = notes[k]; break
        except ValueError: pass
        
        if tgt and start['midi'] and tgt['midi']:
            p1, p2 = start['midi'], tgt['midi']
            r = range(p1, p2+1) if p1 < p2 else range(p1, p2-1, -1)
            dur = start['nominal_q'] / len(r)
            on = start['onset_q']
            for p in r:
                self._add_raw(start['xml'], p, on, dur, start['staff'], start['velocity'], start['finger_gt'], start['tempo'])
                on += dur
        else:
            self._proc_note(start, pid)
    # ---------- helper pour subdivisions musicales ----------
    def _ornament_subdivide(self, nominal_q, tempo, frac, N_min, N_max, notes_per_sec=10, body_len=1, prefix_len=0, suffix_len=0):
        """
        Retourne (t_sub_q, total_events, cycles)
        - nominal_q : durée de la note en noires (quarters)
        - tempo : bpm
        - frac : fraction de la nominal_q à réserver pour l'ornement (0..1)
        - N_min, N_max : bornes pour le nombre de cycles (si body_len>0) ou pour le nombre minimal d'événements
        - notes_per_sec : cible d'événements par seconde (pour estimer cycles)
        - body_len : nombre d'événements par cycle (ex: len(body))
        - prefix_len/suffix_len : nombre d'événements fixes de pré/suffixe
        """
        # Durée nominale en secondes
        nominal_secs = quarters_to_seconds(max(MIN_NOTE_DURATION, nominal_q), tempo)
        # Estimer cycles souhaités (nombre d'occurrences du motif principal)
        if body_len > 0:
            est_cycles = max(1, int(round((nominal_secs * notes_per_sec) / max(1, body_len))))
            cycles = max(N_min, min(N_max, est_cycles))
            total_events = prefix_len + suffix_len + cycles * body_len
        else:
            # Si pas de body (ex: simple prefix/suffix), on répartit les events entre prefix/suffix
            total_events = max(1, prefix_len + suffix_len)
            cycles = 0

        # Eviter division par zéro
        total_events = max(1, total_events)
        # Durée par micro-événement (en quarters)
        t_sub_q = (frac * nominal_q) / total_events
        return t_sub_q, total_events, cycles

    def _resolve_ties(self):
        """
        Fusionne les notes liées (tied) en une seule note longue.
        Version robuste basée sur source_id et continuité temporelle.
        """
        # On ne travaille pas sur ScoreData ici mais sur _raw_notes avant conversion
        if not self._raw_notes: return

        # 1. Grouper par (Voice, Staff, Pitch) pour suivre la monophonie par voix
        # Utilisation de listes pour préserver l'ordre d'insertion (onset croissant)
        streams = defaultdict(list)
        for i, n in enumerate(self._raw_notes):
            # Clé unique de flux : (hand/staff, voice_str, pitch)
            # hand est dérivé de staff dans _add_raw
            k = (n['hand'], n['xml_element'].findtext('voice','1'), n['pitch'])
            streams[k].append(i) # On stocke l'index dans _raw_notes

        indices_to_remove = set()

        for k, indices in streams.items():
            # indices est déjà trié par onset car _raw_notes est appendé séquentiellement
            # mais on refait un tri par sécurité sur l'onset
            indices.sort(key=lambda idx: self._raw_notes[idx]['onset'])
            
            i = 0
            while i < len(indices):
                curr_idx = indices[i]
                curr_note = self._raw_notes[curr_idx]
                xml = curr_note['xml_element']
                
                # Vérifier si c'est un début de liaison
                ties = [t.get('type') for t in (xml.findall('tie') if xml is not None else [])]
                # Fallback notations/tied
                if xml is not None and xml.find('notations') is not None:
                    ties.extend([t.get('type') for t in xml.find('notations').findall('tied')])
                
                if 'start' not in ties:
                    i += 1
                    continue
                
                # C'est un début, on cherche la suite
                total_dur_sec = curr_note['offset'] - curr_note['onset']
                
                j = i + 1
                while j < len(indices):
                    next_idx = indices[j]
                    next_note = self._raw_notes[next_idx]
                    
                    # Vérification contiguïté temporelle (avec epsilon)
                    expected_onset = curr_note['onset'] + total_dur_sec
                    if abs(next_note['onset'] - expected_onset) > 0.05: # 50ms tolérance trou
                        break # Rupture de chaîne
                    
                    # Marquer pour suppression
                    indices_to_remove.add(next_idx)
                    
                    # Accumuler durée
                    seg_dur = next_note['offset'] - next_note['onset']
                    total_dur_sec += seg_dur
                    
                    # Vérifier si fin de liaison
                    next_xml = next_note['xml_element']
                    next_ties = [t.get('type') for t in (next_xml.findall('tie') if next_xml is not None else [])]
                    if next_xml is not None and next_xml.find('notations') is not None:
                        next_ties.extend([t.get('type') for t in next_xml.find('notations').findall('tied')])
                    
                    if 'stop' in next_ties and 'start' not in next_ties:
                        j += 1 # On a consommé la fin
                        break
                    
                    j += 1
                
                # Appliquer la fusion
                curr_note['offset'] = curr_note['onset'] + total_dur_sec
                # On saute les notes qu'on vient de fusionner
                i = j

        # Reconstruction de la liste épurée
        self._raw_notes = [n for idx, n in enumerate(self._raw_notes) if idx not in indices_to_remove]

# ============================================================================
# 5. Injection Logic
# ============================================================================

def inject_fingerings(
    parser_instance: MusicXMLParser,
    score_data: ScoreData,
    output_path: str
):
    """
    Réinjecte les doigtés calculés (score_data.finger_out) dans le XML
    en utilisant source_id pour retrouver les éléments.
    """
    processed_ids = set()
    
    # Itération vectorisée optimisée ? Non, on doit écrire dans le DOM
    for i in range(score_data.size):
        f = score_data.finger_out[i]
        sid = score_data.source_id[i]
        
        if f == 0 or sid == 0:
            continue
            
        if sid in processed_ids:
            continue
            
        xml_elem = parser_instance.xml_map.get(sid)
        if xml_elem is None:
            continue
            
        # Structure XML
        notations = xml_elem.find('notations')
        if notations is None: notations = etree.SubElement(xml_elem, 'notations')
        technical = notations.find('technical')
        if technical is None: technical = etree.SubElement(notations, 'technical')
        fingering = technical.find('fingering')
        if fingering is None: fingering = etree.SubElement(technical, 'fingering')
        
        fingering.text = str(f)
        fingering.set('placement', 'above' if score_data.hand[i] == Hand.RIGHT else 'below')
        
        processed_ids.add(sid)

    parser_instance.tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')

# ============================================================================
# Entry Point Example
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python xml_parser.py <input> <output>")
        sys.exit(1)

    print("1. Parsing vers ScoreData...")
    parser = MusicXMLParser(sys.argv[1])
    score = parser.parse()
    
    print(f"   Structure chargée: {score.size} notes.")
    print(f"   Mains: {np.sum(score.hand == Hand.RIGHT)} Droite, {np.sum(score.hand == Hand.LEFT)} Gauche.")
    
    # Simulation d'un algo qui remplit finger_out
    print("2. Application doigté dummy...")
    score.finger_out[:] = (np.arange(score.size) % 5) + 1
    
    print("3. Injection XML...")
    inject_fingerings(parser, score, sys.argv[2])
    print("   Terminé.")