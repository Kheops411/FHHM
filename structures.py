import numpy as np
from dataclasses import dataclass

# Global Constants for logical consistency
EPSILON_ONSET = 1e-3
HAND_RIGHT = 0
HAND_LEFT = 1
HAND_UNKNOWN = -1

@dataclass
class ScoreData:
    __slots__ = (
        'onset', 'offset', 'pitch', 'velocity', 
        'id', 'source_ref', 'event_id', 'measure', 'hand', 
        'finger_gt', 'finger_out'
    )
    
    # --- Time Data (Seconds, float64) ---
    onset: np.ndarray      
    offset: np.ndarray     
    
    # --- Musical Data ---
    pitch: np.ndarray      # int16 (MIDI 0-127)
    velocity: np.ndarray   # int8
    
    # --- Structure & Identifiers ---
    id: np.ndarray         # int64 (Unique sequential ID)
    source_ref: np.ndarray # int64 (Pointer to external source object)
    event_id: np.ndarray   # int64 (Grouping ID for chords)
    measure: np.ndarray    # int32
    hand: np.ndarray       # int8 (0=Right, 1=Left)
    
    # --- I/O for Algorithms ---
    finger_gt: np.ndarray  # int8 (Ground Truth from file)
    finger_out: np.ndarray # int8 (Output to be computed)

    def __len__(self):
        return len(self.onset)
        
    @property
    def size(self):
        return len(self.onset)

    @classmethod
    def allocate(cls, n_notes: int):
        """
        Allocate contiguous memory blocks for optimal Numba performance.
        Returns an initialized ScoreData instance.
        """
        return cls(
            onset=np.zeros(n_notes, dtype=np.float64),
            offset=np.zeros(n_notes, dtype=np.float64),
            pitch=np.zeros(n_notes, dtype=np.int16),
            velocity=np.zeros(n_notes, dtype=np.int8),
            id=np.arange(n_notes, dtype=np.int64),
            source_ref=np.zeros(n_notes, dtype=np.int64),
            event_id=np.zeros(n_notes, dtype=np.int64),
            measure=np.zeros(n_notes, dtype=np.int32),
            hand=np.full(n_notes, HAND_UNKNOWN, dtype=np.int8),
            finger_gt=np.zeros(n_notes, dtype=np.int8),
            finger_out=np.zeros(n_notes, dtype=np.int8),
        )

    def sort_canonical(self):
        """
        Sorts the data physically in memory: Onset (asc) -> Pitch (asc).
        Ensures arrays remain C-CONTIGUOUS for Numba vectorization.
        """
        # lexsort keys are: (Secondary, Primary) -> (Pitch, Onset)
        sorter = np.lexsort((self.pitch, self.onset))
        
        for attr in self.__slots__:
            arr = getattr(self, attr)
            # ascontiguousarray forces a memory copy to a new aligned buffer
            setattr(self, attr, np.ascontiguousarray(arr[sorter]))