# HMM Model Debugging Findings

This document tracks the process of debugging the Python HMM implementation to align it with the C++ reference.

## HMM2 Mismatch Analysis

### Initial Finding
An initial comparison between the C++ and Python HMM2 outputs for `scores/001-1_fingering.txt` revealed a mismatch at `original_idx=467`.

- **C++ Output:** Finger `4`
- **Python Output:** Finger `5`

### Root Cause Identification
To diagnose this, I created dumper scripts for both the C++ and Python implementations to inspect the exact sequence of notes being fed into the Viterbi algorithm. The comparison revealed a major discrepancy.

- The C++ reference binary (`FingeringHMM2_Run`) processes notes in the exact order they appear in the source `.txt` file.
- The Python script, however, was incorrectly applying the `apply_time_dep_pitch_order` function. This function re-sorts the notes first by their onset time and then by descending pitch, which is a different logic used in other parts of the C++ codebase but not for the simple fingering prediction task.

This difference in the input note sequence was confirmed to be the sole cause of the final fingering mismatch.

### The Fix
The solution was to modify the Python execution script (`run_python_hmm.py`) to remove the erroneous call to `apply_time_dep_pitch_order`. By processing the notes in the same order as the C++ binary, the Python implementation now produces bit-for-bit identical results for both HMM2 and HMM3 models.

The incorrect `python_v2/` implementation was also removed to prevent its use.
