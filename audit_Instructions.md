# **Project Verification: Python HMM Port**

## **Objective**

Your role is to act as the final quality gate for the Python HMM port. The first developer claims to have created a high-performance, "bit-for-bit identical" port. Your task is to **rigorously verify every aspect of this claim**.

This is not a high-level code review. You will perform a deep, adversarial audit of the code, the tests, and the underlying data to confirm its correctness, robustness, and performance. **Trust nothing; verify everything.**

Then you will fix any issues you've found.

Follow this plan sequentially. Do not proceed to the next phase until the current one is fully validated.


## **Phase 0: Frequently asked questions**


1. Definition of "Bit-for-Bit Identical"

    * Final Output: The final integer sequence of fingerings [f_1, f_2, ..., f_n] produced by the Python port must be an exact, 100% match to the sequence produced by the C++ reference binary for any given input score. There is zero tolerance for deviation here, except for the sign of integers (negative or positive) : Note that in PIG files (.txt), a finger on the left hand is represented with a negative integer. The very first check you have to do is : 
        - Does the python code outputs negative or positive integers for the left hand ?
        - Does the C++ code outputs or positive integers for the left hand ?
    A disparity at this level does not constitute a failure, but must be taken into account so that you write reliable tests.

    * Intermediate Values: A small tolerance (e.g., absolute difference < 1e-6) is acceptable for comparing intermediate floating-point values (like log-probabilities in the Viterbi trellis). This is standard practice to account for minor differences in floating-point arithmetic between languages and compilers.

The reason for this strictness on the final output is that the Viterbi algorithm is a series of argmax operations. Even a minuscule floating-point difference (1e-10) at a critical step can cause the argmax to select a different path, leading to a completely different final sequence. Therefore, the integer sequence is the non-negotiable ground truth.

2. Third-Order HMM Implementation
You should expect to find a separate, dedicated viterbi_3rd_order_numba function.

The first developer's claim that the parameter loader is order-agnostic may be true, but the Viterbi algorithm itself is not. Here is why a separate function is required:

    * State Representation: The state in the Viterbi trellis is fundamentally different. For Order-2, the state at step n depends on the finger at n-1. The DP table is effectively 2D ([notes, fingers]). For Order-3, the state depends on the fingers at n-1 and n-2. The DP table is 3D ([notes, finger_prev, finger_curr]).

    * Mathematical Formula: The log-probability calculation for Order-3 involves different parameters (trProb3, outProb3, w3, lam2) and depends on a deeper history (n-3).

    * Performance: A single generic function trying to handle both orders with conditional logic (if order == 2: ... else: ...) would be complex and would prevent Numba from generating maximally efficient, type-specialized machine code.

Actionable Insight: If a viterbi_3rd_order_numba function (or equivalent) is missing, this is the first major finding of your audit. It would indicate an incomplete implementation that does not fulfill the requirements of the original plan. You should document this as a critical failure.

3. Adversarial Test File Selection

Your instinct to first identify the currently used files is correct. A targeted selection is superior to a random one. I recommend choosing three new files based on these adversarial criteria to stress-test known weak points of the algorithm:

The Chord-Heavy Piece: Select a file with dense, complex chords and simultaneous notes.
Why: This directly stress-tests the TimeDepPitchOrder logic, which was identified as a major "trap." A failure here would indicate the developer's fix was not robust.
The Fast, Scalic Piece: Select a file with rapid, sequential, single-note passages (e.g., scales or arpeggios).
Why: This stress-tests the shortTimeCost and delPitch penalty logic, which is applied specifically to fast transitions. It also tests the core transition probability logic in a high-volume scenario.
The "Anomalous" Piece: Select a file that looks different from the others.
Why: To catch "unknown unknowns." Look for a piece with many finger substitutions (e.g., 4_1 notations), unusual rhythmic patterns, or large leaps across the keyboard.
This strategic selection is designed to maximize the probability of finding bugs that a standard test suite might miss.

4. Handling of Discrepancies
Your primary goal is to REPORT, THEN to fix.

As the auditor, your responsibility is to maintain the integrity of the verification process. Fixing the code as soon as you find an error would invalidate the audit, as you would be testing your own work, not the work delivered by the previous developer.

Follow this strict protocol upon discovering any discrepancy:

Halt the specific test or verification step that failed.
Isolate the failure. Create the smallest possible, reproducible case. For example: "Running the score_dump C++ probe on scores/XYZ.txt produces [60, 64, 62] for the first three pitches, but the Python apply_time_dep_pitch_order function produces [60, 62, 64]. This proves the sorting logic is incorrect for this specific chord."
Document this finding precisely and unambiguously in the VERIFICATION_REPORT.md under the "Issues Found" section. Include the command to reproduce the failure.
Proceed with other, unrelated parts of the audit. For example, if an Order-3 Viterbi test fails, you can still proceed to audit the training.py or evaluate.py modules. The goal is to provide a complete picture of the project's state.
The final report should be a clear and factual account of the state of the delivered code. And then, as a final step, you can start fixing.

---

## **Phase 1: Environment Sanity Check & Code Review**

**Goal:** Ensure the project is reproducible and understand the codebase before testing.

1.  **Repository Checkout & Environment Setup:**
    *   Do not work with files from the `main` branch, use only files from the `feat/fix-hmm-viterbi-logic` branch (using `git checkout -b feat/fix-hmm-viterbi-logic origin/feat/fix-hmm-viterbi-logic)
`
    *   Set up your Python environment and install requirements
        ```bash
        pip install -r requirements.txt
        ```
    *   Compile the C++ reference code to ensure your environment is sound:
        ```bash
        cd cpp && ./compile.sh && cd ..
        ```

2.  **Initial Test Run:**
    *   Execute the existing test suite once to establish a baseline.
        ```bash
        pytest -v
        ```
    *   **Expected Outcome:** All tests should pass. If any fail, document this immediately. This is your starting point.

3.  **Critical Code Audit (Read the Code First):**
    *   **Primary checks:** 
        Explore the code and writes scripts to generate outputs with both C++ and Python versions to determine if one or both of the two versions generate negative outputs for the left hand. Remember this information, as it will be essential later.
    *   **`utils.py`:**
        *   **PIG Parser:** Scrutinize the "Token Stream" logic. Does the comment-stripping regex have any edge cases (e.g., musical notation that might look like a comment)? Is it truly whitespace-agnostic?
        *   **`TimeDepPitchOrder`:** This was a known "trap." Manually trace the logic with a small, tricky chord cluster (e.g., notes with identical onsets, notes with onsets differing by `0.029s` and `0.031s`). Does the clustering and sorting logic exactly match your reading of the C++ header `PianoFingering_v170101_2.hpp`?
    *   **`model.py`:**
        *   **`HMMParameters`:** Open `cpp/Code/FingeringHMM_v180925.hpp`. Compare the C++ `ReadParamFile` function for `FingeringHMM_3rd` side-by-side with the Python implementation. Verify the **linear interpolation logic** (`lam1`, `lam2`) is identical. This is a potential source of subtle numerical drift.
        *   **`viterbi_2nd_order_numba`:** Compare the main loop side-by-side with `FingeringHMM_2nd::Viterbi`. Check for:
            *   Identical `shortTimeCost` threshold (`0.03s`).
            *   Identical application of the `delPitch` penalty logic.
            *   Exact replication of the log-probability summation formula.

---

## **Phase 2: Verifying the Verifiers (Testing the Tests)**

**Goal:** Ensure the "Ground Truth" data the tests rely on is valid and reproducible. This step is critical to validate the entire testing strategy.

1.  **Isolate Existing Reference Data:**
    *   Move the existing C++-generated reference files to a temporary location. **Do not delete them yet.**
        ```bash
        mv python/tests/ref_outputs python/tests/ref_outputs_original
        mkdir python/tests/ref_outputs
        ```

2.  **Independently Regenerate All Reference Files:**
    *   You will now re-compile and re-run all C++ probes mentioned in the developer's `feedback_from_the_previous_developer.md`.
        ```bash
        # Re-compile probes
        g++ -O2 -std=c++17 -I cpp/Code cpp/probes/keypos_ref.cpp -o keypos_ref
        g++ -O2 -std=c++17 -I cpp/Code cpp/probes/score_dump.cpp -o score_dump
        # ... compile all other probes ...

        # Re-generate data
        ./keypos_ref > python/tests/ref_outputs/pitch_to_keypos_reference.txt
        ./score_dump scores/001-1_fingering.txt > python/tests/ref_outputs/score_dump_001-1.txt
        # ... generate all other reference files ...
        ```

3.  **Compare New vs. Original Reference Data:**
    *   Use a file comparison tool to check for any differences between your newly generated files and the original ones.
        ```bash
        diff -rq python/tests/ref_outputs/ python/tests/ref_outputs_original/
        ```
    *   **Expected Outcome:** The command should report no differences.
    *   **If differences exist:** This is a **critical failure**. It means the original developer's tests may have been passing against stale or incorrectly generated data. Investigate why before proceeding.

4.  **Re-run the Test Suite:**
    *   With your newly generated reference data in place, run the full test suite again.
        ```bash
        pytest -v
        ```
    *   **Expected Outcome:** All tests must still pass.

---

## **Phase 3: Deep Dive & Adversarial Testing**

**Goal:** Stress-test the system with new, unseen data and configurations to find hidden bugs. The existing tests only prove it works on a known subset.

1.  **Test with New Score Files:**
    *   Select **three PIG files** from `scores/` that are **not** used in the existing test suite. Choose a variety:
        *   One very short and simple file.
        *   One long file with complex passages.
        *   One file with significant polyphony (many chords).
    *   For each file:
        1.  Generate the C++ reference output using `FingeringHMM2_Run`.
        2.  Write a new test case in `test_viterbi.py` that runs the Python port on that file.
        3.  Assert that the Python fingering output is **100% identical** to the C++ reference output.

2.  **Test the Order-3 HMM (Critical Validation):**
    *   The developer claims the parameter loader is order-agnostic. Verify this.
    *   Create a C++ reference output using the **Order-3 model**:
        ```bash
        ./cpp/Binary/FingeringHMM3_Run scores/001-1_fingering.txt python/tests/ref_outputs/ref_001_order3.txt
        ```
    *   Write a new test case that:
        1.  Loads `cpp/param_FHMM3.txt` with `HMMParameters`.
        2.  Runs the Viterbi algorithm (you may need to implement/verify the order-3 version if it doesn't exist).
        3.  Asserts the output is **100% identical** to `ref_001_order3.txt`.

3.  **Edge Case Testing:**
    *   What happens with malformed input? Create temporary score files and write tests for:
        *   An empty file.
        *   A file with only one note (should produce no fingering or handle it gracefully).
        *   A file with only left-hand notes.
        *   A file with invalid characters or missing columns.
    *   **Expected Outcome:** The system should either handle these cases gracefully (e.g., return an empty list) or raise informative, specific exceptions (`ValueError`, `FileNotFoundError`), not crash with an `IndexError` or `TypeError`.

---

## **Phase 4: Performance & Optimization Verification**

**Goal:** Verify the "high-performance" claim and ensure Numba is working correctly.

1.  **Benchmark the Hot Path:**
    *   Write a simple benchmark script (e.g., `python/benchmark.py`).
    *   Use Python's `timeit` module to measure the execution time of a full Viterbi run on a non-trivial score file (e.g., the complex one from Phase 3).
    *   Run it 10-100 times to get a stable average.
    *   **Expected Outcome:** Performance should be reasonable (e.g., well under 1 second for a typical piece). Document the final timing.

2.  **Verify Numba `nopython` Mode:**
    *   Numba can silently fall back to "object mode," which is often slower than pure Python. You must verify this has not happened.
    *   In `model.py`, find the main Numba-jitted Viterbi function. Add `.inspect_types()` to view the compilation report.
        ```python
        # In your test or a temporary script
        from python.model import viterbi_2nd_order_numba
        viterbi_2nd_order_numba.inspect_types()
        ```
    *   **Action:** Read the output. Look for any variables typed as `pyobject`.
    *   **Expected Outcome:** There should be **zero `pyobject` types** in the main Viterbi loop. If you find any, the performance claim is invalid, and this is a failure that needs to be fixed.

---

## **Phase 5: Final Verification Report**

**Goal:** Produce a final report summarizing your findings.

Create a document named `VERIFICATION_REPORT.md` with the following checklist. Fill it out completely.

```md
# HMM Port Verification Report

## Phase 1: Code Review
- [ ] `utils.py`: PIG parser logic is robust and correct.
- [ ] `utils.py`: `TimeDepPitchOrder` logic is a 1:1 match with the C++ implementation.
- [ ] `model.py`: `HMMParameters` correctly implements linear interpolation for Order-3.
- [ ] `model.py`: Viterbi implementation correctly applies `shortTimeCost` and `delPitch` penalties.

## Phase 2: Test Infrastructure
- [ ] Reference files regenerated successfully from C++ probes.
- [ ] Regenerated reference files are identical to the original committed files.
- [ ] Test suite passes with regenerated reference files.

## Phase 3: Deep Dive & Adversarial Testing
- [ ] System produces correct output for 3 new, unseen score files.
- [ ] System produces correct output using Order-3 HMM parameters (`param_FHMM3.txt`).
- [ ] System handles edge cases (empty file, single-note file) gracefully without crashing.

## Phase 4: Performance
- [ ] Benchmark execution time for [specify score file]: ______ seconds.
- [ ] Numba `inspect_types()` report confirms **no `pyobject` fallback** in the Viterbi hot path.

## Overall Assessment
- [ ] **PASS/FAIL:** The project meets the "bit-for-bit identical" and "high-performance" requirements.

## Issues Found (if any)
1.  [Description of Issue 1]
2.  [Description of Issue 2]
3.  and so on ....

## Suggestions for Debug / Improvements / corrections
- [Suggestion 1]
- [Suggestion 2]
- and so on
```

## **Phase 6: Final Corrections**

- Read the C++ and Python code, as well as your report. 
- Remember all the anomalies you found; 

With all this informations, it should be easy to correct each error.

Finally Rerun all the tests you performed to verify that your corrections haven't introduced any regressions. If any problem occurs, restart again.


