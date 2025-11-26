# **Project Verification: Python HMM Port**

## **Objective**

Your role is to act as the final quality gate for the Python HMM port. The first developer claims to have created a high-performance, "bit-for-bit identical" port. Your task is to **rigorously verify every aspect of this claim**.

This is not a high-level code review. You will perform a deep, adversarial audit of the code, the tests, and the underlying data to confirm its correctness, robustness, and performance. **Trust nothing; verify everything.**

Follow this plan sequentially. Do not proceed to the next phase until the current one is fully validated.

---

## **Phase 1: Environment Sanity Check & Code Review**

**Goal:** Ensure the project is reproducible and understand the codebase before testing.

1.  **Repository Checkout & Environment Setup:**
    *   Perform a fresh `git clone` of the repository.
    *   Set up the Python environment exactly as a new user would:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt # Assuming one exists, otherwise install numpy numba pytest
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
    *   You will now re-compile and re-run all C++ probes mentioned in the developer's `README.md`.
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

## Suggestions for Improvement (Optional)
- [Suggestion 1]
