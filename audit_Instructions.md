### Instructions: Debugging and Aligning Python HMM Fingering Models with C++ Reference**

**Objective:**
Your primary mission is to debug and validate one of two Python implementations of a Piano Fingering algorithm (2nd and 3rd order Hidden Markov Models) to ensure it produces the **exact same output** as the reference C++ implementation.

---

### **1. Project Context**

You are working on a program that automatically suggests fingerings for piano scores. The core logic is based on the research paper "Statistical Learning and Estimation of Piano Fingering," which uses statistical models (Hidden Markov Models - HMMs) instead of hard-coded rules.

*   **The Ground Truth:** We have a C++ implementation from the original authors. This code is considered **correct and is our reference**.
*   **The Target:** We have two Python versions (`python/` and `python_v2/`) that are intended to be ports of the C++ HMM models (specifically 2nd and 3rd order). Both use `numpy` and `numba` for performance.
*   **The Data:** We use the **PIG Dataset**, where scores are represented as `.txt` files in the `scores/` directory.
*   **Your Goal:** Find out which Python version is better (less buggy, more performant), fix it completely until it perfectly matches the C++ output, and then discard the other version.

---

### **2. Critical Information: The Fingering Convention**

This is extremely important for comparing outputs:

*   **C++ / PIG Dataset Convention:** The ground truth data and C++ program use **positive integers (1-5)** for the right hand and **negative integers (-1 to -5)** for the left hand.
*   **Python Convention:** It is unknown if the Python versions follow this convention. They might use positive integers for both hands, possibly with a separate flag for the hand.

**Your Action:** When comparing outputs, you must account for this. The **sequence of absolute finger numbers** for a given hand must match. For example, a C++ left-hand output of `[-2, -3, -1]` must correspond to a Python left-hand output of `[2, 3, 1]`. **Do not treat a sign mismatch as a bug in itself.**

---

### **3. Step-by-Step Plan of Action**

Follow this procedure methodically. Do not skip steps.

#### **Step 0: Setup**

1.  **Compile the C++ Code:** Navigate to the `cpp/` directory and run `./compile.sh`. This will create the necessary binaries in `cpp/Binary/`.
2.  **Set up Python Environments:** Ensure you can run both the `python/` and `python_v2/` scripts. Install any necessary dependencies (`numpy`, `numba`, etc.).

#### **Step 1: Establish the Ground Truth**

Before touching any Python code, you must know what the correct output looks like.

1.  Pick a simple test file from the `scores/` directory (e.g., one that is short and for a single hand).
2.  Run the C++ HMM2 and HMM3 models on it:
    ```bash
    cd cpp/
    # For HMM order 2
    ./run_FHMM2.sh ../scores/your_test_file.txt output_cpp_hmm2.txt
    # For HMM order 3
    ./run_FHMM3.sh ../scores/your_test_file.txt output_cpp_hmm3.txt
    ```
3.  Inspect the output files (`output_cpp_hmm2.txt`, `output_cpp_hmm3.txt`). This is the target you must replicate perfectly.

#### **Step 2: Rigorous Testing and Data Flow Analysis**

Your main task is to identify the **first point of divergence** between the Python and C++ logic. Do not just look at the final output. Test both `python/` and `python_v2/` in parallel using the same input file.

For a given model (start with HMM2), you will trace the data flow and compare intermediate results.

1.  **Input Parsing:**
    *   **Action:** Add temporary print statements or use a debugger in the Python code to display the data structure that holds the notes after parsing the input `.txt` file.
    *   **Verification:** Does it correctly parse the pitch, onset, offset, and hand for every note? Does it match the sequence of notes the C++ program would process?

2.  **Data Transformation (Pre-Viterbi):**
    *   **Context:** The paper mentions a "Lattice Representation" for the keyboard pitch. The C++ code implements this logic. The Python code should do the same.
    *   **Action:** Find where the raw MIDI pitch is converted into this internal representation. Print these transformed values.
    *   **Verification:** Manually verify or create a C++ debug build to see its internal representation. Do the Python versions produce the identical transformed data for the same note sequence?

3.  **Viterbi Algorithm - Inputs:**
    *   **Context:** The Viterbi algorithm's correctness depends entirely on the transition and emission probability matrices it receives. The pre-trained parameters are in the `cpp/Code/param_FHMM2.txt` and `param_FHMM3.txt` files.
    *   **Action:** Modify the Python code to load the *exact same* parameter files as the C++ version. Before the Viterbi function is called, print the transition and emission log-probability matrices.
    *   **Verification:** This is the most critical check. The probability matrices fed into the Python Viterbi algorithm **must be numerically identical** to those used by the C++ version. Any discrepancy here is a major bug, likely in the data transformation or probability calculation logic.

4.  **Viterbi Algorithm - Output:**
    *   **Action:** Run the Viterbi algorithm. The immediate output is a sequence of state indices (the "most likely path"). Print this raw sequence of indices.
    *   **Verification:** Does the sequence of indices from the Python Viterbi implementation match the one from the C++ implementation? If the inputs were identical but the outputs differ, the bug is in the Viterbi algorithm's implementation itself.

5.  **Final Output Formatting:**
    *   **Action:** The raw Viterbi path is converted back into finger numbers (1-5).
    *   **Verification:** Check this final conversion step. Compare the final `.txt` output from Python with your ground truth C++ output (`output_cpp_hmm2.txt`). Remember to handle the negative sign convention for the left hand.

#### **Step 3: Choose the Best Candidate Version**

After testing both `python/` and `python_v2/` on HMM2 and HMM3 models:

1.  **Primary Criterion: Correctness.** Which version has fewer bugs or bugs that are easier to fix? Which one is closer to the C++ intermediate values? The version with the more logical and traceable data flow is superior.
2.  **Secondary Criterion: Performance.** If both versions are equally correct (or incorrect), run them on a large score file. Is one significantly faster? A performance difference of less than 10% is negligible.

**Decision:** Choose one version to be the definitive one. You will now focus exclusively on fixing it.

#### **Step 4: Debug and Fix**

Using the insights from your data flow analysis, methodically fix every bug in the chosen Python version until it passes all checks from Step 2.

*   The final code must produce **bit-for-bit identical output files** to the C++ reference for any given input score and HMM order (2 and 3).
*   Test on at least 3-4 different scores (including some with both hands) to ensure your fixes are robust.

#### **Step 5: Cleanup**

1.  Delete the entire directory of the rejected Python version (either `python/` or `python_v2/`).
2.  Rename the directory of the fixed version to `python/` if it isn't already.
3.  Remove all your temporary print statements and debugging code. Ensure the code is clean.

---

### **4. Workflow and Deliverables**

*   **Branch:** All your work—including commits for testing, debugging, and the final fix—**must be done on the `feat/fix-hmm-viterbi-logic` branch.**
*   **Pull Request (PR):** When your work is complete, submit a Pull Request to merge `feat/fix-hmm-viterbi-logic` back into the target branch specified by the project lead (likely `main` or `develop`, but for now, target `feat/fix-hmm-viterbi-logic` if unclear).
*   **PR Description:** Your PR description should summarize which version you chose, why, and the key fixes you implemented.

### **Definition of Done (Checklist)**

Your task is complete when:
- [ ] You have a single `python/` directory.
- [ ] The code inside `python/` correctly implements both HMM2 and HMM3 models.
- [ ] For any given input file from `scores/`, the Python code generates an output file that is **identical** to the one generated by the corresponding C++ binary (`FingeringHMM2_Run`, `FingeringHMM3_Run`).
- [ ] The rejected Python version has been completely removed from the repository.
- [ ] All your work is on the correct branch.