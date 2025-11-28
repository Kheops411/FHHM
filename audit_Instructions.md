### Instructions: Debugging and Aligning Python HMM Fingering Models with C++ Reference**

**Objective:**
Your primary mission is to debug and validate one of two Python implementations of a Piano Fingering algorithm (2nd and 3rd order Hidden Markov Models) to ensure it produces the **exact same output** as the reference C++ implementation.

---

### **1. Project Context**

You are working on a program that automatically suggests fingerings for piano scores. The core logic is based on the research paper "Statistical Learning and Estimation of Piano Fingering," which uses statistical models (Hidden Markov Models - HMMs) instead of hard-coded rules.

*   **The Ground Truth:** We have a C++ implementation from the original authors. This code is considered **correct and is our reference**.
*   **The Target:** We have two Python versions (`python/` and `python_v2/`) that are intended to be ports of the C++ HMM models (specifically 2nd and 3rd order). Both use `numpy` and `numba` for performance.
*   **The Data:** We use the **PIG Dataset**, where scores are represented as `.txt` files in the `scores/` directory.
*   **Your Goal:** Find out which Python version is better (less buggy, more performant, closer to the original C++ version), fix it completely until it perfectly matches the C++ output, and then discard the other version.

### **2. Step-by-Step Plan of Action**

Follow this procedure methodically. Do not skip steps.

1.  Read `cpp/README.txt`
2.  Read `python/README.txt`
3.  Read all `.cpp` files in the `/cpp/Code` directory
4.  Read all `.py` files in the `/python/Code` directory
5.  Based on these readings, decide which version of the Python code is closest to the C++ version; that's the one you'll work on.
6.  Install any necessary dependencies (`numpy`, `numba`, etc.)
7.  Generate the fingering prediction for `scores/001-1_fingering.txt` as input file with the C++ binary (`FingeringHMM2_Run`), name it `output_cpp_hmm2.txt`
8.  Generate the fingering prediction for the same `scores/001-1_fingering.txt` as input file with the python version you're working and HMM order 2 model, name it `output_python_final.txt`
9.  Check if the test script `_test_and_debug_runner.py` is correct, otherwise fix it.
10. Run the `_test_and_debug_runner.py` script. You should expect there to be some mismatches. Your role from this point on is to add debugging instructions to the C++ and Python versions and compare them until you find the problem and fix it :
    a. Find the exact notes (their original ID) where there are discrepancies.
    b. Add log statements to both C++ versions and some notes before and after these IDs with mismatch : Log-Probability matrices (LP Matrix), Parsing, Lattice, Indexing, Transitions, Emissions, Weights. Make sure you don't imitate the display of decimals, so you can see if it's a floating-Point problem.
    c. Analyze the logs and note your findings in a file named `findings.md` in the project's root directory.
    d. Based on your findings, formulate hypotheses that you will also note in `findings.md`
    e. Go back to step (b.) with new log statements to verify your hypotheses.
    f. Once the problem is formally identified, attempt a fix and return to the step (10.) 
11. If it's a success, repeat the entire process for the MMH3 model.
12. Test again both MMH2 and MMH3 models with a new input file from `scores`.
