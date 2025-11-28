import sys

def parse_output_file(filepath):
    """Parses an output file to extract a map of {original_idx: finger}."""
    finger_map = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('//') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    original_idx = int(parts[0])
                    finger = int(parts[7])
                    finger_map[original_idx] = finger
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    except (ValueError, IndexError) as e:
        print(f"Error parsing file {filepath}: {e}")
        sys.exit(1)
    return finger_map

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_outputs.py <cpp_output_file> <python_output_file>")
        sys.exit(1)

    cpp_file = sys.argv[1]
    python_file = sys.argv[2]

    print(f"--- Comparing {cpp_file} (Reference) vs. {python_file} (Python) ---")

    cpp_fingers = parse_output_file(cpp_file)
    python_fingers = parse_output_file(python_file)

    if not cpp_fingers:
        print("Error: Could not parse any valid data from the C++ output file.")
        sys.exit(1)

    all_indices = sorted(list(cpp_fingers.keys()))
    mismatches = []

    for idx in all_indices:
        cpp_finger = cpp_fingers[idx]
        if idx not in python_fingers:
            mismatches.append((idx, cpp_finger, 'Missing'))
        elif python_fingers[idx] != cpp_finger:
            mismatches.append((idx, cpp_finger, python_fingers[idx]))

    if not mismatches:
        print(f"\nSuccess! All {len(all_indices)} notes are identical.")
        sys.exit(0)
    else:
        print(f"\nFound {len(mismatches)} mismatches out of {len(all_indices)} notes:")
        for idx, cpp, py in mismatches:
            print(f"  - Note original_idx={idx}: C++ expected {cpp}, Python got {py}")
        sys.exit(1)

if __name__ == "__main__":
    main()
