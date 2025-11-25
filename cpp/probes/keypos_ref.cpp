#include "KeyPos_v161230.hpp"
#include <iostream>

int main() {
    // Dump all MIDI pitches 0-127
    for (int p = 0; p < 128; ++p) {
        KeyPos kp = PitchToKeyPos(p);
        std::cout << p << " " << kp.x << " " << kp.y << "\n";
    }
    return 0;
}
