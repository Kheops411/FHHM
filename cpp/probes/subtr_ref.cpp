#include "KeyPos_v161230.hpp"
#include <iostream>

int main() {
    // Dump a range of subtractions to cover edge cases
    for (int x1=-40; x1<=40; x1+=5) {
        for (int y1=0; y1<=1; ++y1) {
            for (int x2=-40; x2<=40; x2+=5) {
                for (int y2=0; y2<=1; ++y2) {
                    KeyPos kp1; kp1.x=x1; kp1.y=y1;
                    KeyPos kp2; kp2.x=x2; kp2.y=y2;
                    KeyPos r = SubtrKeyPos(kp1,kp2);
                    std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << " " << r.x << " " << r.y << "\n";
                }
            }
        }
    }
    return 0;
}
