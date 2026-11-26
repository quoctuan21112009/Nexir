#include <cstdint>
#include <iostream>

std::int64_t compute(std::int64_t n) {
    std::int64_t x = 1;
    for (std::int64_t i = 0; i < n; ++i) {
        x = (x * 1664525 + 1013904223) % 2147483647;
    }
    return x;
}

int main() {
    std::cout << compute(20000000) << '\n';
}
