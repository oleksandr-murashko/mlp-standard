#ifndef RANDOMGENERATOR_H
#define RANDOMGENERATOR_H

#include <cstdint>

// a simple linear congruential generator
class RandomGenerator
{
private:
    uint64_t n;

public:
    RandomGenerator(uint64_t seed) {
        n = seed;
    }

    inline uint64_t uint64() {
        const uint64_t a = 3202034522624059733;
        const uint64_t c = 1442695037175000593;
        const uint64_t m = a * n + c;
        n = a * m + c;
        return (m & 0xffffffff00000000LL) | (n >> 32);
    }

    inline double fp64() {
        return uint64() * 5.421010862427522170037264E-20;    // multiply by 2^(-64)
    }
};

#endif // RANDOMGENERATOR_H
