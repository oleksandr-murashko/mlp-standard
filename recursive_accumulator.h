#ifndef RECURSIVE_ACCUMULATOR_H
#define RECURSIVE_ACCUMULATOR_H

#include <cstdint>

class RecursiveAccumulator
{
private:
    // maximum number of layers in the binary tree used for recursive addition;
    // we assume that we can safely add up to (2^32-1) numbers when using uint32_t integer type
    static constexpr uint32_t maxLevels = 32;

    double levels [maxLevels];

public:
    RecursiveAccumulator() {
        reset();
    }

    inline void add(double number) {
        for (uint32_t i = 0;  i < maxLevels;  i++) {
            if (levels[i] == 0) {
                // no previous number in current position
                levels[i] = number;
                break;
            } else {
                // add new number to the value in current position, zero this position and
                // transfer number to the next level at next iteration
                number += levels[i];
                levels[i] = 0;
            }
        }
    }

    inline double result() const {
        // add all remaining elements in the levels starting from the bottom
        // to keep as much precision as possible
        double remainingSum = 0;
        for (uint32_t i = 0;  i < maxLevels;  i++) {
            remainingSum += levels[i];
        }

        // add remaining elements to the entire sum and return as a total sum;
        // note: none of the class members is changed because we may need to continue adding numbers
        return remainingSum;
    }

    inline void reset() {
        for (uint32_t i = 0;  i < maxLevels;  i++) {
            levels[i] = 0;
        }
    }
};

#endif // RECURSIVE_ACCUMULATOR_H
