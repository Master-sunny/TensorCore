#pragma once

#include <array>
#include <cstdint>

namespace tensorcore::test {

struct ExpectedResult {
    std::uint32_t bits;
    bool invalid;
    bool overflow;
    bool underflow;
    bool inexact;
};

ExpectedResult reference_dot(const std::array<std::uint16_t, 32>& a,
                             const std::array<std::uint16_t, 32>& b,
                             bool fp16);

}  // namespace tensorcore::test
