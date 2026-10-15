#include "reference_model.h"

namespace tensorcore::test {
namespace {

using WideInt = __int128_t;
using WideUInt = __uint128_t;

struct Operand {
    bool sign = false;
    bool zero = true;
    bool inf = false;
    bool nan = false;
    std::uint32_t mantissa = 0;
    int scale = 0;
};

Operand decode(std::uint16_t raw, bool fp16) {
    Operand value;
    if (fp16) {
        const std::uint32_t exponent = (raw >> 10) & 0x1f;
        const std::uint32_t fraction = raw & 0x3ff;
        value.sign = (raw >> 15) != 0;
        value.zero = exponent == 0 && fraction == 0;
        value.inf = exponent == 0x1f && fraction == 0;
        value.nan = exponent == 0x1f && fraction != 0;
        if (exponent == 0) {
            value.mantissa = fraction;
            value.scale = -24;
        } else if (exponent != 0x1f) {
            value.mantissa = 0x400 | fraction;
            value.scale = static_cast<int>(exponent) - 25;
        }
    } else {
        raw &= 0xff;
        const std::uint32_t exponent = (raw >> 3) & 0xf;
        const std::uint32_t fraction = raw & 0x7;
        value.sign = (raw >> 7) != 0;
        value.zero = exponent == 0 && fraction == 0;
        value.inf = exponent == 0xf && fraction == 0;
        value.nan = exponent == 0xf && fraction != 0;
        if (exponent == 0) {
            value.mantissa = fraction;
            value.scale = -9;
        } else if (exponent != 0xf) {
            value.mantissa = 0x8 | fraction;
            value.scale = static_cast<int>(exponent) - 10;
        }
    }
    return value;
}

ExpectedResult pack(WideInt sum) {
    ExpectedResult result{0, false, false, false, false};
    if (sum == 0) {
        return result;
    }

    const bool sign = sum < 0;
    const WideUInt magnitude = sign ? static_cast<WideUInt>(-sum)
                                    : static_cast<WideUInt>(sum);
    unsigned leading = 0;
    for (unsigned bit = 0; bit < 128; ++bit) {
        if ((magnitude >> bit) & 1U) {
            leading = bit;
        }
    }
    int exponent = static_cast<int>(leading) - 48;
    WideUInt significand;

    if (leading > 23) {
        const unsigned shift = leading - 23;
        significand = magnitude >> shift;
        const WideUInt remainder_mask = (WideUInt(1) << shift) - 1;
        const WideUInt remainder = magnitude & remainder_mask;
        const WideUInt halfway = WideUInt(1) << (shift - 1);
        result.inexact = remainder != 0;
        if (remainder > halfway ||
            (remainder == halfway && (significand & 1U))) {
            ++significand;
        }
    } else {
        significand = magnitude << (23 - leading);
    }

    if (significand & (WideUInt(1) << 24)) {
        significand >>= 1;
        ++exponent;
    }

    if (exponent > 127) {
        result.bits = sign ? 0xff800000U : 0x7f800000U;
        result.overflow = true;
        result.inexact = true;
        return result;
    }
    if (exponent < -126) {
        result.bits = sign ? 0x80000000U : 0;
        result.underflow = true;
        result.inexact = true;
        return result;
    }

    const std::uint32_t sig = static_cast<std::uint32_t>(significand);
    result.bits = (sign ? 0x80000000U : 0U) |
                  (static_cast<std::uint32_t>(exponent + 127) << 23) |
                  (sig & 0x7fffffU);
    return result;
}

}  // namespace

ExpectedResult reference_dot(const std::array<std::uint16_t, 32>& a,
                             const std::array<std::uint16_t, 32>& b,
                             bool fp16) {
    WideInt sum = 0;
    bool nan = false;
    bool invalid = false;
    bool pos_inf = false;
    bool neg_inf = false;

    for (unsigned i = 0; i < 32; ++i) {
        const Operand lhs = decode(a[i], fp16);
        const Operand rhs = decode(b[i], fp16);
        const bool invalid_product =
            (lhs.zero && rhs.inf) || (lhs.inf && rhs.zero);

        if (lhs.nan || rhs.nan) {
            nan = true;
        } else if (invalid_product) {
            invalid = true;
        } else if (lhs.inf || rhs.inf) {
            if (lhs.sign != rhs.sign) {
                neg_inf = true;
            } else {
                pos_inf = true;
            }
        } else if (!lhs.zero && !rhs.zero) {
            WideInt product = static_cast<WideInt>(lhs.mantissa) * rhs.mantissa;
            product <<= lhs.scale + rhs.scale + 48;
            if (lhs.sign != rhs.sign) {
                product = -product;
            }
            sum += product;
        }
    }

    if (pos_inf && neg_inf) {
        invalid = true;
    }
    if (nan || invalid) {
        return {0x7fc00000U, invalid, false, false, false};
    }
    if (pos_inf) {
        return {0x7f800000U, false, false, false, false};
    }
    if (neg_inf) {
        return {0xff800000U, false, false, false, false};
    }
    return pack(sum);
}

}  // namespace tensorcore::test
