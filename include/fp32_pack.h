#pragma once

#include "fp_formats.h"

namespace tensorcore {

struct PackedResult {
    sc_dt::sc_uint<32> bits;
    bool overflow;
    bool underflow;
    bool inexact;

    PackedResult() : bits(0), overflow(false), underflow(false), inexact(false) {}
};

inline PackedResult pack_finite_fp32(
    const sc_dt::sc_bigint<kAccumulatorWidth>& exact_sum) {
    PackedResult result;
    if (exact_sum != 0) {
        const bool sign = exact_sum < 0;
        sc_dt::sc_biguint<kAccumulatorWidth> magnitude;
        if (sign) {
            sc_dt::sc_bigint<kAccumulatorWidth> positive = -exact_sum;
            magnitude = positive;
        } else {
            magnitude = exact_sum;
        }

        int leading = 0;
        for (int bit = 0; bit < kAccumulatorWidth; ++bit) {
            if (magnitude[bit]) {
                leading = bit;
            }
        }

        int exponent = leading + kAccumulatorLsbExponent;
        sc_dt::sc_biguint<25> significand = 0;
        bool guard = false;
        bool sticky = false;

        if (leading > 23) {
            const int shift = leading - 23;
            significand = magnitude >> shift;
            guard = magnitude[shift - 1];
            for (int bit = 0; bit < kAccumulatorWidth; ++bit) {
                if (bit < shift - 1 && magnitude[bit]) {
                    sticky = true;
                }
            }
        } else {
            significand = magnitude << (23 - leading);
        }

        const bool round_up = guard && (sticky || significand[0]);
        result.inexact = guard || sticky;
        if (round_up) {
            significand = significand + 1;
        }
        if (significand[24]) {
            significand >>= 1;
            ++exponent;
        }

        if (exponent > 127) {
            result.bits = sign ? 0xff800000U : 0x7f800000U;
            result.overflow = true;
            result.inexact = true;
        } else if (exponent < -126) {
            // The current accumulator quantum is 2^-48, so this path is
            // reserved for future extensions with smaller inputs.
            result.underflow = true;
            result.inexact = true;
            result.bits = sign ? 0x80000000U : 0U;
        } else {
            const sc_dt::sc_uint<8> exponent_field = exponent + 127;
            sc_dt::sc_uint<23> fraction;
            fraction = significand.range(22, 0);
            result.bits = (sc_dt::sc_uint<32>(sign) << 31) |
                          (sc_dt::sc_uint<32>(exponent_field) << 23) |
                          fraction;
        }
    }
    return result;
}

}  // namespace tensorcore
