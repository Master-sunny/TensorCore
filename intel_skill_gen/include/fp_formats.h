#pragma once

#include <systemc>

namespace tensorcore {

constexpr unsigned kNumTerms = 32;
constexpr unsigned kOperandSlotWidth = 16;
constexpr unsigned kInputVectorWidth = kNumTerms * kOperandSlotWidth;
constexpr int kAccumulatorLsbExponent = -48;
constexpr int kAccumulatorWidth = 88;

enum InputFormat : unsigned {
    kFp8E4M3 = 0,
    kFp16 = 1,
};

struct DecodedOperand {
    bool sign;
    bool is_zero;
    bool is_inf;
    bool is_nan;
    sc_dt::sc_uint<11> mantissa;
    sc_dt::sc_int<8> scale;

    DecodedOperand()
        : sign(false), is_zero(true), is_inf(false), is_nan(false),
          mantissa(0), scale(0) {}
};

struct SpecialState {
    bool nan;
    bool invalid;
    bool pos_inf;
    bool neg_inf;

    SpecialState()
        : nan(false), invalid(false), pos_inf(false), neg_inf(false) {}
};

}  // namespace tensorcore
