#pragma once

#include "fp_formats.h"

namespace tensorcore {

inline DecodedOperand decode_fp16(const sc_dt::sc_uint<16>& raw) {
    DecodedOperand value;
    const sc_dt::sc_uint<5> exponent = raw.range(14, 10);
    const sc_dt::sc_uint<10> fraction = raw.range(9, 0);

    value.sign = raw[15];
    value.is_zero = exponent == 0 && fraction == 0;
    value.is_inf = exponent == 31 && fraction == 0;
    value.is_nan = exponent == 31 && fraction != 0;

    if (exponent == 0) {
        value.mantissa = fraction;
        value.scale = -24;
    } else if (exponent != 31) {
        value.mantissa = (sc_dt::sc_uint<11>(1) << 10) | fraction;
        value.scale = static_cast<int>(exponent) - 25;
    }
    return value;
}

inline DecodedOperand decode_fp8e4m3(const sc_dt::sc_uint<8>& raw) {
    DecodedOperand value;
    const sc_dt::sc_uint<4> exponent = raw.range(6, 3);
    const sc_dt::sc_uint<3> fraction = raw.range(2, 0);

    value.sign = raw[7];
    value.is_zero = exponent == 0 && fraction == 0;
    value.is_inf = exponent == 15 && fraction == 0;
    value.is_nan = exponent == 15 && fraction != 0;

    if (exponent == 0) {
        value.mantissa = fraction;
        value.scale = -9;
    } else if (exponent != 15) {
        value.mantissa = (sc_dt::sc_uint<11>(1) << 3) | fraction;
        value.scale = static_cast<int>(exponent) - 10;
    }
    return value;
}

inline DecodedOperand decode_operand(const sc_dt::sc_uint<16>& raw,
                                     bool fp16) {
    if (fp16) {
        return decode_fp16(raw);
    }
    const sc_dt::sc_uint<8> fp8 = raw.range(7, 0);
    return decode_fp8e4m3(fp8);
}

}  // namespace tensorcore
