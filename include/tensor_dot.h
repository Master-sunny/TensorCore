#pragma once

#include <systemc>

#include "fp_decode.h"
#include "fp32_pack.h"

namespace tensorcore {

SC_MODULE(TensorDot) {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_in<bool> nrst{"nrst"};

    sc_core::sc_in<bool> in_valid{"in_valid"};
    sc_core::sc_out<bool> in_ready{"in_ready"};
    sc_core::sc_in<bool> in_format_fp16{"in_format_fp16"};
    sc_core::sc_in<sc_dt::sc_biguint<kInputVectorWidth>> in_a{"in_a"};
    sc_core::sc_in<sc_dt::sc_biguint<kInputVectorWidth>> in_b{"in_b"};

    sc_core::sc_out<bool> out_valid{"out_valid"};
    sc_core::sc_in<bool> out_ready{"out_ready"};
    sc_core::sc_out<sc_dt::sc_uint<32>> out_result{"out_result"};
    sc_core::sc_out<bool> out_invalid{"out_invalid"};
    sc_core::sc_out<bool> out_overflow{"out_overflow"};
    sc_core::sc_out<bool> out_underflow{"out_underflow"};
    sc_core::sc_out<bool> out_inexact{"out_inexact"};

    using AccType = sc_dt::sc_bigint<kAccumulatorWidth>;

    AccType products[32];
    AccType sums16[16];
    AccType sums8[8];
    AccType sums4[4];
    AccType sums2[2];
    AccType exact_sum;

    SpecialState special1;
    SpecialState special2;
    SpecialState special3;
    SpecialState special4;
    SpecialState special5;
    SpecialState special6;

    sc_core::sc_signal<bool> valid1{"valid1"};
    sc_core::sc_signal<bool> valid2{"valid2"};
    sc_core::sc_signal<bool> valid3{"valid3"};
    sc_core::sc_signal<bool> valid4{"valid4"};
    sc_core::sc_signal<bool> valid5{"valid5"};
    sc_core::sc_signal<bool> valid6{"valid6"};
    sc_core::sc_signal<bool> result_valid{"result_valid"};
    sc_core::sc_signal<sc_dt::sc_uint<32>> result_bits{"result_bits"};
    sc_core::sc_signal<bool> result_invalid{"result_invalid"};
    sc_core::sc_signal<bool> result_overflow{"result_overflow"};
    sc_core::sc_signal<bool> result_underflow{"result_underflow"};
    sc_core::sc_signal<bool> result_inexact{"result_inexact"};

    void drive_ports() {
        const bool can_advance = !result_valid.read() || out_ready.read();
        in_ready.write(can_advance);
        out_valid.write(result_valid.read());
        out_result.write(result_bits.read());
        out_invalid.write(result_invalid.read());
        out_overflow.write(result_overflow.read());
        out_underflow.write(result_underflow.read());
        out_inexact.write(result_inexact.read());
    }

    static AccType make_product(const DecodedOperand& a,
                                const DecodedOperand& b) {
        AccType aligned = 0;
        if (!a.is_zero && !b.is_zero && !a.is_inf && !b.is_inf &&
            !a.is_nan && !b.is_nan) {
            const sc_dt::sc_uint<22> product = a.mantissa * b.mantissa;
            const int shift = static_cast<int>(a.scale) +
                              static_cast<int>(b.scale) -
                              kAccumulatorLsbExponent;
            sc_dt::sc_biguint<kAccumulatorWidth> magnitude;
            magnitude = product;
            magnitude <<= shift;
            aligned = magnitude;
            if (a.sign != b.sign) {
                aligned = -aligned;
            }
        }
        return aligned;
    }

    static void classify_product(const DecodedOperand& a,
                                 const DecodedOperand& b,
                                 SpecialState& state) {
        const bool invalid_product =
            (a.is_zero && b.is_inf) || (a.is_inf && b.is_zero);
        if (a.is_nan || b.is_nan) {
            state.nan = true;
        } else if (invalid_product) {
            state.invalid = true;
        } else if (a.is_inf || b.is_inf) {
            if (a.sign != b.sign) {
                state.neg_inf = true;
            } else {
                state.pos_inf = true;
            }
        }
    }

    void pipeline_proc() {
        if (!nrst.read()) {
            valid1.write(false);
            valid2.write(false);
            valid3.write(false);
            valid4.write(false);
            valid5.write(false);
            valid6.write(false);
            result_valid.write(false);
            result_bits.write(0);
            result_invalid.write(false);
            result_overflow.write(false);
            result_underflow.write(false);
            result_inexact.write(false);
            for (unsigned i = 0; i < 32; ++i) products[i] = 0;
            for (unsigned i = 0; i < 16; ++i) sums16[i] = 0;
            for (unsigned i = 0; i < 8; ++i) sums8[i] = 0;
            for (unsigned i = 0; i < 4; ++i) sums4[i] = 0;
            for (unsigned i = 0; i < 2; ++i) sums2[i] = 0;
            exact_sum = 0;
            special1 = SpecialState();
            special2 = SpecialState();
            special3 = SpecialState();
            special4 = SpecialState();
            special5 = SpecialState();
            special6 = SpecialState();
        } else if (!result_valid.read() || out_ready.read()) {
            if (valid6.read()) {
            const bool infinity_conflict = special6.pos_inf && special6.neg_inf;
            const bool invalid = special6.invalid || infinity_conflict;
            result_invalid.write(invalid);
            result_overflow.write(false);
            result_underflow.write(false);
            result_inexact.write(false);

            if (special6.nan || invalid) {
                result_bits.write(0x7fc00000U);
            } else if (special6.pos_inf) {
                result_bits.write(0x7f800000U);
            } else if (special6.neg_inf) {
                result_bits.write(0xff800000U);
            } else {
                const PackedResult packed = pack_finite_fp32(exact_sum);
                result_bits.write(packed.bits);
                result_overflow.write(packed.overflow);
                result_underflow.write(packed.underflow);
                result_inexact.write(packed.inexact);
            }
        }
        result_valid.write(valid6.read());

        if (valid5.read()) {
            exact_sum = sums2[0] + sums2[1];
            special6 = special5;
        }
        valid6.write(valid5.read());

        if (valid4.read()) {
            for (unsigned i = 0; i < 2; ++i) {
                sums2[i] = sums4[2 * i] + sums4[2 * i + 1];
            }
            special5 = special4;
        }
        valid5.write(valid4.read());

        if (valid3.read()) {
            for (unsigned i = 0; i < 4; ++i) {
                sums4[i] = sums8[2 * i] + sums8[2 * i + 1];
            }
            special4 = special3;
        }
        valid4.write(valid3.read());

        if (valid2.read()) {
            for (unsigned i = 0; i < 8; ++i) {
                sums8[i] = sums16[2 * i] + sums16[2 * i + 1];
            }
            special3 = special2;
        }
        valid3.write(valid2.read());

        if (valid1.read()) {
            for (unsigned i = 0; i < 16; ++i) {
                sums16[i] = products[2 * i] + products[2 * i + 1];
            }
            special2 = special1;
        }
        valid2.write(valid1.read());

        const bool accept = in_valid.read();
        if (accept) {
            SpecialState incoming;
            const bool fp16 = in_format_fp16.read();
            const sc_dt::sc_biguint<kInputVectorWidth> a_vector = in_a.read();
            const sc_dt::sc_biguint<kInputVectorWidth> b_vector = in_b.read();
            for (unsigned i = 0; i < 32; ++i) {
                sc_dt::sc_uint<16> a_raw;
                sc_dt::sc_uint<16> b_raw;
                a_raw = a_vector.range(16 * i + 15, 16 * i);
                b_raw = b_vector.range(16 * i + 15, 16 * i);
                const DecodedOperand a = decode_operand(a_raw, fp16);
                const DecodedOperand b = decode_operand(b_raw, fp16);
                products[i] = make_product(a, b);
                classify_product(a, b, incoming);
            }
            special1 = incoming;
        }
            valid1.write(accept);
        }
    }

    SC_CTOR(TensorDot) {
        SC_METHOD(drive_ports);
        sensitive << result_valid << result_bits << result_invalid
                  << result_overflow << result_underflow << result_inexact
                  << out_ready;

        SC_METHOD(pipeline_proc);
        sensitive << clk.pos() << nrst.neg();
    }
};

}  // namespace tensorcore
