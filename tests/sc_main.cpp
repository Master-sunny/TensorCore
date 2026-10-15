#include <systemc>

#include <array>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "reference_model.h"
#include "tensor_dot.h"

namespace {

using tensorcore::kInputVectorWidth;
using tensorcore::test::ExpectedResult;

struct Transaction {
    bool fp16;
    std::array<std::uint16_t, 32> a;
    std::array<std::uint16_t, 32> b;
    ExpectedResult expected;
    std::uint64_t id;
};

SC_MODULE(Testbench) {
    sc_core::sc_in<bool> clk{"clk"};
    sc_core::sc_out<bool> nrst{"nrst"};
    sc_core::sc_out<bool> in_valid{"in_valid"};
    sc_core::sc_in<bool> in_ready{"in_ready"};
    sc_core::sc_out<bool> in_format_fp16{"in_format_fp16"};
    sc_core::sc_out<sc_dt::sc_biguint<kInputVectorWidth>> in_a{"in_a"};
    sc_core::sc_out<sc_dt::sc_biguint<kInputVectorWidth>> in_b{"in_b"};
    sc_core::sc_in<bool> out_valid{"out_valid"};
    sc_core::sc_out<bool> out_ready{"out_ready"};
    sc_core::sc_in<sc_dt::sc_uint<32>> out_result{"out_result"};
    sc_core::sc_in<bool> out_invalid{"out_invalid"};
    sc_core::sc_in<bool> out_overflow{"out_overflow"};
    sc_core::sc_in<bool> out_underflow{"out_underflow"};
    sc_core::sc_in<bool> out_inexact{"out_inexact"};

    std::mt19937_64 random{0x4e4e50545f444f54ULL};
    std::deque<Transaction> expected_queue;
    std::uint64_t sent = 0;
    std::uint64_t received = 0;

    static sc_dt::sc_biguint<kInputVectorWidth> pack(
        const std::array<std::uint16_t, 32>& values) {
        sc_dt::sc_biguint<kInputVectorWidth> packed = 0;
        for (unsigned i = 0; i < 32; ++i) {
            packed.range(16 * i + 15, 16 * i) = values[i];
        }
        return packed;
    }

    Transaction make_random(bool fp16, std::uint64_t id) {
        Transaction tx{};
        tx.fp16 = fp16;
        tx.id = id;
        for (unsigned i = 0; i < 32; ++i) {
            tx.a[i] = fp16 ? static_cast<std::uint16_t>(random())
                           : static_cast<std::uint16_t>(random() & 0xff);
            tx.b[i] = fp16 ? static_cast<std::uint16_t>(random())
                           : static_cast<std::uint16_t>(random() & 0xff);
            // Most vectors exercise finite arithmetic. Every eighth vector
            // remains raw-bit random to cover NaNs, infinities, and 0*Inf.
            if ((id & 7U) != 0U) {
                if (fp16) {
                    if ((tx.a[i] & 0x7c00U) == 0x7c00U) {
                        tx.a[i] &= 0x7bffU;
                    }
                    if ((tx.b[i] & 0x7c00U) == 0x7c00U) {
                        tx.b[i] &= 0x7bffU;
                    }
                } else {
                    if ((tx.a[i] & 0x78U) == 0x78U) {
                        tx.a[i] &= 0x77U;
                    }
                    if ((tx.b[i] & 0x78U) == 0x78U) {
                        tx.b[i] &= 0x77U;
                    }
                }
            }
        }
        tx.expected = tensorcore::test::reference_dot(tx.a, tx.b, fp16);
        return tx;
    }

    Transaction make_uniform(bool fp16, std::uint16_t a_value,
                             std::uint16_t b_value, std::uint64_t id) {
        Transaction tx{};
        tx.fp16 = fp16;
        tx.id = id;
        tx.a.fill(a_value);
        tx.b.fill(b_value);
        tx.expected = tensorcore::test::reference_dot(tx.a, tx.b, fp16);
        return tx;
    }

    Transaction make_cancellation(bool fp16, std::uint64_t id) {
        const std::uint16_t one = fp16 ? 0x3c00U : 0x38U;
        const std::uint16_t minus_one = fp16 ? 0xbc00U : 0xb8U;
        Transaction tx = make_uniform(fp16, one, one, id);
        for (unsigned i = 0; i < 32; ++i) {
            tx.b[i] = (i & 1U) ? minus_one : one;
        }
        tx.expected = tensorcore::test::reference_dot(tx.a, tx.b, fp16);
        return tx;
    }

    Transaction make_infinity_conflict(bool fp16, std::uint64_t id) {
        const std::uint16_t one = fp16 ? 0x3c00U : 0x38U;
        const std::uint16_t pos_inf = fp16 ? 0x7c00U : 0x78U;
        const std::uint16_t neg_inf = fp16 ? 0xfc00U : 0xf8U;
        Transaction tx = make_uniform(fp16, 0, one, id);
        tx.a[0] = pos_inf;
        tx.a[1] = neg_inf;
        tx.expected = tensorcore::test::reference_dot(tx.a, tx.b, fp16);
        return tx;
    }

    [[noreturn]] void fail(const Transaction& tx, const char* field,
                           std::uint32_t actual) {
        std::ostringstream message;
        message << "transaction " << tx.id << " ("
                << (tx.fp16 ? "FP16" : "FP8 E4M3") << ") " << field
                << " mismatch: expected 0x" << std::hex << std::setw(8)
                << std::setfill('0') << tx.expected.bits << ", actual 0x"
                << std::setw(8) << actual;
        throw std::runtime_error(message.str());
    }

    void check_output() {
        if (!out_valid.read() || !out_ready.read()) {
            return;
        }
        if (expected_queue.empty()) {
            throw std::runtime_error("DUT produced an unexpected response");
        }
        const Transaction tx = expected_queue.front();
        expected_queue.pop_front();
        ++received;

        if (out_result.read().to_uint() != tx.expected.bits) {
            fail(tx, "result", out_result.read().to_uint());
        }
        if (out_invalid.read() != tx.expected.invalid ||
            out_overflow.read() != tx.expected.overflow ||
            out_underflow.read() != tx.expected.underflow ||
            out_inexact.read() != tx.expected.inexact) {
            std::ostringstream message;
            message << "transaction " << tx.id << " ("
                    << (tx.fp16 ? "FP16" : "FP8 E4M3")
                    << ") flags mismatch: expected " << tx.expected.invalid
                    << tx.expected.overflow << tx.expected.underflow
                    << tx.expected.inexact << ", actual "
                    << out_invalid.read() << out_overflow.read()
                    << out_underflow.read() << out_inexact.read();
            throw std::runtime_error(message.str());
        }
    }

    void drive(const Transaction& tx, bool valid) {
        in_valid.write(valid);
        in_format_fp16.write(tx.fp16);
        in_a.write(pack(tx.a));
        in_b.write(pack(tx.b));
    }

    void run() {
        in_valid.write(false);
        in_format_fp16.write(false);
        in_a.write(0);
        in_b.write(0);
        out_ready.write(false);
        nrst.write(false);
        wait(4, sc_core::SC_NS);
        wait(clk.negedge_event());
        nrst.write(true);
        out_ready.write(true);

        std::deque<Transaction> pending;
        pending.push_back(make_uniform(true, 0x0000, 0x0000, sent++));
        pending.push_back(make_uniform(true, 0x3c00, 0x3c00, sent++));
        pending.push_back(make_uniform(true, 0x7c00, 0x0000, sent++));
        pending.push_back(make_uniform(true, 0x0001, 0x0001, sent++));
        pending.push_back(make_uniform(true, 0x7bff, 0x7bff, sent++));
        pending.push_back(make_cancellation(true, sent++));
        pending.push_back(make_infinity_conflict(true, sent++));
        pending.push_back(make_uniform(false, 0x38, 0x38, sent++));
        pending.push_back(make_uniform(false, 0x78, 0x00, sent++));
        pending.push_back(make_uniform(false, 0x01, 0x01, sent++));
        pending.push_back(make_uniform(false, 0x77, 0x77, sent++));
        pending.push_back(make_cancellation(false, sent++));
        pending.push_back(make_infinity_conflict(false, sent++));

        constexpr std::uint64_t kRandomPerFormat = 30000;
        for (std::uint64_t i = 0; i < kRandomPerFormat; ++i) {
            pending.push_back(make_random(true, sent++));
            pending.push_back(make_random(false, sent++));
        }

        bool holding = false;
        bool reset_injected = false;
        Transaction current{};
        std::uint64_t cycles = 0;
        while (!pending.empty() || holding || !expected_queue.empty()) {
            wait(clk.negedge_event());
            if (!reset_injected && cycles >= 1000) {
                if (holding) {
                    pending.push_front(current);
                    holding = false;
                }
                in_valid.write(false);
                out_ready.write(false);
                nrst.write(false);
                expected_queue.clear();
                wait(clk.posedge_event());
                wait(clk.negedge_event());
                nrst.write(true);
                reset_injected = true;
                ++cycles;
                continue;
            }
            out_ready.write((random() & 7U) != 0U);

            if (!holding && !pending.empty() && (random() & 3U) != 0U) {
                current = pending.front();
                pending.pop_front();
                holding = true;
            }
            if (holding) {
                drive(current, true);
            } else {
                in_valid.write(false);
            }

            wait(clk.posedge_event());
            const bool accepted = holding && in_ready.read();
            check_output();
            if (accepted) {
                expected_queue.push_back(current);
                holding = false;
            }
            wait(sc_core::SC_ZERO_TIME);
            wait(sc_core::SC_ZERO_TIME);
            if (++cycles > 500000) {
                throw std::runtime_error("testbench timeout");
            }
        }

        in_valid.write(false);
        out_ready.write(true);
        std::cout << "PASS: " << received
                  << " fused dot products, seed=0x4e4e50545f444f54\n";
        sc_core::sc_stop();
    }

    SC_CTOR(Testbench) { SC_THREAD(run); }
};

}  // namespace

int sc_main(int, char**) {
    sc_core::sc_clock clk("clk", 2, sc_core::SC_NS);
    sc_core::sc_signal<bool> nrst;
    sc_core::sc_signal<bool> in_valid;
    sc_core::sc_signal<bool> in_ready;
    sc_core::sc_signal<bool> in_format_fp16;
    sc_core::sc_signal<sc_dt::sc_biguint<kInputVectorWidth>> in_a;
    sc_core::sc_signal<sc_dt::sc_biguint<kInputVectorWidth>> in_b;
    sc_core::sc_signal<bool> out_valid;
    sc_core::sc_signal<bool> out_ready;
    sc_core::sc_signal<sc_dt::sc_uint<32>> out_result;
    sc_core::sc_signal<bool> out_invalid;
    sc_core::sc_signal<bool> out_overflow;
    sc_core::sc_signal<bool> out_underflow;
    sc_core::sc_signal<bool> out_inexact;

    tensorcore::TensorDot dut("dut");
    dut.clk(clk);
    dut.nrst(nrst);
    dut.in_valid(in_valid);
    dut.in_ready(in_ready);
    dut.in_format_fp16(in_format_fp16);
    dut.in_a(in_a);
    dut.in_b(in_b);
    dut.out_valid(out_valid);
    dut.out_ready(out_ready);
    dut.out_result(out_result);
    dut.out_invalid(out_invalid);
    dut.out_overflow(out_overflow);
    dut.out_underflow(out_underflow);
    dut.out_inexact(out_inexact);

    Testbench tb("tb");
    tb.clk(clk);
    tb.nrst(nrst);
    tb.in_valid(in_valid);
    tb.in_ready(in_ready);
    tb.in_format_fp16(in_format_fp16);
    tb.in_a(in_a);
    tb.in_b(in_b);
    tb.out_valid(out_valid);
    tb.out_ready(out_ready);
    tb.out_result(out_result);
    tb.out_invalid(out_invalid);
    tb.out_overflow(out_overflow);
    tb.out_underflow(out_underflow);
    tb.out_inexact(out_inexact);

    try {
        sc_core::sc_start();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
