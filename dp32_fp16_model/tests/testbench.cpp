#include "dp32.hpp"
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
std::size_t checks = 0;
void require(bool condition, const std::string& name) {
    ++checks;
    if (!condition) throw std::runtime_error(name);
}
std::uint32_t hex32(const std::string& s) {
    return static_cast<std::uint32_t>(std::stoul(s, nullptr, 16));
}
std::uint64_t modulo_sum(const std::vector<std::uint64_t>& v) {
    std::uint64_t sum = 0;
    for (auto x : v) sum = (sum + x) & dp32::kMask;
    return sum;
}
// Independent binary16 -> binary32 widening: all finite FP16 values are exact
// in float, so ldexp is sufficient here. This checks EVERY FP16 input encoding.
std::uint32_t widened_half(std::uint16_t h, bool ftz) {
    unsigned e = (h >> 10) & 31, m = h & 1023;
    if (e == 31) return m ? 0x7fc00000u : ((h & 0x8000) ? 0xff800000u : 0x7f800000u);
    if (e == 0 && (ftz || m == 0)) return 0; // canonical exact +0
    float f = e == 0 ? std::ldexp(static_cast<float>(m), -24)
                     : std::ldexp(static_cast<float>(1024 + m), static_cast<int>(e) - 25);
    if (h & 0x8000) f = -f;
    return dp32::float_to_bits(f);
}
void directed() {
    dp32::HalfVector a{}, b{};
    a.fill(0x3c00); b.fill(0x3c00);
    require(dp32::dot(a, b).bits == 0x42000000u, "32 ones = 32");
    a.fill(0xbc00);
    require(dp32::dot(a, b).bits == 0xc2000000u, "negative 32 ones");
    a.fill(0); b.fill(0); a[0] = b[0] = 0x0c00;
    require(dp32::dot(a, b, 0x3f800000).bits == 0x3f800000, "RNE tie even");
    require(dp32::dot(a, b, 0x3f800001).bits == 0x3f800002, "RNE tie odd");
    a[0] = b[0] = 1;
    require(dp32::dot(a, b).bits == dp32::float_to_bits(std::ldexp(1.0f, -48)), "subnormal x subnormal");
    require(dp32::dot(a, b, 0, {true}).bits == 0, "FP16 input FTZ");
    a.fill(0); b.fill(0);
    a[0] = 0x7bff; a[1] = 0xfbff; a[2] = 1;
    b[0] = b[1] = 0x7bff; b[2] = 0x3c00;
    auto r = dp32::dot(a, b);
    require(r.bits == 0 && r.alignment_truncated, "cancellation reveals alignment loss");
    require(dp32::dot(a, b, 1).bits == 1, "late C minimum FP32 subnormal retained");
}
void exhaustive_decode() {
    dp32::HalfVector a{}, b{}; b[0] = 0x3c00;
    for (bool ftz : {false, true}) {
        for (unsigned h = 0; h < 65536; ++h) {
            a[0] = static_cast<std::uint16_t>(h);
            auto r = dp32::dot(a, b, 0, {ftz});
            require(r.bits == widened_half(a[0], ftz), "exhaustive FP16 widening: " + std::to_string(h));
            bool snan = (h & 0x7c00) == 0x7c00 && (h & 1023) && !(h & 512);
            require(r.invalid == snan, "exhaustive sNaN flag");
        }
    }
    std::cout << "[PASS] All 65536 FP16 encodings, gradual + FTZ\n";
}
void pipeline_test() {
    dp32::Pipeline9 pipe;
    struct Due { unsigned cycle; dp32::Response response; };
    std::deque<Due> expected;
    for (unsigned cycle = 0; cycle < 260; ++cycle) {
        std::optional<dp32::Request> in;
        if (cycle < 240 && cycle % 7 != 3) {
            dp32::Request req;
            req.a.fill(0x3c00); req.b.fill(0x4000);
            req.c = dp32::float_to_bits(static_cast<float>(cycle)); req.tag = cycle;
            in = req;
            expected.push_back({cycle + 9, {dp32::dot(req.a, req.b, req.c), req.tag}});
        }
        auto out = pipe.tick(in);
        bool due = !expected.empty() && expected.front().cycle == cycle;
        require(out.has_value() == due, "pipeline valid/latency");
        if (due) {
            require(out->tag == expected.front().response.tag, "pipeline tag order");
            require(out->result.bits == expected.front().response.result.bits, "pipeline data");
            expected.pop_front();
        }
    }
    require(expected.empty(), "pipeline drained");
    dp32::Request req; pipe.tick(req); pipe.reset();
    for (unsigned i = 0; i < 12; ++i) require(!pipe.tick(), "reset flushes pipeline");
    std::cout << "[PASS] 9-cycle latency, II=1, bubbles, tags, reset\n";
}
void vectors(const std::string& path) {
    std::ifstream file(path);
    require(static_cast<bool>(file), "open vectors: " + path);
    std::string line;
    std::size_t n = 0, exact_differences = 0, truncated = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        std::string name, cs, ms, es, word;
        unsigned ftz, invalid, lost, inexact, flushed, overflow;
        require(static_cast<bool>(row >> name >> ftz >> cs >> ms >> es >> invalid >> lost >> inexact >> flushed >> overflow), "parse header");
        dp32::HalfVector a{}, b{};
        for (auto& h : a) { require(static_cast<bool>(row >> word), "parse a"); h = static_cast<std::uint16_t>(hex32(word)); }
        for (auto& h : b) { require(static_cast<bool>(row >> word), "parse b"); h = static_cast<std::uint16_t>(hex32(word)); }
        require(!(row >> word), "no trailing vector fields");
        dp32::Trace trace;
        auto r = dp32::dot(a, b, hex32(cs), {ftz != 0}, &trace);
        auto expected = hex32(ms);
        if (r.bits != expected) {
            std::cerr << name << " ftz=" << ftz << " got=0x" << std::hex << r.bits
                      << " expected=0x" << expected << std::dec << '\n';
        }
        require(r.bits == expected, "Fraction model golden: " + name);
        require(r.invalid == (invalid != 0), "invalid flag: " + name);
        require(r.alignment_truncated == (lost != 0), "alignment flag: " + name);
        require(r.final_round_inexact == (inexact != 0), "rounding flag: " + name);
        require(r.input_flushed == (flushed != 0), "FTZ flag: " + name);
        require(r.overflow == (overflow != 0), "overflow flag: " + name);
        if (trace.finite_path) {
            require(trace.csa_levels.size() == 9, "CSA level count");
            auto sum = modulo_sum(trace.csa_levels.front());
            for (const auto& level : trace.csa_levels)
                require(modulo_sum(level) == sum, "CSA modular sum invariant");
            std::int64_t scalar_sum = 0;
            for (const auto& p : trace.products) {
                require(p.aligned_magnitude < (std::uint64_t{1} << 31), "37-bit headroom bound");
                scalar_sum += p.negative ? -static_cast<std::int64_t>(p.aligned_magnitude)
                                        : static_cast<std::int64_t>(p.aligned_magnitude);
            }
            require(scalar_sum == trace.reduced_integer, "CSA signed sum equals scalar sum");
            if (scalar_sum != 0) {
                auto magnitude = static_cast<std::uint64_t>(scalar_sum < 0 ? -scalar_sum : scalar_sum);
                int shift = trace.common_lsb_exponent - trace.normalized_lsb_exponent;
                require(shift >= 0 && shift < 37, "normalization shift range");
                require((trace.normalized_significand >> shift) == magnitude, "normalization preserves value");
                require((trace.normalized_significand >> 36) == 1, "normalized leading bit");
            }
        }
        ++n; exact_differences += r.bits != hex32(es); truncated += r.alignment_truncated;
    }
    require(n > 0, "nonempty vectors");
    require(exact_differences > 0, "corpus exercises approximate reduction");
    std::cout << "[PASS] " << n << " independent Fraction golden vectors\n"
              << "[INFO] Alignment-truncated cases: " << truncated
              << "; bit differences vs exact-dot RNE: " << exact_differences << '\n';
}
} // namespace
int main(int argc, char** argv) {
    try {
        directed(); exhaustive_decode(); pipeline_test();
        vectors(argc > 1 ? argv[1] : "tests/golden_vectors.txt");
        std::cout << "ALL TESTS PASSED; checks=" << checks << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << '\n'; return 1;
    }
}
