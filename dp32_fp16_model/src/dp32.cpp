#include "dp32.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace dp32 {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Display helpers require IEEE binary32 float");
struct Number {
    bool sign = false, inf = false, nan = false, snan = false;
    std::uint32_t sig = 0;
    int exp2 = 0;
};
unsigned bit_length(std::uint64_t v) {
    unsigned n = 0;
    while (v) { ++n; v >>= 1; }
    return n;
}
Number decode16(std::uint16_t h, Config cfg, bool& flushed) {
    Number n;
    n.sign = (h >> 15) != 0;
    unsigned e = (h >> 10) & 31, f = h & 1023;
    if (e == 31) {
        n.inf = f == 0; n.nan = f != 0; n.snan = n.nan && !(f & 512);
    } else if (e == 0) {
        if (f && cfg.flush_fp16_subnormals) flushed = true;
        n.sig = cfg.flush_fp16_subnormals ? 0 : f;
        n.exp2 = -24;
    } else {
        n.sig = 1024 + f; n.exp2 = static_cast<int>(e) - 25;
    }
    return n;
}
Number decode32(std::uint32_t f) {
    Number n;
    n.sign = (f >> 31) != 0;
    unsigned e = (f >> 23) & 255, m = f & 0x7fffff;
    if (e == 255) {
        n.inf = m == 0; n.nan = m != 0; n.snan = n.nan && !(m & 0x400000);
    } else if (e == 0) {
        n.sig = m; n.exp2 = -149;
    } else {
        n.sig = 0x800000 + m; n.exp2 = static_cast<int>(e) - 150;
    }
    return n;
}

// Simulation-only exact representation for the LAST floating-point adder.
// Unit is 2^-149, five limbs cover binary32 C and the finite FP16 reduction.
// It avoids host FP/double rounding. It is NOT a 320-bit hardware accumulator:
// truncation has ALREADY occurred in the 37-bit dot-product datapath.
struct Wide {
    std::array<std::uint64_t, 5> w{};
    static Wide shifted(std::uint64_t m, int shift) {
        Wide x;
        if (!m) return x;
        if (shift < 0 || shift + static_cast<int>(bit_length(m)) > 320)
            throw std::logic_error("wide-adder range");
        unsigned limb = static_cast<unsigned>(shift) / 64;
        unsigned off = static_cast<unsigned>(shift) % 64;
        x.w[limb] = m << off;
        if (off && limb + 1 < x.w.size()) x.w[limb + 1] = m >> (64 - off);
        return x;
    }
    int compare(const Wide& b) const {
        for (int i = 4; i >= 0; --i) {
            if (w[i] < b.w[i]) return -1;
            if (w[i] > b.w[i]) return 1;
        }
        return 0;
    }
    Wide add(const Wide& b) const {
        Wide out; std::uint64_t carry = 0;
        for (unsigned i = 0; i < 5; ++i) {
            auto s = w[i] + b.w[i];
            auto c1 = static_cast<std::uint64_t>(s < w[i]);
            auto t = s + carry;
            auto c2 = static_cast<std::uint64_t>(t < s);
            out.w[i] = t; carry = c1 | c2;
        }
        if (carry) throw std::logic_error("wide-adder overflow");
        return out;
    }
    Wide subtract(const Wide& b) const { // precondition: *this >= b
        Wide out; std::uint64_t borrow = 0;
        for (unsigned i = 0; i < 5; ++i) {
            auto s = w[i] - b.w[i];
            auto b1 = static_cast<std::uint64_t>(w[i] < b.w[i]);
            auto t = s - borrow;
            auto b2 = static_cast<std::uint64_t>(s < borrow);
            out.w[i] = t; borrow = b1 | b2;
        }
        if (borrow) throw std::logic_error("wide-adder underflow");
        return out;
    }
    bool bit(unsigned p) const { return ((w[p / 64] >> (p % 64)) & 1) != 0; }
    bool any_below(unsigned end) const { // [0, end)
        for (unsigned i = 0; i < end; ++i) if (bit(i)) return true;
        return false;
    }
    int top() const {
        for (int i = 4; i >= 0; --i)
            if (w[i]) return i * 64 + static_cast<int>(bit_length(w[i])) - 1;
        return -1;
    }
};
std::uint32_t round32(const Wide& mag, bool sign, Result& r) {
    int top = mag.top();
    if (top < 0) return 0; // model convention: exact zero always +0
    std::uint32_t signbit = sign ? 0x80000000u : 0;
    if (top < 23) return signbit | static_cast<std::uint32_t>(mag.w[0]);
    unsigned shift = static_cast<unsigned>(top - 23);
    std::uint32_t sig = 0;
    for (int i = top; i >= top - 23; --i)
        sig = (sig << 1) | static_cast<unsigned>(mag.bit(static_cast<unsigned>(i)));
    if (shift) {
        bool guard = mag.bit(shift - 1), sticky = mag.any_below(shift - 1);
        r.final_round_inexact = guard || sticky;
        if (guard && (sticky || (sig & 1))) ++sig;
    }
    if (sig == 0x1000000u) { sig >>= 1; ++top; }
    int exponent = top - 149;
    if (exponent > 127) {
        r.overflow = true; r.final_round_inexact = true;
        return signbit | 0x7f800000u;
    }
    return signbit | (static_cast<std::uint32_t>(exponent + 127) << 23)
                   | (sig & 0x7fffffu);
}
std::uint32_t add_late(bool neg, std::uint64_t magnitude, int exp2, const Number& c, Result& r) {
    Wide a = Wide::shifted(magnitude, exp2 + 149);
    Wide b = Wide::shifted(c.sig, c.exp2 + 149);
    if (neg == c.sign) return round32(a.add(b), neg, r);
    int cmp = a.compare(b);
    if (cmp >= 0) return round32(a.subtract(b), neg, r);
    return round32(b.subtract(a), c.sign, r);
}
} // namespace

Result dot(const HalfVector& a, const HalfVector& b, std::uint32_t c_bits,
           Config config, Trace* trace) {
    Trace local;
    Trace& t = trace ? *trace : local;
    t = Trace{};
    Result r;
    Number c = decode32(c_bits);
    bool nan = c.nan, positive_inf = c.inf && !c.sign, negative_inf = c.inf && c.sign;
    r.invalid = c.snan;
    int maximum = std::numeric_limits<int>::min();
    for (unsigned i = 0; i < kTerms; ++i) {
        Number x = decode16(a[i], config, r.input_flushed);
        Number y = decode16(b[i], config, r.input_flushed);
        nan = nan || x.nan || y.nan;
        r.invalid = r.invalid || x.snan || y.snan;
        bool invalid_product = (x.inf && !y.sig && !y.inf && !y.nan)
                            || (y.inf && !x.sig && !x.inf && !x.nan);
        r.invalid = r.invalid || invalid_product;
        auto& p = t.products[i];
        p.negative = x.sign != y.sign;
        if (x.nan || y.nan || invalid_product) continue;
        if (x.inf || y.inf) {
            if (p.negative) negative_inf = true; else positive_inf = true;
            continue;
        }
        p.significand_product = x.sig * y.sig;
        p.exponent2 = x.exp2 + y.exp2;
        if (p.significand_product) {
            p.normalized_exponent = p.exponent2
                + static_cast<int>(bit_length(p.significand_product)) - 1;
            if (p.normalized_exponent > maximum) maximum = p.normalized_exponent;
            t.all_products_zero = false;
        }
    }
    r.invalid = r.invalid || (positive_inf && negative_inf);
    if (nan || r.invalid) { r.bits = 0x7fc00000u; return r; }
    if (positive_inf || negative_inf) {
        r.bits = negative_inf ? 0xff800000u : 0x7f800000u;
        return r;
    }
    t.finite_path = true;
    t.max_product_exponent = t.all_products_zero ? 0 : maximum;
    t.common_lsb_exponent = t.all_products_zero ? 0 : maximum - static_cast<int>(kFractionBits);
    std::vector<std::uint64_t> operands;
    for (auto& p : t.products) {
        if (p.significand_product) {
            p.right_shift = t.common_lsb_exponent - p.exponent2;
            std::uint64_t m = p.significand_product;
            if (p.right_shift >= 64) { p.truncated = m != 0; m = 0; }
            else if (p.right_shift > 0) {
                auto mask = (std::uint64_t{1} << p.right_shift) - 1;
                p.truncated = (m & mask) != 0; m >>= p.right_shift;
            } else if (p.right_shift < 0) {
                unsigned left = static_cast<unsigned>(-p.right_shift);
                if (left >= 64) throw std::logic_error("alignment shift range");
                m <<= left;
            }
            p.aligned_magnitude = m;
            p.twos_complement = (p.negative ? std::uint64_t{0} - m : m) & kMask;
            r.alignment_truncated = r.alignment_truncated || p.truncated;
        }
        operands.push_back(p.twos_complement);
    }
    if (trace) t.csa_levels.push_back(operands);
    while (operands.size() > 2) {
        std::vector<std::uint64_t> next;
        std::size_t i = 0;
        for (; i + 2 < operands.size(); i += 3) {
            auto x = operands[i], y = operands[i + 1], z = operands[i + 2];
            next.push_back((x ^ y ^ z) & kMask);
            next.push_back((((x & y) | (x & z) | (y & z)) << 1) & kMask);
        }
        for (; i < operands.size(); ++i) next.push_back(operands[i]);
        operands = std::move(next);
        if (trace) t.csa_levels.push_back(operands);
    }
    t.carry_save_sum = operands[0]; t.carry_save_carry = operands[1];
    auto word = (operands[0] + operands[1]) & kMask;
    t.reduced_integer = (word & (std::uint64_t{1} << 36))
        ? static_cast<std::int64_t>(word) - (std::int64_t{1} << 37)
        : static_cast<std::int64_t>(word);
    auto magnitude = static_cast<std::uint64_t>(t.reduced_integer < 0
                         ? -t.reduced_integer : t.reduced_integer);
    t.normalized_significand_bits = bit_length(magnitude);
    if (magnitude) {
        t.normalized_exponent = t.common_lsb_exponent
                               + static_cast<int>(bit_length(magnitude)) - 1;
        unsigned left = kWidth - bit_length(magnitude);
        t.normalized_significand = magnitude << left;
        t.normalized_lsb_exponent = t.common_lsb_exponent - static_cast<int>(left);
    }
    r.bits = add_late(t.reduced_integer < 0, t.normalized_significand,
                     t.normalized_lsb_exponent, c, r);
    return r;
}
float bits_to_float(std::uint32_t bits) {
    float f; std::memcpy(&f, &bits, sizeof f); return f;
}
std::uint32_t float_to_bits(float value) {
    std::uint32_t bits; std::memcpy(&bits, &value, sizeof bits); return bits;
}
std::optional<Response> Pipeline9::tick(std::optional<Request> request) {
    auto out = slots_.back();
    for (std::size_t i = slots_.size() - 1; i > 0; --i) slots_[i] = slots_[i - 1];
    slots_[0].reset();
    if (request) slots_[0] = Response{dot(request->a, request->b, request->c, config_), request->tag};
    return out;
}
void Pipeline9::reset() { slots_.fill(std::nullopt); }
} // namespace dp32
