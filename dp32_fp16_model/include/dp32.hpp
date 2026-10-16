#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace dp32 {
constexpr unsigned kTerms = 32;
constexpr unsigned kWidth = 37;
constexpr unsigned kFractionBits = 30; // 1 sign + 5 growth + 31 significand bits
constexpr std::uint64_t kMask = (std::uint64_t{1} << kWidth) - 1;
using HalfVector = std::array<std::uint16_t, kTerms>;

struct Config {
    // IEEE binary16 subnormals are supported by default (extension to the paper).
    // With true, subnormal FP16 inputs become signed zero before multiplication.
    bool flush_fp16_subnormals = false;
};
struct ProductTrace {
    bool negative = false;
    std::uint32_t significand_product = 0;
    int exponent2 = 0; // exact product = sign * significand_product * 2^exponent2
    int normalized_exponent = 0;
    int right_shift = 0; // negative means left shift
    std::uint64_t aligned_magnitude = 0;
    std::uint64_t twos_complement = 0; // low 37 bits
    bool truncated = false;
};
struct Trace {
    bool finite_path = false;
    bool all_products_zero = true;
    int max_product_exponent = 0;
    int common_lsb_exponent = 0;
    std::array<ProductTrace, kTerms> products{};
    // First vector has 32 operands; each following vector is one 3:2 CSA level.
    std::vector<std::vector<std::uint64_t>> csa_levels;
    std::uint64_t carry_save_sum = 0;
    std::uint64_t carry_save_carry = 0; // already shifted, add directly to sum
    std::int64_t reduced_integer = 0;
    unsigned normalized_significand_bits = 0;
    int normalized_exponent = 0;
    std::uint64_t normalized_significand = 0; // sign separated, left-justified to 37 bits
    int normalized_lsb_exponent = 0;
};
struct Result {
    std::uint32_t bits = 0;
    bool invalid = false;
    bool overflow = false;
    bool alignment_truncated = false;
    bool final_round_inexact = false;
    bool input_flushed = false;
};

// d = reduce_37(a[0]*b[0], ..., a[31]*b[31]) + c, final RNE to FP32.
// a/b/c are raw IEEE encodings, NOT numerically converted integer values.
Result dot(const HalfVector& a, const HalfVector& b, std::uint32_t c = 0,
           Config config = {}, Trace* trace = nullptr);
float bits_to_float(std::uint32_t bits);
std::uint32_t float_to_bits(float value);

struct Request {
    HalfVector a{}, b{};
    std::uint32_t c = 0;
    std::uint64_t tag = 0;
};
struct Response { Result result; std::uint64_t tag = 0; };

// Behavioral latency adapter, not a gate/stage-accurate RTL implementation.
// An input accepted on tick n emerges on tick n+9. II=1, no backpressure.
class Pipeline9 {
public:
    explicit Pipeline9(Config config = {}) : config_(config) {}
    std::optional<Response> tick(std::optional<Request> request = std::nullopt);
    void reset();
private:
    Config config_;
    std::array<std::optional<Response>, 9> slots_{};
};
} // namespace dp32
