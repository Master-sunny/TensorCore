#include "dp32.hpp"
#include <iomanip>
#include <iostream>

int main() {
    dp32::HalfVector a{}, b{};
    a.fill(0x3c00); // FP16 +1.0, NOT the C++ integer 1
    b.fill(0x4000); // FP16 +2.0
    dp32::Trace trace;
    const auto result = dp32::dot(a, b, dp32::float_to_bits(3.0f), {}, &trace);
    std::cout << "D = sum(32 * (1 * 2)) + 3 = " << dp32::bits_to_float(result.bits)
              << "\nFP32 bits = 0x" << std::hex << std::setw(8) << std::setfill('0') << result.bits
              << std::dec << "\nEmax = " << trace.max_product_exponent
              << ", common LSB exponent = " << trace.common_lsb_exponent
              << "\nCSA operand counts:";
    for (const auto& level : trace.csa_levels) std::cout << ' ' << level.size();
    std::cout << "\nReduced signed integer = " << trace.reduced_integer
              << "\nAlignment truncated = " << result.alignment_truncated << '\n';
    return result.bits == dp32::float_to_bits(67.0f) ? 0 : 1;
}
