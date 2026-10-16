# Fused floating-point Tensor dot product

This repository contains a synthesizable-style SystemC model of the fused
many-term dot product used by the Intel NNP-T tensor datapath. One request
computes 32 products and rounds their exact sum once to IEEE-754 binary32:

```text
result = round_to_fp32(sum(a[i] * b[i]), i = 0..31)
```

The input format is selected per request. Supported formats are IEEE binary16
and an IEEE-style FP8 E4M3 with exponent bias 7, subnormals, infinities, and
NaNs. E4M3FN semantics are intentionally not used.

## Architecture

Each operand is decoded into an integer significand and a binary scale. The
32 full-precision significand products are shifted into a shared 88-bit fixed
point accumulator whose least-significant bit represents 2^-48. A balanced
32/16/8/4/2/1 signed adder tree produces an exact sum. Leading-one detection,
normalization, and round-to-nearest-ties-to-even are performed only after the
tree, so no product or partial sum is rounded.

The datapath has seven registered stages from acceptance to response. With no
backpressure it accepts one request per cycle. If the response is blocked,
the whole pipeline stalls and keeps every payload stable.

The request ports are:

- `in_valid`, `in_ready`: request handshake.
- `in_format_fp16`: 1 selects FP16; 0 selects FP8 E4M3.
- `in_a`, `in_b`: 512-bit vectors containing 32 16-bit slots. FP8 values use
  bits 7:0 of each slot.

The response uses `out_valid`, `out_ready`, a 32-bit FP32 result, and IEEE-style
`invalid`, `overflow`, `underflow`, and `inexact` status flags. NaNs are
canonicalized to `0x7fc00000`.

## Build and test

The bootstrap script installs CMake and SystemC 3.0.2 into user-writable
directories adjacent to the checkout, then builds and runs the tests:

```bash
./scripts/bootstrap.sh
```

For an existing SystemC installation:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/systemc
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The testbench uses an independent signed 128-bit integer reference model. That
width exactly contains the full FP16 product range and all 32 accumulated
terms. It checks directed corner cases and 60,000 random FP16/FP8 vectors with
random input bubbles, response backpressure, and an in-flight reset. The random
seed is fixed and printed on success, so a failure is reproducible.

## Source layout

- `include/fp_decode.h`: bit-exact FP16 and FP8 decoding.
- `include/fp32_pack.h`: final FP32 normalization and rounding.
- `include/tensor_dot.h`: ports, exact accumulator, reduction tree, and control.
- `tests/reference_model.cpp`: independent wide-integer golden model.
- `tests/sc_main.cpp`: SystemC driver, scoreboard, and randomized regression.
