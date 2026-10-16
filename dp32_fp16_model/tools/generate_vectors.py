#!/usr/bin/env python3
"""Independent scalar/Fraction oracle; does NOT implement the C++ CSA tree.

Writes model goldens AND exact-dot goldens. Bit mismatches to exact-dot are
recorded as numerical differences, not functional test failures.
Only Python standard library is required.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import random


def pow2(e):
    return F(1 << e) if e >= 0 else F(1, 1 << -e)


def decode(bits, width, ftz=False):
    fracbits, expbits, bias = (10, 5, 15) if width == 16 else (23, 8, 127)
    sign = -1 if bits >> (width - 1) else 1
    exp = (bits >> fracbits) & ((1 << expbits) - 1)
    frac = bits & ((1 << fracbits) - 1)
    if exp == (1 << expbits) - 1:
        return ('nan' if frac else 'inf', sign,
                bool(frac) and not bool(frac & (1 << (fracbits - 1))))
    if exp == 0:
        value = F(0) if ftz else frac * pow2(1 - bias - fracbits)
    else:
        value = ((1 << fracbits) + frac) * pow2(exp - bias - fracbits)
    return ('finite', sign * value, False)


def floor_log2(x):
    assert x > 0
    e = x.numerator.bit_length() - x.denominator.bit_length()
    return e - (x < pow2(e))


def rne_integer(x):
    q, rem = divmod(x.numerator, x.denominator)
    twice = rem * 2
    return q + (twice > x.denominator or (twice == x.denominator and q % 2))


def fp32(x):
    if not x:
        return 0
    sign = 0x80000000 if x < 0 else 0
    x = abs(x)
    e = floor_log2(x)
    if e < -126:
        m = rne_integer(x / pow2(-149))
        return sign | m
    m = rne_integer(x / pow2(e - 23))
    if m == 1 << 24:
        m >>= 1
        e += 1
    if e > 127:
        return sign | 0x7f800000
    return sign | ((e + 127) << 23) | (m - (1 << 23))


def evaluate(a, b, c_bits, ftz):
    c = decode(c_bits, 32)
    invalid, nan = c[2], c[0] == 'nan'
    infinities = {c[1]} if c[0] == 'inf' else set()
    products = []
    flushed = False
    for ha, hb in zip(a, b):
        x, y = decode(ha, 16, ftz), decode(hb, 16, ftz)
        flushed |= ftz and any((h & 0x7c00) == 0 and (h & 1023) for h in (ha, hb))
        nan |= x[0] == 'nan' or y[0] == 'nan'
        invalid |= x[2] or y[2]
        if x[0] == 'nan' or y[0] == 'nan':
            continue
        if x[0] == 'inf' or y[0] == 'inf':
            if x[0] == 'finite' and x[1] == 0 or y[0] == 'finite' and y[1] == 0:
                invalid = True
            else:
                sx = -1 if ha & 0x8000 else 1
                sy = -1 if hb & 0x8000 else 1
                infinities.add(sx * sy)
        else:
            products.append(x[1] * y[1])
    invalid |= len(infinities) == 2
    if nan or invalid:
        return 0x7fc00000, 0x7fc00000, int(invalid), 0, 0, int(flushed), 0
    if infinities:
        bits = 0x7f800000 if 1 in infinities else 0xff800000
        return bits, bits, 0, 0, 0, int(flushed), 0
    nonzero = [abs(p) for p in products if p]
    grid_exp = max(map(floor_log2, nonzero)) - 30 if nonzero else 0
    grid = pow2(grid_exp)
    quantized = []
    for p in products:
        mag = abs(p) / grid
        q = mag.numerator // mag.denominator # magnitude truncation before sign
        quantized.append((-q if p < 0 else q) * grid)
    lost = any(p != q for p, q in zip(products, quantized))
    reduced = sum(quantized, F(0))
    value = reduced + c[1]
    bits = fp32(value)
    final_value = decode(bits, 32)
    inexact = final_value[0] != 'finite' or final_value[1] != value
    overflow = final_value[0] == 'inf'
    exact = fp32(sum(products, F(0)) + c[1])
    return bits, exact, 0, int(lost), int(inexact), int(flushed), int(overflow)


def corpus(seed, count):
    rng = random.Random(seed)
    def lane(a=0, b=0, c=0):
        return [a] + [0] * 31, [b] + [0] * 31, c
    yield 'zero', *lane()
    yield 'negative_zero', *lane(0x8000, 0x3c00, 0x80000000)
    yield 'ones', [0x3c00] * 32, [0x3c00] * 32, 0
    yield 'negative_ones', [0xbc00] * 32, [0x3c00] * 32, 0
    yield 'max_positive', [0x7bff] * 32, [0x7bff] * 32, 0
    yield 'max_negative', [0xfbff] * 32, [0x7bff] * 32, 0
    yield 'min_subnormal_product', *lane(1, 1)
    yield 'min_normal_product', *lane(0x0400, 0x0400)
    yield 'tie_even', *lane(0x0c00, 0x0c00, 0x3f800000)
    yield 'tie_odd', *lane(0x0c00, 0x0c00, 0x3f800001)
    yield 'c_min_subnormal', *lane(c=1)
    yield 'c_max_subnormal', *lane(c=0x007fffff)
    yield 'c_max_finite', *lane(0x7bff, 0x7bff, 0x7f7fffff)
    yield 'late_c_preserves_dot', [0x3c00] * 32, [0x3c00] * 32, 0x4c800000
    a, b, c = lane()
    a[:3], b[:3] = [0x7bff, 0xfbff, 1], [0x7bff, 0x7bff, 0x3c00]
    yield 'cancel_exposes_truncation', a, b, c
    yield 'cancel_leaves_fp32_subnormal', a, b, 1
    a, b, _ = lane()
    a[:2], b[:2] = [0xbc00, 1], [0x3c00, 0x3c00]
    yield 'negative_truncate_before_twos', a, b, 0x3f800000
    for label, x, y, c in [
        ('inf', 0x7c00, 0x3c00, 0), ('negative_inf', 0xfc00, 0x3c00, 0),
        ('inf_times_zero', 0x7c00, 0, 0), ('qnan', 0x7e01, 0x3c00, 0),
        ('snan', 0x7c01, 0x3c00, 0), ('opposite_inf_c', 0x7c00, 0x3c00, 0xff800000),
        ('c_qnan', 0x3c00, 0x3c00, 0x7fc00001),
        ('c_snan', 0x3c00, 0x3c00, 0x7f800001),
        ('ftz_then_inf_zero', 1, 0x7c00, 0)]:
        yield label, *lane(x, y, c)
    a, b, c = lane()
    a[:2], b[:2] = [0x7c00, 0xfc00], [0x3c00, 0x3c00]
    yield 'opposite_infinite_products', a, b, c
    def finite_half():
        return rng.getrandbits(1) << 15 | rng.randrange(31) << 10 | rng.randrange(1024)
    for i in range(count):
        kind = i % 5
        if kind == 0:
            a, b = [finite_half() for _ in range(32)], [finite_half() for _ in range(32)]
        elif kind == 1:
            ea, eb = rng.randrange(1, 29), rng.randrange(1, 29)
            def clustered(e):
                return rng.getrandbits(1) << 15 | (e + rng.randrange(3)) << 10 | rng.randrange(1024)
            a, b = [clustered(ea) for _ in range(32)], [clustered(eb) for _ in range(32)]
        elif kind == 2:
            a, b = [0] * 32, [0] * 32
            for j in range(0, 30, 2):
                x, y = finite_half(), finite_half()
                a[j:j+2], b[j:j+2] = [x, x ^ 0x8000], [y, y]
            a[30:], b[30:] = [finite_half(), finite_half()], [finite_half(), finite_half()]
        elif kind == 3:
            a, b = [rng.getrandbits(16) for _ in range(32)], [rng.getrandbits(16) for _ in range(32)]
        else:
            a, b = [0] * 32, [0] * 32
            for j in rng.sample(range(32), rng.randrange(1, 9)):
                a[j], b[j] = finite_half(), finite_half()
        c = rng.choice([0, 1, 0x80000001, 0x3f800000, 0xbf800000, rng.getrandbits(32)])
        yield f'random_{kind}_{i}', a, b, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'tests/golden_vectors.txt')
    ap.add_argument('--count', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=20261016)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total, differences = 0, 0
    with args.output.open('w') as out:
        out.write('# name ftz c model exact invalid lost inexact flushed overflow a[32] b[32]; encodings hex\n')
        for name, a, b, c in corpus(args.seed, args.count):
            for ftz in (False, True):
                model, exact, invalid, lost, inexact, flushed, overflow = evaluate(a, b, c, ftz)
                fields = [name, str(int(ftz)), f'{c:08x}', f'{model:08x}', f'{exact:08x}',
                          *map(str, (invalid, lost, inexact, flushed, overflow)),
                          *(f'{h:04x}' for h in a), *(f'{h:04x}' for h in b)]
                out.write(' '.join(fields) + '\n')
                total += 1
                differences += model != exact
    print(f'Generated {total} vectors; model vs exact-dot bit differences: {differences}; seed={args.seed}')


if __name__ == '__main__':
    main()
