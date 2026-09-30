"""The device's host math, bit for bit: what `mx_host.h` computes on the RISC-V core, in numpy.

The golden has to reproduce the HOST stages exactly, not just the mesh, because a host value that is
one float32 ulp off flips an FP8 code whenever it sits on an E4M3 rounding tie. numpy's float32
`np.exp` is not correctly rounded (vectorized kernel, ~40% of values 1 ulp off) while the device calls
newlib's `expf`; on ~1.4M softmax values per 64-token model one of them landed on a tie (layer 14,
head 5, row 41 of llama_model_m64: P code 0x0d on the device vs 0x0c golden).

    expf(x)               newlib __ieee754_expf (fdlibm) as compiled in the riscv64 libm.a: the object
                          code's fmadd/fnmsub are fused here too (fma32)
    rmsnorm(h, w, eps)    mx_rmsnorm: sequential double sum of squares, float32 1/sqrtf, (h*w)*inv
    rope(x, cos, sin)     mx_rope_at: fma(x, cos, rot*sin), the contraction the compiler emits
    silu(x)               mx_silu: x / (1 + expf(-x))
    softmax_causal(s)     mx_softmax_causal on already-scaled scores: row max, expf(v - max),
                          SEQUENTIAL sum, multiply by 1/sum

Drop-in for `app.capture_llama_layer.rmsnorm` / `rope` / `silu` / `softmax_causal` (same signatures), which stay
the fp32 REFERENCE implementations. numpy's rmsnorm (float32 pairwise mean, (h*inv)*w) is off by an ulp
somewhere in essentially every 4096-element block; at 64 tokens that flipped one code in layer 20. Checked against the device on spike over ~4M values (a probe compiled with
the kernels' flags: expf on a dense grid over [-104, 88] and 1M random inputs, mx_silu on [-40, 40],
mx_softmax_causal on 256 random 64x64 score matrices, mx_rmsnorm on 1024 rows of D=2048 with
activations over 2^-10..2^8, mx_rope_at on 256 random 64x64 heads): every value identical.
"""
from __future__ import annotations

import numpy as np

__all__ = ["expf", "rmsnorm", "rope", "silu", "softmax_causal", "fma32"]

F32, F64, U32 = np.float32, np.float64, np.uint32


def _c(bits: int) -> np.float32:
    return np.array([bits], U32).view(F32)[0]


# ef_exp.o's constants, from its .srodata.cst4
INVLN2, LN2HI, LN2LO = _c(0x3fb8aa3b), _c(0x3f317180), _c(0x3717f7d1)
P1, P2, P3, P4, P5 = _c(0x3e2aaaab), _c(0xbb360b61), _c(0x388ab355), _c(0xb5ddea0e), _c(0x3331bb4c)
ONE, TWO, TWOM100 = F32(1.0), F32(2.0), _c(0x0d800000)


def fma32(a, b, c) -> np.ndarray:
    """float32 fused multiply-add, exact: a*b is exact in float64 and TwoSum recovers the add's
    error, which decides the one case the float64 sum cannot -- landing on a float32 midpoint."""
    a, b, c = (np.asarray(v, F32).astype(F64) for v in (a, b, c))
    p = a * b
    s = p + c
    bb = s - p
    e = (p - (s - bb)) + (c - bb)                              # s + e == p + c exactly
    r = s.astype(F32)
    rd = r.astype(F64)
    other = np.nextafter(r, np.where(s > rd, F32(np.inf), F32(-np.inf)))
    tie = (rd != s) & (s == (rd + other.astype(F64)) / 2) & (e != 0)
    return np.where(tie, np.where(e > 0, np.maximum(r, other), np.minimum(r, other)), r).astype(F32)


def expf(x) -> np.ndarray:
    """newlib __ieee754_expf, as compiled (the wrapper's errno paths return the same inf / 0)."""
    x = np.asarray(x, F32)
    bits = x.view(U32)
    sx = x.view(np.int32)
    xsb = (bits >> 31) & 1
    hx = bits & 0x7fffffff
    out = np.empty_like(x)

    nan = hx > 0x7f800000
    inf = hx == 0x7f800000
    ovf = ~nan & ~inf & (sx > 0x42b17217)
    unf = ~nan & ~inf & (sx < 0) & (hx > 0x42cff1b5)
    rest = ~(nan | inf | ovf | unf)
    tiny = rest & (hx < 0x34000000)
    small = rest & ~tiny & (hx <= 0x3eb17218)                  # k = 0
    red1 = rest & (hx > 0x3eb17218) & (hx <= 0x3f851591)       # |x| in (0.5 ln2, 1.5 ln2]
    red2 = rest & (hx > 0x3f851591)

    out[nan] = x[nan] + x[nan]
    out[inf] = np.where(xsb[inf] == 0, x[inf], F32(0))
    out[ovf] = F32(np.inf)
    out[unf] = F32(0)
    out[tiny] = ONE + x[tiny]

    red = red1 | red2
    sgn = np.where(xsb == 1, F32(-1), F32(1))
    hi = np.zeros_like(x)
    lo = np.zeros_like(x)
    k = np.zeros(x.shape, np.int64)
    hi[red1] = x[red1] - LN2HI * sgn[red1]
    lo[red1] = LN2LO * sgn[red1]
    k[red1] = 1 - 2 * xsb[red1].astype(np.int64)
    halF = np.where(xsb == 1, F32(-0.5), F32(0.5))
    k[red2] = np.trunc(fma32(x[red2], INVLN2, halF[red2])).astype(np.int64)
    t = k[red2].astype(F32)
    hi[red2] = fma32(-t, LN2HI, x[red2])
    lo[red2] = t * LN2LO

    xr = np.where(red, (hi - lo).astype(F32), x)
    poly = small | red
    xp = xr[poly]
    tt = xp * xp
    p = fma32(fma32(fma32(fma32(tt, P5, P4), tt, P3), tt, P2), tt, P1)
    c = fma32(-p, tt, xp)
    xc = xp * c
    kk = k[poly]
    y = np.where(kk == 0, ONE - ((xc / (c - TWO)) - xp),
                 ONE - ((lo[poly] - (xc / (TWO - c))) - hi[poly])).astype(F32)
    yb = y.view(U32).astype(np.int64)
    normal = kk >= -125
    res = np.where(normal, yb + (kk << 23), yb + ((kk + 100) << 23)).astype(np.uint64).astype(U32).view(F32)
    res = np.where(normal, res, res * TWOM100).astype(F32)
    out[poly] = np.where(kk == 0, y, res)
    return out


def rmsnorm(h: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """mx_rmsnorm: squares summed LEFT TO RIGHT in double (each exact: h is bf16-valued), mean cast to
    float32, + eps, 1/sqrtf, then (h * w) * inv in float32 -- the source says h * inv * w, but -ffast-math
    reassociates it to put the loop-invariant inv last, and that is what the device computes."""
    h = np.asarray(h, F32)
    D = h.shape[-1]
    sq = h.astype(F64) * h.astype(F64)
    ss = np.add.accumulate(sq, axis=-1)[..., -1:]               # sequential, not np.sum's pairwise
    inv = (ONE / np.sqrt(((ss / F64(D)).astype(F32) + F32(eps)).astype(F32))).astype(F32)
    return ((h * np.asarray(w, F32)).astype(F32) * inv).astype(F32)


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """mx_rope_at: x*cos + rotate_half(x)*sin, which the compiler contracts to fma(x, cos, rot*sin)
    (rot*sin rounded, then one fused multiply-add)."""
    x = np.asarray(x, F32)
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[:, half:], x[:, :half]], axis=-1)
    return fma32(x, np.asarray(cos, F32), (rot * np.asarray(sin, F32)).astype(F32))


def silu(x: np.ndarray) -> np.ndarray:
    """mx_silu: x / (1 + expf(-x)), single precision."""
    x = np.asarray(x, F32)
    return (x / (ONE + expf(-x))).astype(F32)


def softmax_causal(s: np.ndarray) -> np.ndarray:
    """mx_softmax_causal on already-scaled fp32 scores [M, M]: per row m over columns 0..m, subtract
    the max, expf, sum LEFT TO RIGHT in float32, multiply by 1/sum. Columns past m are 0."""
    s = np.asarray(s, F32)
    M = s.shape[0]
    mask = np.tril(np.ones((M, M), dtype=bool))
    v = np.where(mask, s, F32(-np.inf))
    mx = v.max(axis=1, keepdims=True)
    e = np.where(mask, expf(np.where(mask, s - mx, F32(0)).astype(F32)), F32(0))
    acc = np.zeros(M, F32)
    for j in range(M):                                          # the device's sequential order;
        acc = (acc + e[:, j]).astype(F32)                       # adding the masked 0s changes nothing
    return (e * (ONE / acc)[:, None]).astype(F32)
