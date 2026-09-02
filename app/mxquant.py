"""MX (microscaling) quantization: float tensors -> operand codes + E8M0 block scales.

App layer — this produces OPERANDS, which is what the compiler backend receives on the command
buffer and never computes itself.

Every routine here is transcribed from the target's own functional model,
``software/libgemmini/mx_fp_math.h``, so the codes we hand the accelerator decode to exactly the
values it will compute with. Line references are to that file. Nothing here is derived from
radiance-kernels.

The MX datapath (from ``gemmini.cc`` around the LOOP_WS MX kernel):

* operands are 8-bit codes; each block of ``BLOCK=32`` elements along K shares one E8M0 exponent
* A's scale is indexed **per row of A, per K-group**  -> ``a_scales[k // 32][m]``
* B's scale is indexed **per column of B, per K-group** -> ``b_scales[k // 32][n]``
* the two scales MULTIPLY: ``e = sa + sb - 127``, i.e. ``2^(sa-127) * 2^(sb-127)``
* the group result is rounded to bf16 and accumulated in bf16
"""
from __future__ import annotations

import numpy as np

BLOCK = 32          # E8M0 group size along K (MxRequantizer.scala)
E8M0_BIAS = 127


# --- fp8 e4m3, transcribed from mx_fp_math.h -------------------------------------------------------

def fp8_e4m3_decode(code: np.ndarray) -> np.ndarray:
    """mx_fp_math.h:232. 1 sign | 4 exp | 3 mantissa, bias 7; e==0 is subnormal."""
    code = np.asarray(code, dtype=np.uint8)
    s = (code >> 7) & 1
    e = (code >> 3) & 0xF
    m = code & 0x7
    val = np.where(e == 0,
                   (m / 8.0) * 2.0 ** (1 - 7),
                   (1.0 + m / 8.0) * 2.0 ** (e.astype(np.int32) - 7))
    return np.where(s == 1, -val, val).astype(np.float32)


def _round_half_to_even(x: np.ndarray) -> np.ndarray:
    """mx_fp_math.h:196 — np.rint is round-half-to-even, matching the model's helper."""
    return np.rint(x)


def fp8_e4m3_to_code(v: np.ndarray) -> np.ndarray:
    """mx_fp_math.h:203. Vectorized transcription; the branch structure is preserved exactly."""
    v = np.asarray(v, dtype=np.float32)
    out = np.zeros(v.shape, dtype=np.uint8)
    finite_nz = np.isfinite(v) & (v != 0.0)
    if not finite_nz.any():
        return out

    s = (np.signbit(v) & finite_nz).astype(np.uint8)
    av = np.abs(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = np.floor(np.log2(np.where(finite_nz, av, 1.0))).astype(np.int32)
    emin, emax = -6, 8

    # Subnormal range [2^-9, 2^-6): quantum = 2^(emin - m_bits) = 2^-9   (mx_fp_math.h:211-217)
    sub = finite_nz & (E < emin)
    if sub.any():
        quantum = 2.0 ** (emin - 3)
        k = _round_half_to_even(av / quantum).astype(np.int32)
        code_sub = np.where(k <= 0, 0,
                            np.where(k >= 8, (1 << 3), k)).astype(np.uint8)
        out[sub] = ((s[sub] << 7) | code_sub[sub]).astype(np.uint8)

    # Normal range (mx_fp_math.h:219-229)
    nrm = finite_nz & (E >= emin)
    if nrm.any():
        E_used = np.minimum(E, emax).astype(np.int32)
        base = np.ldexp(np.ones_like(av), E_used).astype(np.float32)
        delta = base / 8.0
        k = _round_half_to_even((av - base) / delta).astype(np.int32)

        over = E > emax                      # saturate: E_used = emax, mant = 6
        carry = (~over) & (k >= 8)           # mantissa carried into the exponent
        E_used = np.where(carry, E_used + 1, E_used)
        k = np.where(carry, 0, k)
        sat_after_carry = carry & (E_used > emax)
        E_used = np.where(sat_after_carry, emax, E_used)
        k = np.where(sat_after_carry, 6, k)

        hi = np.where(E_used == emax, 6, 7)
        k = np.where(over, 6, np.clip(k, 0, hi))
        E_used = np.where(over, emax, E_used)

        code_nrm = ((s.astype(np.int32) << 7)
                    | (((E_used + 7) & 0xF) << 3)
                    | (k & 0x7)).astype(np.uint8)
        out[nrm] = code_nrm[nrm]
    return out


def e8m0_decode(code: np.ndarray) -> np.ndarray:
    """mx_fp_math.h:254. Code 0xFF is NaN; otherwise 2^(code-127)."""
    code = np.asarray(code, dtype=np.uint8).astype(np.int32)
    return np.where(code == 0xFF, np.nan, 2.0 ** (code - E8M0_BIAS)).astype(np.float32)


# --- Quantization ----------------------------------------------------------------------------------

#: Target exponent for a block's largest magnitude, i.e. codes are scaled to peak near
#: ``2**TARGET_CODE_EXP``. **This is NOT the textbook MX choice, and the difference matters.**
#:
#: Standard OCP MX puts the block max at the top of the element format's range (±448 for e4m3), to
#: use every code. On THIS datapath that overflows: the intermediate accumulator inside a 16-deep
#: column pass has a **4-bit exponent** (``gemmini.cc``: ``prod_e = 4``,
#: ``acc_e[] = {4 x15, 8}``), so it saturates around 2**8 = 256 —
#: ``fp_quantize_rne_scalar`` returns ±INFINITY above it (mx_fp_math.h:167) and
#: ``fp_add_exact`` turns mixed-sign infinities into NaN (:180). Full-range operands therefore
#: produce an all-NaN tile, not a degraded one.
#:
#: Bound: 16 accumulated products of codes with magnitude <= C need ``16 * C**2 <= 256``, so
#: ``C <= 4``. Hence exponent 2. The shipped ``matmul_fp8_64x64.h`` operands independently top out
#: near ±4, which corroborates it.
#:
#: Precision cost is small: e4m3 keeps 3 mantissa bits at every exponent, so *relative* precision is
#: unchanged; only the within-block dynamic range shrinks (still ~2**11 before subnormals).
TARGET_CODE_EXP = 2


def _shared_exponent(amax: np.ndarray, target_exp: int = TARGET_CODE_EXP) -> np.ndarray:
    """The E8M0 code for a block whose largest magnitude is ``amax``.

    Chooses ``2^e`` so that ``amax / 2^e`` lands near ``2**target_exp``. An all-zero block gets code
    127 (scale 1.0) rather than 0, so it stays exactly zero and introduces no denormal scale.
    """
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - target_exp
    e = np.where(amax > 0, e, 0.0)
    return np.clip(e + E8M0_BIAS, 0, 254).astype(np.uint8)


def quantize_rows(x: np.ndarray, *,
                  target_exp: int = TARGET_CODE_EXP) -> tuple[np.ndarray, np.ndarray]:
    """Quantize ``x[R][K]`` blockwise along K.

    Returns ``(codes[R][K], scales[K//BLOCK][R])`` — the scale layout the datapath indexes,
    ``scales[group][row]``, matching ``a_off = group * M + row`` in the spike kernel.
    """
    x = np.asarray(x, dtype=np.float32)
    r, k = x.shape
    if k % BLOCK:
        raise ValueError(f"K={k} must be a multiple of the block-scale group {BLOCK}")
    groups = k // BLOCK

    blocks = x.reshape(r, groups, BLOCK)
    amax = np.abs(blocks).max(axis=2)                       # [R][groups]
    scale_codes = _shared_exponent(amax, target_exp)        # [R][groups]
    scale = e8m0_decode(scale_codes)[:, :, None]            # [R][groups][1]
    codes = fp8_e4m3_to_code(blocks / scale).reshape(r, k)
    return codes, np.ascontiguousarray(scale_codes.T)       # -> [groups][R]


def quantize_matmul_operands(a: np.ndarray, b: np.ndarray, *,
                             target_exp: int = TARGET_CODE_EXP) -> dict[str, np.ndarray]:
    """Quantize ``A[M][K] @ B[K][N]`` into the ``mx_operands`` bundle the backend expects.

    B is quantized along ITS K axis, which means transposing to ``[N][K]`` first — B's scales are
    per output column per K-group (``b_off = group * N + col``).
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"contraction mismatch: A is {a.shape}, B is {b.shape}")
    a_codes, a_scales = quantize_rows(a, target_exp=target_exp)
    # B is quantized along ITS K axis, so transpose to [N][K] to block along K, then transpose the
    # CODES back to [K][N] — the on-device layout, verified against the shipped
    # matmul_fp8_64x64.h operands (whose B_in really is [K][N], as declared). Scales stay
    # [group][N]: b_off = group * N + col in the spike kernel.
    b_codes_nk, b_scales = quantize_rows(np.ascontiguousarray(b.T), target_exp=target_exp)
    return {
        "a_codes": a_codes,
        "b_codes": np.ascontiguousarray(b_codes_nk.T),   # [K][N], as the device expects
        "a_scales": a_scales,
        "b_scales": b_scales,
    }


# --- Device output decoding --------------------------------------------------------------------

def bf16_bits_to_float(bits: np.ndarray) -> np.ndarray:
    """Decode the bf16 bit patterns the device reports over OUT back to float32."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


# --- Chaining: rescaling the requantizer's output for a following GEMM -----------------------------

#: Exponent shift applied to requantizer output before it becomes the next GEMM's operand.
#:
#: The requantizer normalizes each output block's max to the element format's FULL range — RTL
#: ``MxRequantizer.scala``: ``scale_exponent = max_exp - log2_pmax_floor``, with
#: ``log2_pmax_floor = 8`` for FP8 — so codes come back peaking at 448. But the mesh accumulates a
#: 16-deep column at **exponent width 4** for 15 of its 16 rows
#: (``ConfigsFP.scala`` ``meshAccPrecisionList``), saturating near 2**8. Feeding requant output
#: straight into a second GEMM therefore overflows: measured 4096/4096 NaN.
#:
#: Both halves are individually correct and standard; they just do not compose. Shifting by 2**6
#: brings the peak to 7.0 and the chain is clean (measured: shift 0 -> 4096 NaN, 4 -> 341, 6 -> 0).
CHAIN_EXP_SHIFT = 6


def rescale_for_next_gemm(codes: np.ndarray, scales: np.ndarray, *,
                          shift: int = CHAIN_EXP_SHIFT) -> tuple[np.ndarray, np.ndarray]:
    """Make requantizer output safe as the next GEMM's A operand, preserving its value.

    Divides every code by ``2**shift`` and adds ``shift`` to the E8M0 scale, so
    ``code * 2**(scale-127)`` is unchanged. e4m3 keeps 3 mantissa bits at every exponent, so this is
    a pure exponent shift and is **lossless** until a value falls into the subnormal range.

    ``scales`` arrives as the requantizer writes it, ``[row][block]``; the returned scales are
    transposed to ``[group][row]``, the layout the A-side scale memory indexes
    (``a_off = group * M + row``). The two line up because GEMM 1's N equals GEMM 2's K.
    """
    codes = np.asarray(codes, dtype=np.uint8)
    shifted = fp8_e4m3_to_code(fp8_e4m3_decode(codes) / (2.0 ** shift))
    new_scales = np.clip(np.asarray(scales, dtype=np.int32) + shift, 0, 254).astype(np.uint8)
    return shifted, np.ascontiguousarray(new_scales.T)
