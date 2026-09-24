"""Golden generator: MXQuant's end-to-end block quantizer -> hardware wire format.

This is the **single source of truth** for what the spike model and the RTL requantizer are
supposed to produce. It does not reimplement any quantization arithmetic: it calls
``quantize_mx_block32`` -- MXQuant's API, computed by the mxq library through
``models/mxquant/block.py`` and proved bit-identical to MXQuant's own quantizer on every format
(``tests/selftest_block.py``) -- so it cannot drift from the reference the LLM evaluations use.

``quantize_mx_block32`` returns *values* — ``(P, X)`` with ``V_hat = P * broadcast(X)``. The
hardware instead emits a **wire format**: one E8M0 scale byte per block plus one E4M3 code byte
per element. The only thing this module adds is the conversion between those two, and that
conversion is lossless by construction:

* ``P`` is already an exact E4M3 value (that is what ``_quantize_elemwise`` produced), so it is
  encoded by an **exact 256-entry table lookup**, not by rounding. A value that is not in the
  table is an error, raised loudly, not rounded away.
* ``X`` is an exact power of two, so its E8M0 code is ``log2(X) + 127``, verified by re-decoding.

``golden()`` therefore returns codes whose decode is *bit-identical* to the reference's own
``P * X``; :func:`self_check` asserts exactly that on every element it is given.

The reference's convention, for the record (measured, see ``planning/chain_seam_hw_notes.md``):

* scale exponent is ``floor(log2(max(amax, 2**-23)))`` with **no** ``- emax`` term, so a block's
  max lands in ``[1, 2)`` -- NOT the ``[256, 512)`` of ``microxcaling.mx.mx_ops._quantize_mx``.
  The two differ by exactly ``emax = 8`` exponents.
* elements round **half to even** (RNE) — see :data:`ROUND_MODE`, which overrides MXQuant's
  ``"nearest"`` default so that operands and the hardware's requant output share one convention.
  They saturate to +-448, and E4M3 subnormals are allowed down to ``2**-9``.

Layouts, matching the two places the datapath needs blocks:

* ``axis="row"`` -- 1x32 blocks along columns, ``X`` is ``[R][C/32]``. This is the
  **requantizer output** layout (``MxRequantizer.scala:565`` writes one byte per row per 32
  output columns).
* ``axis="col"`` -- 32x1 blocks along rows, ``X`` is ``[R/32][C]``. This is the **operand**
  layout, and the one ``MXQuant/end_to_end_linear/stats_from_pairs.py`` uses.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# --- the reference quantizer: MXQuant's API, computed by mxq ---------------------------------------
# ``models/mxquant/block.py`` provides ``quantize_mx_block32`` / ``_broadcast_scales`` / ``BLOCK`` with
# MXQuant's shapes and numerics (round-to-nearest-even, block max floored at FLT_EPSILON) on the mxq
# submodule, so the MXQuant clone is no longer needed to compile a kernel. ``MXQ_ROOT`` is kept only
# for the optional real-tile data that ``_llama_tiles`` / ``self_check`` read when it is present.

MXQ_ROOT = Path(__file__).resolve().parent.parent / "MXQuant"

import torch  # noqa: E402
from models.mxquant.block import BLOCK, _broadcast_scales, quantize_mx_block32  # noqa: E402

from . import mxformats  # noqa: E402
from .mxwire import E8M0_BIAS, fp8_e4m3_decode  # noqa: E402

# The block size is stated in three places -- the quantizer, our format table, and the RTL. Two of
# them are importable, so check them against each other rather than trusting that they agree.
if BLOCK != mxformats.BLOCK:
    raise ImportError(
        f"block-scale group disagreement: models/mxquant/block.py says {BLOCK}, app/mxformats.py says "
        f"{mxformats.BLOCK}. One of them is wrong about the hardware.")

Axis = Literal["row", "col"]

#: Rounding mode for EVERY quantization this module performs — operands included.
#:
#: MXQuant's own default is ``"nearest"``, which in microxcaling's vocabulary means half AWAY from
#: zero, not RNE. The hardware requantizer used to agree; on 2026-09-10 it moved to RNE for every
#: format (``mx_fp_math.h``, plan §4.18), so leaving operands on the default would have put two
#: rounding conventions inside a single ELF — layer 0's operands half-away, every chained
#: intermediate RNE.
#:
#: Measured impact of the switch on random data: 0/4096 elements in all five formats, because an
#: exact tie is measure-zero in random floats. That is precisely why it needed fixing deliberately
#: rather than being left to surface later: it is the same shape as the `fp8_e4m3` seam that broke
#: when spike moved, a convention that holds right up until one side changes.
ROUND_MODE = "even"


def _require_no_pmax_shift(pmax_shift: int, fmt: str) -> None:
    """Refuse a nonzero ``pmax_shift`` instead of silently ignoring it.

    The shift existed because the requantizer used to normalize each block to
    ``[2**out_pmax, 2**(out_pmax+1))`` -- ``- 4`` on the FP6 path, ``- 2`` on FP4. The RTL and spike
    now use ``log2_pmax = 0`` for every format, so every ``MxFormat.out_pmax`` is 0 and the code
    that re-ran the element quantizer against a shifted scale became unreachable. It is deleted
    rather than kept warm, because it was the only thing in this module reaching into
    ``microxcaling`` internals.

    This guard is what makes the deletion safe: a format that reintroduces a nonzero ``out_pmax``
    gets an error naming the cause, not a quietly wrong golden. Its sibling is
    ``mxformats.chain_refusal``, dormant for the same reason.
    """
    if pmax_shift:
        raise NotImplementedError(
            f"pmax_shift={pmax_shift} requested for {fmt}, but the shifted-scale quantizer was "
            "removed when the hardware moved to log2_pmax = 0 for every format (plan §4.9). "
            "Reinstate it from git history if a format brings a nonzero out_pmax back.")


#: E4M3 code for NaN. ``mx_fp_math.h:232`` and ``mxwire.fp8_e4m3_decode`` both decode 0x7F as
#: 480.0 rather than NaN, so these two codes are held out of the lookup table and handled
#: explicitly -- ``saturate_normals=True`` means the reference never emits 480, so nothing is lost.
E4M3_NAN = 0x7F
E4M3_NAN_NEG = 0xFF
#: E8M0 code for NaN (``mx_fp_math.h:254`` decodes it as NaN).
E8M0_NAN = 0xFF


def _e4m3_value_table() -> dict[int, int]:
    """fp32 bit pattern of every finite E4M3 value -> its code.

    Keyed on the *bit pattern* so that -0.0 (0x80000000) and +0.0 (0x00000000) stay distinct
    instead of comparing equal.
    """
    table: dict[int, int] = {}
    codes = np.array([c for c in range(256) if c not in (E4M3_NAN, E4M3_NAN_NEG)], dtype=np.uint8)
    vals = fp8_e4m3_decode(codes)
    for c, v in zip(codes.tolist(), vals.tolist()):
        bits = int(np.float32(v).view(np.uint32))
        table.setdefault(bits, int(c))
    return table


_E4M3_TABLE = _e4m3_value_table()


def e4m3_encode_exact(vals: np.ndarray) -> np.ndarray:
    """Encode already-E4M3-exact values to codes by table lookup. Never rounds.

    Raises if a value is not representable, which would mean the reference produced something
    outside E4M3 and silently rounding it would hide a real bug.
    """
    v = np.asarray(vals, dtype=np.float32)
    out = np.zeros(v.shape, dtype=np.uint8)
    bits = v.view(np.uint32)
    flat_bits, flat_out, flat_v = bits.ravel(), out.ravel(), v.ravel()
    bad: list[float] = []
    for i in range(flat_v.size):
        x = float(flat_v[i])
        if np.isnan(x):
            flat_out[i] = E4M3_NAN
            continue
        code = _E4M3_TABLE.get(int(flat_bits[i]))
        if code is None:
            bad.append(x)
            continue
        flat_out[i] = code
    if bad:
        uniq = sorted(set(bad))[:8]
        raise ValueError(
            f"{len(bad)} value(s) from the reference are not exact E4M3 values, e.g. {uniq}. "
            "The wire encoding is meant to be lossless; rounding here would hide the cause.")
    return out


def e8m0_encode_exact(X: np.ndarray) -> np.ndarray:
    """Encode exact powers of two to E8M0 codes. Non-finite scales become the E8M0 NaN code."""
    x = np.asarray(X, dtype=np.float32)
    out = np.full(x.shape, E8M0_NAN, dtype=np.uint8)
    finite = np.isfinite(x) & (x > 0)
    if finite.any():
        exp = np.log2(x[finite].astype(np.float64))
        if not np.all(exp == np.floor(exp)):
            off = exp[exp != np.floor(exp)][:8]
            raise ValueError(f"block scale(s) are not powers of two: log2 = {off}")
        code = exp + E8M0_BIAS
        if np.any((code < 0) | (code > 254)):
            rng = (float(code.min()), float(code.max()))
            raise ValueError(
                f"block scale exponent(s) fall outside the E8M0 code range [0, 254]: {rng}. "
                "The reference clamps amax at 2**-23 but places no upper bound, so this means "
                "the input tile itself is out of range.")
        out[finite] = code.astype(np.uint8)
    return out


@dataclass(frozen=True)
class Golden:
    """The reference answer for one tile, in the hardware's wire format."""
    codes: np.ndarray    #: uint8 [R][C] -- one E4M3 code per element
    scales: np.ndarray   #: uint8, [R][C//32] for axis="row", [R//32][C] for axis="col"
    recon: np.ndarray    #: float32 [R][C] -- decode(codes) * broadcast(decode(scales))
    P: np.ndarray        #: float32 [R][C] -- the reference's own normalized quantized values
    X: np.ndarray        #: float32 -- the reference's own block scales
    axis: str

    @property
    def scale_layout(self) -> str:
        return "[R][C/32]" if self.axis == "row" else "[R/32][C]"


def golden(V: np.ndarray, *, axis: Axis = "row", fmt: str = "MXFP8_E4M3",
           pmax_shift: int = 0, encode=None, decode=None) -> Golden:
    """Quantize ``V[R][C]`` with MXQuant's e2e quantizer and return it in wire format.

    ``pmax_shift`` must be 0; see :func:`_require_no_pmax_shift`. It survives as a parameter only so
    that a caller carrying a format's ``out_pmax`` gets a named error rather than silence.

    The returned ``recon`` is bit-identical to ``P * broadcast(X)`` -- asserted here, not assumed,
    so a drift in either the reference or the encoding fails at the source.
    """
    V = np.ascontiguousarray(V, dtype=np.float32)
    if V.ndim != 2:
        raise ValueError(f"V must be 2-D, got shape {V.shape}")
    R, C = V.shape
    span = C if axis == "row" else R
    if span % BLOCK:
        raise ValueError(
            f"axis={axis!r} blocks along a length-{span} axis, which must be a multiple of "
            f"{BLOCK}; got shape {V.shape}")

    _require_no_pmax_shift(pmax_shift, fmt)
    out = quantize_mx_block32(torch.from_numpy(V), fmt=fmt, axis=axis,
                              round_mode=ROUND_MODE)
    P = out.P.numpy().astype(np.float32)
    X = out.X.numpy().astype(np.float32)

    codes = (encode or e4m3_encode_exact)(P)
    scales = e8m0_encode_exact(X)

    # Reconstruct from the WIRE values only, then require it to equal the reference's own product.
    from .mxwire import e8m0_decode
    scale_vals = e8m0_decode(scales)
    tiles = _broadcast_scales(torch.from_numpy(scale_vals), (R, C), axis).numpy()
    recon = ((decode or fp8_e4m3_decode)(codes) * tiles).astype(np.float32)

    with np.errstate(invalid="ignore"):   # a non-finite block legitimately yields nan * inf
        ref = (P * _broadcast_scales(out.X, (R, C), axis).numpy()).astype(np.float32)
    same = (recon == ref) | (np.isnan(recon) & np.isnan(ref))
    if not same.all():
        i = int(np.argmax(~same.ravel()))
        raise AssertionError(
            f"wire encoding is not lossless at flat index {i}: reference P*X = "
            f"{ref.ravel()[i]!r}, decode(wire) = {recon.ravel()[i]!r}")

    return Golden(codes=codes, scales=scales, recon=recon, P=P, X=X, axis=axis)


# --- the operand entry point ----------------------------------------------------------------------
#
# This is what the live path calls. It exists so that the two questions the seam used to answer
# implicitly -- WHICH AXIS a tensor blocks along, and WHICH LAYOUT its scales come back in -- are
# answered at the call site, in the caller's own vocabulary ("this is the A operand"), instead of
# by a transpose buried three modules away.

def _exact_encoder(fmt) -> "callable":
    """Build an exact value->code table for ``fmt`` by inverting its decoder over the code space.

    Same construction as :func:`e4m3_encode_exact`, generalized: the reference's ``P`` is already an
    exact value of the element format, so encoding is a lookup, never a rounding. A value that is
    not in the table raises — it would mean the reference produced something outside the format, and
    rounding it here would hide that.

    Keyed on the fp32 BIT PATTERN so -0.0 and +0.0 stay distinct.
    """
    from .mxwire import DECODERS

    decode = DECODERS.get(fmt.name)
    if decode is None:
        raise MxGoldenError(
            f"no element decoder for {fmt.name!r}; add one to app/mxwire.DECODERS. The encoder is "
            "derived from the decoder so that a format's semantics live in exactly one place.")
    codes = np.arange(1 << fmt.bits, dtype=np.uint8)
    table: dict[int, int] = {}
    for c, v in zip(codes.tolist(), decode(codes).tolist()):
        table.setdefault(int(np.float32(v).view(np.uint32)), int(c))

    def encode(vals: np.ndarray) -> np.ndarray:
        v = np.ascontiguousarray(vals, dtype=np.float32)
        bits = v.view(np.uint32).ravel()
        out = np.zeros(v.size, dtype=np.uint8)
        bad = []
        for i, b in enumerate(bits.tolist()):
            c = table.get(b)
            if c is None:
                bad.append(float(v.ravel()[i]))
            else:
                out[i] = c
        if bad:
            raise MxGoldenError(
                f"{len(bad)} value(s) are not exact {fmt.name} values, e.g. {sorted(set(bad))[:8]}. "
                "The wire encoding is meant to be lossless; rounding here would hide the cause.")
        return out.reshape(v.shape)

    return encode


class MxGoldenError(RuntimeError):
    """The reference produced something the wire format cannot carry losslessly."""


def normalized(V: np.ndarray, *, fmt: str = "MXFP8_E4M3", axis: Axis = "row",
               pmax_shift: int = 0) -> np.ndarray:
    """The block-normalized values ``P`` alone -- no wire encoding, no losslessness assertion.

    This is what a *codebook* is built from: the requantizer divides by the block scale before
    projecting, so the values a codebook must span are ``P``, not ``V``.

    ``pmax_shift`` matters and is easy to forget. The requantizer's scale is
    ``2**(floor(log2 amax) - out_pmax)``, so its normalized output spans
    ``[2**out_pmax, 2**(out_pmax+1))`` -- ``[16, 32)`` for E3M2, against the ``[1, 2)`` MXQuant's own
    convention produces. Pass the format's ``out_pmax`` or the codebook will be built for the wrong
    decade and every value will saturate onto one entry.
    """
    V = np.ascontiguousarray(V, dtype=np.float32)
    _require_no_pmax_shift(pmax_shift, fmt)
    out = quantize_mx_block32(torch.from_numpy(V), fmt=fmt, axis=axis,
                              round_mode=ROUND_MODE)
    return out.P.numpy().astype(np.float32)


def pack_operand(codes: np.ndarray, *, side: Literal["a", "b"],
                 dtype: str = "fp8_e4m3") -> np.ndarray:
    """Pack element codes into the exact byte layout the mesh reads.

    An 8-bit format is already its own wire layout and passes through. A 4-bit format is packed two
    codes per byte — and WHICH two differs by side, which is the part that is easy to get wrong:

    ``gemmini.cc:1533-1541`` (the FP4 kernel) reads
    ``A: spad[A_t + (m >> 1)][kk]``, nibble ``(m & 1) ? high : low`` — two **m-rows** share a byte
    ``B: spad[B_t + kk][n >> 1]``, nibble ``(n & 1) ? high : low`` — two **n-columns** share a byte

    So A packs DOWN its rows to ``[M/2][K]`` and B packs ACROSS its columns to ``[K][N/2]``. They are
    not the same operation, and at a square shape they are indistinguishable by size alone.
    """
    from . import mxformats

    f = mxformats.get(dtype, where=f"pack_operand(side={side!r})")
    c = np.ascontiguousarray(codes, dtype=np.uint8)
    if f.bits == 8:
        return c
    if f.bits != 4:
        raise mxformats.MxFormatError(f"no packing defined for {f.bits}-bit {f.name}")
    if (c > 0xF).any():
        raise ValueError(f"{f.name} codes must be 4-bit; found values above 0xF")

    rows, cols = c.shape
    if side == "a":
        if rows % 2:
            raise ValueError(f"A has {rows} rows; 4-bit packing pairs rows, so M must be even")
        lo, hi = c[0::2, :], c[1::2, :]              # even m -> low nibble, odd m -> high
        return np.ascontiguousarray((lo | (hi << 4)).astype(np.uint8))
    if cols % 2:
        raise ValueError(f"B has {cols} columns; 4-bit packing pairs columns, so N must be even")
    lo, hi = c[:, 0::2], c[:, 1::2]                  # even n -> low nibble, odd n -> high
    return np.ascontiguousarray((lo | (hi << 4)).astype(np.uint8))


def quantize_operand(V: np.ndarray, *, side: Literal["a", "b"],
                     dtype: str = "fp8_e4m3") -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Quantize one matmul operand to the wire format the device reads.

    ``side`` names the operand, and that alone fixes both the blocking axis and the scale layout,
    because the datapath indexes the two scale memories differently
    (``gemmini.cc``: ``a_off = group * M + row``, ``b_off = group * N + col``):

    ======  ==============  =====================  =========================
    side    tensor          blocks along           scales come back as
    ======  ==============  =====================  =========================
    ``a``   ``A[M][K]``     K, i.e. ``axis="row"``  ``[K/32][M]`` (transposed)
    ``b``   ``B[K][N]``     K, i.e. ``axis="col"``  ``[K/32][N]`` (as produced)
    ======  ==============  =====================  =========================

    Both block along **K**, the contraction axis — they differ only in which way K runs in the
    stored tensor. Codes are returned in the tensor's own layout, unmoved.

    Returns ``(codes, scales, codebooks)``. ``codebooks`` is ``None`` for a direct format and the
    packed ``[n_groups][words]`` uint32 LUT for a codebook format -- which is compile OUTPUT derived
    from the data, so it cannot be reconstructed downstream and travels with the operands.

    This matches the baremetal headers byte for byte: ``gen_matmul_llama.py:280-281`` quantizes A
    with ``axis="row"`` and writes ``A_scales_row[GK][M]`` from the transpose, and B with
    ``axis="col"`` and writes ``B_scales_col[GK][N]`` directly. Verified in
    ``tests/selftest_quantizer.py``.
    """
    from . import mxformats

    f = mxformats.get(dtype, where=f"quantize_operand(side={side!r})")
    V = np.ascontiguousarray(V, dtype=np.float32)

    # Refuse non-finite input rather than coding it. The element encoders map NaN/Inf to a code
    # that decodes back to something finite-looking, so an upstream accumulator overflow would
    # vanish here and the run would look clean -- the failure mode recorded in
    # npu_exploration_bridge_plan.md §14.1, which turned a loud failure into a silent wrong answer.
    if not np.isfinite(V).all():
        bad = int((~np.isfinite(V)).sum())
        raise ValueError(
            f"{bad} non-finite value(s) in the {side.upper()} operand to quantize. These would be "
            "silently coded rather than raised; an earlier stage probably overflowed the mesh "
            "accumulator.")

    if side not in ("a", "b"):
        raise ValueError(f"side must be 'a' or 'b', got {side!r}")
    from .mxwire import DECODERS

    axis = "row" if side == "a" else "col"

    if f.lut:
        # A codebook format does not put element codes on the wire: it puts 4-bit INDICES into a
        # per-group codebook built from the data. MXQuant still does the block quantization -- the
        # codebook is a second, coarser step on top of its output, exactly as
        # microxcaling's level2_scratch does. See app/mxlut.py.
        from . import mxlut

        out = quantize_mx_block32(torch.from_numpy(V), fmt=f.mxq, axis=axis,
                                  round_mode=ROUND_MODE)
        P = out.P.numpy().astype(np.float32)
        books = mxlut.build_codebooks(P, axis=axis, fmt=f)
        idx = mxlut.assign_indices(P, books, axis=axis)
        codes = pack_operand(idx, side=side, dtype=dtype)
        scales = e8m0_encode_exact(out.X.numpy().astype(np.float32))
        scales = np.ascontiguousarray(scales.T) if side == "a" else np.ascontiguousarray(scales)
        return codes, scales, mxlut.pack_codebooks(books, fmt=f)

    # Both halves must be the FORMAT's own: golden() asserts decode(encode(P)) == P*X on every
    # call, and passing an encoder without its matching decoder turns that gate into a false alarm.
    g = golden(V, axis=axis, fmt=f.mxq, pmax_shift=0,
               encode=_exact_encoder(f), decode=DECODERS[f.name])
    codes = pack_operand(g.codes, side=side, dtype=dtype)
    # A blocks along K with scales [M][GK]; the A-side scale memory indexes [GK][M]
    # (a_off = group*M + row), so it is transposed. B already comes back [GK][N].
    scales = np.ascontiguousarray(g.scales.T) if side == "a" else np.ascontiguousarray(g.scales)
    return codes, scales, None


class NotModelled(RuntimeError):
    """The reference cannot reproduce this edge, so no golden should be claimed for it."""


def requantize_chained(C_bf16: np.ndarray, *, dtype: str = "fp8_e4m3",
                       books: np.ndarray | None = None):
    """The A operand a CHAINED stage receives: what the device's requantizer wrote.

    Not the same thing as quantizing the intermediate on the host — the device requantizes with its
    own rounding, and every format now goes through a transcription of the block that does it
    rather than through MXQuant.

    That was not always so. E4M3-single used to be routed to :func:`quantize_operand`, on the
    grounds that its requantizer had been migrated to MXQuant's convention. The 2026-09-10 header
    revision moved the hardware to RNE while MXQuant's default stayed half-away, and the chain
    dropped to 1242/4096 — a shared convention is only shared until one side changes. Transcribing
    the device block means the reference tracks the hardware by construction.

    Returns ``(P, X)`` in the model's orientation, ready for `rtl_datapath`'s wire-operand hook.
    """
    from . import mxformats
    from .mxwire import e8m0_decode

    f = mxformats.get(dtype, where="requantize_chained")
    C_bf16 = np.ascontiguousarray(C_bf16, dtype=np.float32)

    if not f.lut and f.bits == 8:
        return _requantize_direct_e4m3(C_bf16, f)

    if f.lut:
        return _requantize_codebook(C_bf16, f, books)

    # The hardware's own requantizer, from the extracted mesh model.
    from .mxmesh import fp4 as M4, fp8 as M8
    mm = M8 if (f.spec.startswith("fp8") and not f.lut) else M4
    C_q, C_sc = mm.matrix_mx_requantize(torch.from_numpy(C_bf16), f.spec)
    P = C_q.numpy().astype(np.float32)
    X = C_sc.numpy().astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(P.T)), torch.from_numpy(X.T)


def _requantize_direct_e4m3(C: np.ndarray, f):
    """The DIRECT 8-bit E4M3 requant-output path, from ``gemmini.cc:1303-1378``.

    Sibling of :func:`_requantize_codebook`, and the differences from it are the point — sharing
    one routine would have to be parameterized on all three:

    * **no bf16 rounding of the SCALED value.** The codebook path stores ``f32_to_bf16_rne(scaled)``
      and encodes those bits; this one calls ``fp8_e4m3_to_code(v / scale)`` on the raw float. (The
      INPUT is bf16-rounded in both, because both read it out of smem, which is bf16 — two
      different roundings that are easy to conflate.)
    * **an epsilon clamp on the block max** — ``amax = max(amax, FLT_EPSILON)``, so an all-zero
      block gets code 104, not 0. The comment at ``gemmini.cc:1341`` explains why: code 0 used to
      mean both "all-zero block" and "the accumulator overflowed".
    * **NaN/Inf propagate out of band** as E8M0 code ``0xFF``, with the scale itself set to NaN/Inf
      so the element codes match the reference byte for byte.

    No finder and no codebook: the output is an 8-bit code the next stage decodes directly.
    """
    from .mxwire import (BLOCK, bf16_bits_to_float, float_to_bf16_bits, fp8_e4m3_decode,
                         fp8_e4m3_encode_f32)

    C = bf16_bits_to_float(float_to_bf16_bits(C))     # smem holds bf16; the block max is over those
    M, N = C.shape
    if N % BLOCK:
        raise NotModelled(
            f"{f.name} chained: the requantizer blocks its output in {BLOCK}s along N, and this "
            f"stage's N is {N}.")
    blocks = C.reshape(M, N // BLOCK, BLOCK)

    finite = np.isfinite(blocks)
    bad = ~finite.all(axis=2)                       # a block holding any NaN or Inf
    has_nan = np.isnan(blocks).any(axis=2)
    max_abs = np.where(finite, np.abs(blocks), 0.0).max(axis=2)
    amax = np.maximum(max_abs, np.finfo(np.float32).eps)
    scale_code = np.clip(np.floor(np.log2(amax)).astype(np.int64) + E8M0_BIAS, 0, 254).astype(np.uint8)
    X = np.exp2((scale_code.astype(np.int64) - E8M0_BIAS).astype(np.float64)).astype(np.float32)
    # The out-of-band case, reproduced rather than smoothed over: /inf sends a block's finite
    # siblings to zero and the inf itself to NaN, /nan sends every element to NaN.
    if bad.any():
        X = np.where(bad, np.where(has_nan, np.float32(np.nan), np.float32(np.inf)), X)

    scaled = (blocks / X[:, :, None]).reshape(M, N)
    codes = np.array([fp8_e4m3_encode_f32(v) for v in scaled.ravel().tolist()],
                     dtype=np.uint8).reshape(M, N)
    P = fp8_e4m3_decode(codes).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(P.T)), torch.from_numpy(np.ascontiguousarray(X.T))


def _requantize_codebook(C: np.ndarray, f, books):
    """The codebook requant-output path, transcribed statement for statement from ``gemmini.cc``.

    A codebook intermediate never leaves the device as a value: the requantizer picks a 4-bit INDEX
    into the OUTPUT book, and the next matmul multiplies whatever entry that index names. So the
    thing to reproduce is the index, and the whole sequence has to be right to get it
    (``gemmini.cc:1445-1509``):

    1. block along COLUMNS in 32s, per row -- one E8M0 scale per (row, block), ``log2_pmax = 0``;
    2. ``scaled = v / scale``, rounded to **bf16** before anything looks at it;
    3. the element code, rounded **as that format rounds** -- :mod:`app.mxwire`'s ``ENCODERS``,
       not a generic nearest-grid-point (see the note there: three separate conventions);
    4. the hardware's fixed-point nearest-finder picks the index (:func:`mxlut.finder_indices`);
    5. the value the next matmul sees is ``book[index]``, decoded.

    Verified against a C oracle compiled from ``mx_fp_math.h`` itself: 4096/4096 identical at every
    one of steps 1, 3 and 4, for all four codebook formats. Getting any one of them generically
    right is not enough -- an earlier version had the finder exactly right and still matched
    27/4096, because step 3 rounded ties the wrong way and dropped the sign of zero.
    """
    from . import mxlut
    from .mxwire import BLOCK, bf16_bits_to_float, encode_requant, float_to_bf16_bits

    if books is None:
        raise ValueError(f"{f.name} chains through a codebook; its C book is needed")
    vals = mxlut.unpack_codebooks(books, fmt=f)
    g = mxformats.LUT_GRANULARITY
    # The requantizer reads the accumulator out of SMEM, where it is bf16 -- so the block maximum
    # is a maximum over bf16 values, not over the fp32 the caller happens to hold. Under
    # `rtl_exact` the caller's array is already bf16-valued and this is a no-op; it is here so the
    # function is right for any input, not only the one path that reaches it today.
    C = bf16_bits_to_float(float_to_bf16_bits(C))
    M, N = C.shape
    if N % BLOCK:
        raise NotModelled(
            f"{f.name} chained: the requantizer blocks its output in {BLOCK}s along N, and this "
            f"stage's N is {N}. A partial block is not what the device does with a full tile.")
    nb = N // BLOCK

    blocks = C.reshape(M, nb, BLOCK)
    max_abs = np.abs(blocks).max(axis=2)
    with np.errstate(divide="ignore"):
        max_exp = np.floor(np.log2(max_abs.astype(np.float32))).astype(np.int64)
    scale_code = np.clip(max_exp + E8M0_BIAS, 0, 254).astype(np.uint8)
    scale_code[max_abs == 0.0] = 0                     # log2(0) is -inf, and the device special-cases it
    X = np.exp2((scale_code.astype(np.int64) - E8M0_BIAS).astype(np.float64)).astype(np.float32)

    elem = encode_requant((blocks / X[:, :, None]).reshape(M, N), dtype=f.name)
    idx = mxlut.finder_indices(elem, vals, fmt=f, axis="row", g=g)
    P = np.take_along_axis(vals[np.arange(M) >> g], idx.astype(np.intp), axis=1).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(P.T)), torch.from_numpy(np.ascontiguousarray(X.T))


def wire_to_px(codes: np.ndarray, scales: np.ndarray, *, side: Literal["a", "b"],
               dtype: str = "fp8_e4m3", books: np.ndarray | None = None,
               shape: tuple[int, int] | None = None):
    """Decode wire bytes back to the ``(P, X)`` pair a datapath model consumes.

    ``P`` is the block-normalized VALUE of each element and ``X`` its block scale — the two things
    the mesh multiplies. Going wire -> (P, X) rather than float -> (P, X) is what lets a model be
    driven by exactly what the device received, codebooks included.

    Returned in the model's orientation: ``P`` is ``[K][M]`` for the A side and ``[K][N]`` for B,
    with ``X`` ``[K/32][·]`` in both cases.
    """
    from . import mxformats, mxlut
    from .mxwire import DECODERS, e8m0_decode

    f = mxformats.get(dtype, where="wire_to_px")
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    if f.bits == 4:                                  # unpack the nibbles this side packs
        if side == "a":
            m2, k = codes.shape
            idx = np.empty((m2 * 2, k), np.uint8)
            idx[0::2], idx[1::2] = codes & 0xF, codes >> 4
        else:
            k, n2 = codes.shape
            idx = np.empty((k, n2 * 2), np.uint8)
            idx[:, 0::2], idx[:, 1::2] = codes & 0xF, codes >> 4
    else:
        idx = codes

    if f.lut:
        if books is None:
            raise ValueError(f"{f.name} is codebook-indexed; its books are needed to decode")
        g = mxformats.LUT_GRANULARITY
        vals = mxlut.unpack_codebooks(books, fmt=f)
        r, c = idx.shape
        if side == "a":                              # one book per 2**G ROWS of A
            P = np.array([[vals[i >> g, idx[i, j]] for j in range(c)] for i in range(r)], np.float32)
        else:                                        # one book per 2**G COLUMNS of B
            P = np.array([[vals[j >> g, idx[i, j]] for j in range(c)] for i in range(r)], np.float32)
    else:
        P = DECODERS[f.name](idx).astype(np.float32)

    X = e8m0_decode(np.ascontiguousarray(scales, dtype=np.uint8)).astype(np.float32)
    if side == "a":
        return torch.from_numpy(np.ascontiguousarray(P.T)), torch.from_numpy(X)
    return torch.from_numpy(P), torch.from_numpy(X)


def save(path, tiles: dict[str, np.ndarray], *, axis: Axis = "row") -> None:
    """Write goldens for ``{name: tile}`` to an ``.npz`` for the spike and RTL test benches.

    Each tile ``k`` contributes ``k/in`` (the fp32 input), ``k/codes``, ``k/scales`` and
    ``k/recon``. Consumers compare their own codes against ``k/codes`` byte-for-byte.
    """
    out: dict[str, np.ndarray] = {"__axis__": np.array(axis)}
    for name, V in tiles.items():
        g = golden(V, axis=axis)
        out[f"{name}/in"] = np.asarray(V, dtype=np.float32)
        out[f"{name}/codes"] = g.codes
        out[f"{name}/scales"] = g.scales
        out[f"{name}/recon"] = g.recon
    np.savez_compressed(path, **out)


def load(path) -> tuple[str, dict[str, dict[str, np.ndarray]]]:
    """Inverse of :func:`save`: returns ``(axis, {name: {"in","codes","scales","recon"}})``."""
    with np.load(path) as z:
        axis = str(z["__axis__"])
        tiles: dict[str, dict[str, np.ndarray]] = {}
        for k in z.files:
            if k == "__axis__":
                continue
            name, field = k.rsplit("/", 1)
            tiles.setdefault(name, {})[field] = z[k]
    return axis, tiles


# --- self-check ------------------------------------------------------------------------------------

def _corner_tiles() -> list[tuple[str, np.ndarray]]:
    """One 32x32 tile per corner the two oracles disagree on today."""
    def tile(col0: list[float]) -> np.ndarray:
        t = np.full((BLOCK, BLOCK), 0.25, dtype=np.float32)
        t[:, 0] = np.array(col0, dtype=np.float32)
        return t

    return [
        ("all-zero block", np.zeros((BLOCK, BLOCK), dtype=np.float32)),
        ("max significand 1.99 (clip probe)", tile([1.99219] + [0.25] * 31)),
        ("max significand 1.75", tile([1.75] + [0.25] * 31)),
        ("subnormal max 2^-133", np.full((BLOCK, BLOCK), 5.877471754111438e-39, dtype=np.float32)),
        ("tiny max 2^-120 with exact zeros", tile([2.0 ** -120] + [0.0] * 31)),
        ("one +Inf", tile([0.5, 0.5, 0.5, np.inf] + [0.5] * 28)),
        ("one NaN", tile([np.nan] + [1.0] * 31)),
        ("rounding ties (half away from zero)",
         tile([1.0625, 1.1875, -1.0625, 2.0 ** -9 * 0.5, 2.0 ** -9 * 2.5] + [0.5] * 27)),
    ]


def _llama_tiles(limit: int | None = None) -> list[tuple[str, np.ndarray]]:
    """The real TinyLlama A/W tiles logged by MXQuant's eval run, if present."""
    root = MXQ_ROOT / "end_to_end_linear" / "systolic_simulation" / "data_evalrun_01"
    if not root.is_dir():
        return []
    files = sorted(root.glob("layer*/*/[AW]_square_0*.npz"))
    if limit is not None:
        files = files[:limit]
    tiles = []
    for f in files:
        with np.load(f) as z:
            tiles.append((str(f.relative_to(root)), z["data"].astype(np.float32)))
    return tiles


def self_check(verbose: bool = True) -> None:
    """Assert the wire encoding is lossless on corner cases and on the real TinyLlama tiles."""
    from .mxwire import e8m0_decode

    if verbose:
        print(f"reference: {MXQ_ROOT}/end_to_end_linear/mx_block_quant.py::quantize_mx_block32")
        print(f"\n--- corner tiles (axis='row', reporting column 0's block) ---")
    for name, V in _corner_tiles():
        golden(V, axis="row")                 # asserts losslessness internally
        g = golden(V, axis="col")
        sc = int(g.scales[0, 0])
        col = g.codes[:, 0]
        uniq = sorted(set(int(c) for c in col))[:5]
        if verbose:
            print(f"  {name:38s} scale=0x{sc:02X}({sc:3d}) "
                  f"={e8m0_decode(np.uint8(sc)):>12.6g}  codes[:,0] uniq={[hex(c) for c in uniq]}")

    tiles = _llama_tiles()
    if not tiles:
        if verbose:
            print("\n--- real TinyLlama tiles: NOT FOUND, skipped ---")
        return
    n_elem = n_blk = 0
    for name, V in tiles:
        for axis in ("row", "col"):
            g = golden(V, axis=axis)
            n_elem += g.codes.size
            n_blk += g.scales.size
    if verbose:
        print(f"\n--- real TinyLlama tiles ---")
        print(f"  {len(tiles)} tiles x 2 axes: {n_blk} blocks, {n_elem} elements, "
              f"all losslessly encoded")
        # Distribution facts the RTL/spike fixes will be judged against.
        subs = clips = zeros = tot = 0
        for name, V in tiles:
            g = golden(V, axis="col")
            z = g.P
            tot += z.size
            subs += int(((np.abs(z) > 0) & (np.abs(z) < 2.0 ** -6)).sum())
            clips += int((np.abs(z) == 448.0).sum())
            zeros += int((z == 0).sum())
        print(f"  elements: {100.0*subs/tot:.3f}% in the E4M3 subnormal tail, "
              f"{100.0*clips/tot:.3f}% saturated at 448, {100.0*zeros/tot:.3f}% zero")

    # Scale layouts, so a consumer cannot get the two axes the wrong way round.
    V = tiles[0][1]
    for axis in ("row", "col"):
        g = golden(V, axis=axis)
        if verbose:
            print(f"  axis={axis!r}: tile {V.shape} -> codes {g.codes.shape}, "
                  f"scales {g.scales.shape} = {g.scale_layout}")

    # save/load round trip
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.npz")
        save(p, {name: V for name, V in _corner_tiles()}, axis="row")
        ax, back = load(p)
        assert ax == "row" and len(back) == len(_corner_tiles())
        for name, V in _corner_tiles():
            assert np.array_equal(back[name]["codes"], golden(V, axis="row").codes)
    if verbose:
        print(f"  save/load round trip: OK")


if __name__ == "__main__":
    self_check()
