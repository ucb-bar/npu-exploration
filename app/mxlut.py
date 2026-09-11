"""Codebooks for the LUT-indexed MX formats.

Four of the six formats do not put element codes on the wire at all: they put **4-bit indices into a
16-entry codebook**, one codebook per ``2**G`` rows of A / columns of B
(``gemmini.cc:1392``: ``lut_idx = (i*TM + m) >> G``). The codebook is therefore part of the compiled
program, and — unlike everything else the backend emits — it is a function of the DATA, not just of
the shapes. That is why it rides the operand side channel and why the backend refuses to invent one.

The scheme follows MXQuant's ``prodacc_bundle/lut_quantization.py`` (itself a copy of
``microxcaling/mx/level2_scratch.py``): **MX-quantize first**, so every value is already a valid
element code, then reduce that per-group code set to 16 signposts by 1-D k-means. Snapping the
centroids back onto the format's own code set is what keeps every entry representable — a centroid
is a mean, and a mean of two codes is generally not a code.

Deterministic by construction (quantile init, no RNG), so the same tensor always compiles to the
same codebook and two builds of a kernel are comparable.

Ported from ``gemmini-rocc-tests/llama_operands.py`` (D1: the workload side moves into this repo).
"""
from __future__ import annotations

import numpy as np

from . import mxformats
from .mxwire import DECODERS

#: Entries per codebook. Fixed by the hardware: the index is a nibble.
LUT_SIZE = 16


# --- what the hardware's nearest-entry finder can actually tell apart ------------------------------
#
# THE non-obvious constraint on a codebook. The finder does not compare floats: it converts both the
# incoming code and every entry to a FIXED-POINT magnitude and takes the smallest absolute
# difference (`FP6E3M2NearestFinder.scala`, mirrored in `mx_fp_math.h::*_to_fixed_point`). Those
# conversions apply a WIDTH MASK, so large magnitudes wrap and alias onto small ones:
#
#     E3M2:  4.0 -> 64    20.0 -> 64  (collide)     16.0 -> 0  (collides with ZERO)
#
# A codebook holding an aliased entry is not merely coarse, it is *wrong*: values near zero get
# assigned to the aliasing large entry and decode to something enormous. Measured on a 2-stage FP6
# chain, a book containing -16 and 20 gave 77% error where a filtered one gives ~26%.
#
# So the value space a codebook may draw from is not "everything the format represents" -- it is
# everything the FINDER can distinguish. That is what :func:`codebook_values` returns.

def _fixed_point(fmt: mxformats.MxFormat) -> "callable":
    """The finder's code -> signed fixed-point magnitude, per format.

    Transcribed from ``mx_fp_math.h`` (``fp6_to_fixed_point``, ``fp6_e2m3_to_fixed_point``,
    ``fp8_e5m2_to_fixed_point``, ``fp8_e4m3_to_fixed_point``), which mirror the RTL finders. The
    masks are the point of this function -- dropping them would hide the aliasing it exists to find.
    """
    def e3m2(v):                                   # 1s|3e|2m, sigW 3, 9-bit shift, 8-bit magnitude
        v &= 0x3F
        sign, exp, mant = (v >> 5) & 1, (v >> 2) & 0x7, v & 0x3
        if exp == 0 and mant == 0:
            return 0
        s_exp = -2 if exp == 0 else exp - 3
        sig = (0 if exp == 0 else 4) | mant
        mag = ((sig << ((s_exp + 2) & 0b111)) & 0x1FF) & 0xFF
        return -mag if sign else mag

    def e2m3(v):                                   # exact: value * 8, no mask
        v &= 0x3F
        sign, exp, mant = (v >> 5) & 1, (v >> 3) & 0x3, v & 0x7
        mag = mant if exp == 0 else ((8 + mant) << (exp - 1))
        return -mag if sign else mag

    def e5m2(v):                                   # sigW 3, shiftW 5, fixedW 32
        v &= 0xFF
        sign, exp, mant = (v >> 7) & 1, (v >> 2) & 0x1F, v & 0x3
        if exp == 0 and mant == 0:
            return 0
        sig = (0 if exp == 0 else 4) | mant
        s_exp = (1 - 15) if exp == 0 else exp - 15
        mag = (sig << ((s_exp + 14) & 0x1F)) & 0xFFFFFFFF
        return -mag if sign else mag

    def e4m3(v):                                   # sigW 4, shiftW 4, fixedW 18
        v &= 0xFF
        sign, exp, mant = (v >> 7) & 1, (v >> 3) & 0xF, v & 0x7
        if exp == 0 and mant == 0:
            return 0
        sig = (0 if exp == 0 else 8) | mant
        s_exp = (1 - 7) if exp == 0 else exp - 7
        mag = (sig << ((s_exp + 6) & 0xF)) & 0x3FFFF
        return -mag if sign else mag

    return {"fp6_e3m2": e3m2, "fp6_e2m3": e2m3, "fp8_e5m2": e5m2,
            "fp8_e4m3": e4m3, "fp8_e4m3_quad": e4m3}[fmt.name]


#: Width the finder masks each |difference| to, per format, mirroring the `*NearestFinder.scala`
#: comparators. E2M3's fixed-point is exact (value * 8) and needs none.
_DIFF_MASK = {"fp6_e3m2": 0x1FF, "fp6_e2m3": None, "fp8_e5m2": 0x1FFFFFFFF,
              "fp8_e4m3": 0x7FFFF, "fp8_e4m3_quad": 0x7FFFF}


def finder_indices(codes: np.ndarray, books: np.ndarray, *, fmt: mxformats.MxFormat,
                   axis: str = "row", g: int = mxformats.LUT_GRANULARITY) -> np.ndarray:
    """Index assignment **as the HARDWARE does it** — nearest in fixed-point, ties to lower index.

    Distinct from :func:`assign_indices`, and the distinction is not pedantry:

    * For an **operand**, WE choose the index on the host and send it; the hardware only looks the
      entry up. Nearest-by-value is therefore correct, and is what `assign_indices` does.
    * For a **requant output**, the HARDWARE chooses, with `fp6e3m2_nearest_finder` and friends —
      a comparison in a fixed-point domain with a width mask, not a float comparison. A model that
      picks by value disagrees wherever the two orderings differ, which measured 27/4096 matching
      on a 2-stage FP6 chain.

    Takes ELEMENT CODES, not values, because that is what the finder compares — the requantizer
    has already rounded to the element format by the time the projection happens
    (`matrix_mx_requantize` only divides by the block scale; `tensor_to_custom_fp_codes` does the
    rounding, as that model's own comment says).

    Mirrors `mx_fp_math.h::*_to_fixed_point` plus the finder's `Mux(d1 <= d2, i1, i2)` tie-break.
    """
    fp = _fixed_point(fmt)
    mask = _DIFF_MASK[fmt.name]
    encode = _value_to_code(fmt)
    codes = np.asarray(codes, dtype=np.uint8)
    out = np.zeros(codes.shape, dtype=np.uint8)

    for grp in range(books.shape[0]):
        lut_fx = [fp(encode(v)) for v in books[grp].tolist()]
        sl = slice(grp << g, (grp + 1) << g)
        block = codes[sl, :] if axis == "row" else codes[:, sl]
        idx = np.empty(block.shape, dtype=np.uint8)
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                fx = fp(int(block[i, j]))
                best, bd = 0, None
                for e, lf in enumerate(lut_fx):
                    d = abs(fx - lf)
                    if mask is not None:
                        d &= mask
                    if bd is None or d < bd:      # strictly <, so ties keep the LOWER index
                        best, bd = e, d
                idx[i, j] = best
        if axis == "row":
            out[sl, :] = idx
        else:
            out[:, sl] = idx
    return out


def codebook_values(fmt: mxformats.MxFormat) -> np.ndarray:
    """The values a codebook may hold: distinct, finite, and **unambiguous to the finder**.

    Derived from the format's own decoder, so it cannot disagree with what the mesh will do with an
    index -- then filtered so that no two entries share a fixed-point magnitude and no entry aliases
    a smaller one. Ties are resolved toward the SMALLER magnitude, since that is the one whose
    fixed-point is faithful.
    """
    decode = DECODERS.get(fmt.name)
    if decode is None:
        raise mxformats.MxFormatError(
            f"no element decoder for {fmt.name!r}; add one to app/mxwire.DECODERS")
    codes = np.arange(1 << fmt.entry_bits, dtype=np.uint8)
    vals = decode(codes)
    fp = _fixed_point(fmt)

    seen: dict[int, float] = {}
    for c, v in zip(codes.tolist(), vals.tolist()):
        if not np.isfinite(v):
            continue
        key = fp(c)
        if key not in seen or abs(v) < abs(seen[key]):
            seen[key] = v
    return np.unique(np.array(sorted(seen.values()), dtype=np.float32))


def _kmeans_1d(values: np.ndarray, k: int) -> np.ndarray:
    """Deterministic weighted 1-D k-means over the DISTINCT values present.

    The support is tiny (an FP6 codebook has 64 members), so this collapses to a weighted Lloyd over
    distinct values and converges in a few passes.

    Quantile init spreads seeds by MASS rather than by range: most of a real activation block sits
    near zero, and a range-uniform init would spend most of its 16 slots on the sparse tail.
    """
    uniq, counts = np.unique(values, return_counts=True)
    if uniq.size <= k:
        return uniq
    cdf = np.cumsum(counts) / counts.sum()
    probes = (np.arange(k) + 0.5) / k
    centers = np.unique(uniq[np.searchsorted(cdf, probes).clip(0, uniq.size - 1)])
    for _ in range(50):
        lab = np.abs(uniq[:, None] - centers[None, :]).argmin(axis=1)
        new = np.unique(np.array([
            (uniq[lab == c] * counts[lab == c]).sum() / counts[lab == c].sum()
            if np.any(lab == c) else centers[c]
            for c in range(centers.size)]))
        if new.size == centers.size and np.allclose(new, centers):
            break
        centers = new
    return centers


def build_codebooks(P: np.ndarray, *, axis: str, fmt: mxformats.MxFormat,
                    g: int = mxformats.LUT_GRANULARITY) -> np.ndarray:
    """``[n_groups][16]`` codebook VALUES for an already-MX-quantized tile.

    ``axis="row"`` groups rows (the A side), ``axis="col"`` groups columns (the B side) — matching
    how the hardware indexes them.

    Entries are deduplicated and padded to exactly 16 with unused codes, smallest magnitude first.
    A duplicate slot would make index assignment ambiguous: two indices decoding to the same value
    means the encoder's choice between them is arbitrary and unreproducible.
    """
    P = np.asarray(P, dtype=np.float32)
    cb = codebook_values(fmt)
    span = P.shape[0] if axis == "row" else P.shape[1]
    if span % (1 << g):
        raise ValueError(f"axis={axis!r} has {span} entries, not a multiple of 2**G = {1 << g}")

    out = np.zeros((span >> g, LUT_SIZE), dtype=np.float32)
    for grp in range(span >> g):
        sl = slice(grp << g, (grp + 1) << g)
        vals = (P[sl, :] if axis == "row" else P[:, sl]).ravel()
        centers = _kmeans_1d(vals, LUT_SIZE)
        snapped = cb[np.abs(centers[:, None] - cb[None, :]).argmin(axis=1)]
        entries = list(dict.fromkeys(snapped.tolist()))          # order-preserving dedupe
        for c in sorted(cb.tolist(), key=abs):                   # pad with unused codes
            if len(entries) >= LUT_SIZE:
                break
            if c not in entries:
                entries.append(c)
        out[grp] = sorted(entries[:LUT_SIZE])
    return out


def assign_indices(P: np.ndarray, books: np.ndarray, *, axis: str,
                   g: int = mxformats.LUT_GRANULARITY) -> np.ndarray:
    """Nearest-entry index for every element, against ITS group's codebook. ``[R][C]`` of 0..15."""
    P = np.asarray(P, dtype=np.float32)
    idx = np.zeros(P.shape, dtype=np.uint8)
    for grp in range(books.shape[0]):
        sl = slice(grp << g, (grp + 1) << g)
        block = P[sl, :] if axis == "row" else P[:, sl]
        near = np.abs(block[..., None] - books[grp][None, None, :]).argmin(axis=-1).astype(np.uint8)
        if axis == "row":
            idx[sl, :] = near
        else:
            idx[:, sl] = near
    return idx


def pack_codebooks(books: np.ndarray, *, fmt: mxformats.MxFormat) -> np.ndarray:
    """``[n_groups][words]`` uint32, the wire form MX_LOAD_LUT reads.

    16 entries of ``entry_bits``, LE-packed: entry *i* occupies bits ``[i*e, i*e+e)`` of the little-
    endian bit stream, so 6-bit codebooks take 3 words (96 bits) and 8-bit ones take 4 (128).
    ``gemmini.h:68-76``.
    """
    e = fmt.entry_bits
    words = mxformats.lut_words(e)
    encode = _value_to_code(fmt)
    out = np.zeros((books.shape[0], words), dtype=np.uint32)
    for grp, row in enumerate(books):
        stream = 0
        for i, v in enumerate(row):
            stream |= (int(encode(v)) & ((1 << e) - 1)) << (i * e)
        for w in range(words):
            out[grp, w] = (stream >> (32 * w)) & 0xFFFFFFFF
    return out


def unpack_codebooks(packed: np.ndarray, *, fmt: mxformats.MxFormat) -> np.ndarray:
    """Inverse of :func:`pack_codebooks`: ``[n_groups][words]`` uint32 -> ``[n_groups][16]`` values.

    Mirrors ``mx_fp_math.h``'s ``unpack_lut_96bit`` / ``unpack_lut_128bit`` -- entry *i* occupies
    bits ``[i*e, i*e+e)`` of the little-endian stream.
    """
    e = fmt.entry_bits
    decode = DECODERS[fmt.name]
    packed = np.ascontiguousarray(packed, dtype=np.uint32)
    out = np.zeros((packed.shape[0], LUT_SIZE), dtype=np.float32)
    for g, words in enumerate(packed):
        stream = 0
        for w, v in enumerate(words.tolist()):
            stream |= int(v) << (32 * w)
        codes = np.array([(stream >> (i * e)) & ((1 << e) - 1) for i in range(LUT_SIZE)], np.uint8)
        out[g] = decode(codes)
    return out


def _value_to_code(fmt: mxformats.MxFormat):
    """Exact value -> element code, by inverting the decoder. Raises rather than rounding."""
    decode = DECODERS[fmt.name]
    codes = np.arange(1 << fmt.entry_bits, dtype=np.uint8)
    table = {}
    for c, v in zip(codes.tolist(), decode(codes).tolist()):
        table.setdefault(int(np.float32(v).view(np.uint32)), int(c))

    def encode(v: float) -> int:
        key = int(np.float32(v).view(np.uint32))
        if key not in table:
            raise mxformats.MxFormatError(
                f"{v!r} is not an exact {fmt.name} value, so it cannot be a codebook entry")
        return table[key]

    return encode
