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

The rule itself is ``mxq.lut`` (microscaling-quant), shared with the perplexity path; this module keeps
the compiler's side: the settings it takes from the recipes, and the wire form (pack / unpack / encode).

Ported from ``gemmini-rocc-tests/llama_operands.py`` (D1: the workload side moves into this repo).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import models  # noqa: F401  -- puts the mxq submodule on sys.path
from compiler import formats
from compiler.wire import DECODERS
from mxq import lut as mxq_lut

#: Entries per codebook: a fact of the wire, whose index is a nibble (formats.FORMATS ``bits=4``).
#: config.recipe.check refuses a build whose mx.lut.raddrWidth says otherwise.
LUT_SIZE = 16


@dataclass(frozen=True)
class Settings:
    """What shapes a codebook, from the two recipes (``config.recipe.lut_settings``); no defaults.

    group      G: one codebook per ``2**G`` rows of A / columns of B / rows of C
    max_iters  the k-means fit's most Lloyd passes
    """
    group: int
    max_iters: int


# --- the rule lives in mxq --------------------------------------------------------------------------
#
# Which values a codebook may hold (the finder's fixed point aliases large magnitudes onto small ones,
# so only what it can tell apart), how a group's 16 entries are fitted (weighted k-means on the distinct
# codes, quantile seeds, snapped, padded), the host's nearest-entry pick and the hardware finder are all
# ``mxq.lut`` -- the same code the perplexity path runs, so the two paths cannot drift. mxq takes a K x n
# tensor and groups its columns; ``axis="row"`` (the A side, and C) is the transpose of that.
# tests/selftest_codebook_mxq.py holds mxq.lut to this module's former numpy rule
# (tests/fixtures/codebook_numpy.py), bit for bit, on all four formats and G = 0, 1, 2.

def _kn(X: np.ndarray, axis: str) -> torch.Tensor:
    """The tile as mxq.lut's K x n: one table per 2**G columns."""
    if axis not in ("row", "col"):
        raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")
    return torch.from_numpy(np.ascontiguousarray(X.T if axis == "row" else X))


def _back(T: torch.Tensor, axis: str) -> np.ndarray:
    a = T.numpy()
    return np.ascontiguousarray(a.T if axis == "row" else a)


def codebook_values(fmt: formats.MxFormat) -> np.ndarray:
    """The values a codebook may hold: distinct, finite, and unambiguous to the finder (``mxq.lut.values``)."""
    return mxq_lut.values(fmt.mxq).numpy()


def build_codebooks(P: np.ndarray, *, axis: str, fmt: formats.MxFormat, lut: Settings) -> np.ndarray:
    """``[n_groups][16]`` codebook VALUES for an already-MX-quantized tile (``mxq.lut.tables``).

    ``axis="row"`` groups rows (the A side), ``axis="col"`` groups columns (the B side) -- matching
    how the hardware indexes them. Each row is 16 distinct entries, ascending.
    """
    return mxq_lut.tables(_kn(np.asarray(P, dtype=np.float32), axis), fmt.mxq, group=lut.group,
                          max_iters=lut.max_iters).numpy()


def assign_indices(P: np.ndarray, books: np.ndarray, *, axis: str, g: int) -> np.ndarray:
    """Nearest-entry index for every element, against ITS group's codebook (``mxq.lut.pick``). ``[R][C]`` of 0..15."""
    I = mxq_lut.pick(_kn(np.asarray(P, dtype=np.float32), axis), torch.from_numpy(np.asarray(books, np.float32)), group=g)
    return _back(I, axis).astype(np.uint8)


def finder_indices(codes: np.ndarray, books: np.ndarray, *, fmt: formats.MxFormat,
                   axis: str, g: int) -> np.ndarray:
    """Index assignment **as the HARDWARE does it** (``mxq.lut.finder``) -- nearest in fixed point, ties to
    the lower index.

    Distinct from :func:`assign_indices`: for an operand the host chooses the index and the hardware only
    looks the entry up, so nearest-by-value is right; for a requant output the hardware's finder chooses,
    comparing masked fixed-point magnitudes, and a by-value pick disagrees wherever the two orderings differ
    (measured 27/4096 matching on a 2-stage FP6 chain). Takes ELEMENT CODES, because that is what the
    finder compares.
    """
    I = mxq_lut.finder(_kn(np.asarray(codes, dtype=np.uint8).astype(np.int64), axis),
                   torch.from_numpy(np.asarray(books, np.float32)), fmt.mxq, group=g)
    return _back(I, axis).astype(np.uint8)


def pack_codebooks(books: np.ndarray, *, fmt: formats.MxFormat) -> np.ndarray:
    """``[n_groups][words]`` uint32, the wire form MX_LOAD_LUT reads.

    16 entries of ``entry_bits``, LE-packed: entry *i* occupies bits ``[i*e, i*e+e)`` of the little-
    endian bit stream, so 6-bit codebooks take 3 words (96 bits) and 8-bit ones take 4 (128).
    ``gemmini.h:68-76``.
    """
    e = fmt.entry_bits
    words = formats.lut_words(e)
    encode = _value_to_code(fmt)
    out = np.zeros((books.shape[0], words), dtype=np.uint32)
    for grp, row in enumerate(books):
        stream = 0
        for i, v in enumerate(row):
            stream |= (int(encode(v)) & ((1 << e) - 1)) << (i * e)
        for w in range(words):
            out[grp, w] = (stream >> (32 * w)) & 0xFFFFFFFF
    return out


def unpack_codebooks(packed: np.ndarray, *, fmt: formats.MxFormat) -> np.ndarray:
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


def _value_to_code(fmt: formats.MxFormat):
    """Exact value -> element code, by inverting the decoder. Raises rather than rounding."""
    decode = DECODERS[fmt.name]
    codes = np.arange(1 << fmt.entry_bits, dtype=np.uint8)
    table = {}
    for c, v in zip(codes.tolist(), decode(codes).tolist()):
        table.setdefault(int(np.float32(v).view(np.uint32)), int(c))

    def encode(v: float) -> int:
        key = int(np.float32(v).view(np.uint32))
        if key not in table:
            raise formats.MxFormatError(
                f"{v!r} is not an exact {fmt.name} value, so it cannot be a codebook entry")
        return table[key]

    return encode
