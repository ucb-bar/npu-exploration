"""The chained requantizer, against a C oracle compiled from the datapath's own header.

Every other tier in this repo compares a NUMBER. This one compares a PROCEDURE, and it exists
because comparing numbers was not enough: a model that reproduced the finder exactly and got the
element rounding generically wrong still matched 27/4096 on a two-stage FP6 chain, and the only
visible symptom was a bad final answer. There was nothing to point at.

So ``tests/oracle/requant_oracle.cc`` lifts ``gemmini.cc:1445-1509`` verbatim, links it against
``libgemmini/mx_fp_math.h`` — the real encoders, the real finders — and exposes the three
intermediate results the Python cannot otherwise see:

    scale codes  ->  element codes  ->  codebook indices

A disagreement therefore names the STEP, not just the kernel. That is the whole value: the same
three-step comparison is what a new format's encoder will be checked with, and
``app.mxwire.encode_requant`` refuses a format it has no entry for rather than rounding it by a
generic rule, so a format cannot silently arrive here unmodelled.

The oracle is compiled on demand with the host ``g++``; if there is none, the test SKIPS rather
than passing vacuously — a self-test that cannot run must not report green.

    .venv/bin/python tests/selftest_requant.py
"""
from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app import mxformats, mxlut, mxwire            # noqa: E402
from app.mxq_golden import requantize_chained       # noqa: E402

# The chipyard tree ($MERLIN_CHIPYARD) is the canonical source of the header; the old
# sibling software/ layout stays as a fallback for checkouts that used it.
from config.build_spike import gemmini_root          # noqa: E402

LIBGEMMINI = gemmini_root() / "software" / "libgemmini"
if not (LIBGEMMINI / "mx_fp_math.h").exists():
    LIBGEMMINI = REPO.parent / "software" / "libgemmini"
SRC = Path(__file__).parent / "oracle" / "requant_oracle.cc"
BIN = REPO / "out" / "requant_oracle"

#: The oracle's format selector, matching the ``(mx_out_fmt, altfmt)`` decode in ``gemmini.cc``.
#: ``fp8_e4m3`` is 4, the DIRECT 8-bit path: it has no codebook and no finder, so its "index" is
#: just its element code. It was added on 2026-09-10 -- until then it was the one format whose
#: chained requant did not go through a device transcription, and that is exactly why it was the
#: one format the RNE header change broke without a self-test catching it first.
FMT_ID = {"fp8_e5m2": 0, "fp8_e4m3_quad": 1, "fp6_e2m3": 2, "fp6_e3m2": 3, "fp8_e4m3": 4}

#: Formats whose requant output is a 4-bit index into a codebook. The rest write an element code.
LUT_FMTS = ("fp8_e5m2", "fp8_e4m3_quad", "fp6_e2m3", "fp6_e3m2")


def build() -> Path | None:
    """Compile the oracle, or ``None`` if this host cannot."""
    if not (LIBGEMMINI / "mx_fp_math.h").exists():
        return None
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        return None
    BIN.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([cxx, "-O2", "-o", str(BIN), str(SRC), f"-I{LIBGEMMINI}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"the oracle no longer builds against mx_fp_math.h:\n{r.stderr}")
    return BIN


def oracle(binary: Path, C: np.ndarray, book_codes: np.ndarray, dtype: str, g: int):
    """``(scale_codes, element_codes, indices)`` straight out of the datapath's own C."""
    M, N = C.shape
    buf = struct.pack("<4i", M, N, FMT_ID[dtype], g)
    buf += np.ascontiguousarray(C, np.float32).tobytes()
    buf += np.ascontiguousarray(book_codes, np.uint8).tobytes()
    out = subprocess.run([str(binary)], input=buf, capture_output=True, check=True).stdout
    nb = N // mxwire.BLOCK
    return (np.frombuffer(out[:M * nb], np.uint8).reshape(M, nb),
            np.frombuffer(out[M * nb:M * nb + M * N], np.uint8).reshape(M, N),
            np.frombuffer(out[M * nb + M * N:], np.uint8).reshape(M, N))


def python_steps(C: np.ndarray, book_vals: np.ndarray, dtype: str, g: int):
    """The same three steps, from the modules the grader actually uses."""
    f = mxformats.get(dtype, where="selftest_requant")
    C = mxwire.bf16_bits_to_float(mxwire.float_to_bf16_bits(C))
    M, N = C.shape
    blocks = C.reshape(M, N // mxwire.BLOCK, mxwire.BLOCK)
    max_abs = np.abs(blocks).max(axis=2)
    if dtype in LUT_FMTS:
        with np.errstate(divide="ignore"):
            max_exp = np.floor(np.log2(max_abs.astype(np.float32))).astype(np.int64)
        sc = np.clip(max_exp + mxwire.E8M0_BIAS, 0, 254).astype(np.uint8)
        sc[max_abs == 0.0] = 0
        scale = np.exp2((sc.astype(np.int64) - mxwire.E8M0_BIAS).astype(np.float64)).astype(np.float32)
        elem = mxwire.encode_requant((blocks / scale[:, :, None]).reshape(M, N), dtype=dtype)
        return sc, elem, mxlut.finder_indices(elem, book_vals, fmt=f, axis="row", g=g)

    # The direct 8-bit path: epsilon-clamped scale, no bf16 pre-round, no finder.
    amax = np.maximum(max_abs, np.finfo(np.float32).eps)
    sc = np.clip(np.floor(np.log2(amax)).astype(np.int64) + mxwire.E8M0_BIAS, 0, 254).astype(np.uint8)
    scale = np.exp2((sc.astype(np.int64) - mxwire.E8M0_BIAS).astype(np.float64)).astype(np.float32)
    scaled = (blocks / scale[:, :, None]).reshape(M, N)
    elem = np.array([mxwire.fp8_e4m3_encode_f32(v) for v in scaled.ravel().tolist()],
                    dtype=np.uint8).reshape(M, N)
    return sc, elem, elem


def books_for(f, rng, nbooks: int):
    """A codebook drawn from the values the finder can actually distinguish."""
    cand = mxlut.codebook_values(f)
    vals = np.stack([np.sort(rng.choice(cand, mxlut.LUT_SIZE, replace=False))
                     for _ in range(nbooks)])
    enc = mxlut._value_to_code(f)
    codes = np.array([[enc(v) for v in row] for row in vals.tolist()], np.uint8)
    return vals, codes


def main() -> int:
    binary = build()
    if binary is None:
        print("SKIP -- no C++ compiler, or libgemmini/mx_fp_math.h is not present.")
        print("       This test compares against the hardware header; it cannot be faked.")
        return 0

    rng = np.random.default_rng(0)
    g = mxformats.LUT_GRANULARITY
    failures = 0
    # A spread of magnitudes on purpose: the divergences this test was built to catch live at the
    # edges -- values that round to zero (signed zero), and values that sit exactly on a tie.
    for scale in (0.25, 4.0, 64.0):
        for dtype in FMT_ID:
            f = mxformats.get(dtype, where="selftest_requant")
            M, N = 64, 64
            C = (rng.standard_normal((M, N)) * scale).astype(np.float32)
            # The direct path has no codebook; the oracle still reads a book block, so send zeros.
            vals, codes = (books_for(f, rng, ((M - 1) >> g) + 1) if dtype in LUT_FMTS
                           else (np.zeros((((M - 1) >> g) + 1, mxlut.LUT_SIZE), np.float32),
                                 np.zeros((((M - 1) >> g) + 1, mxlut.LUT_SIZE), np.uint8)))

            osc, oel, oix = oracle(binary, C, codes, dtype, g)
            psc, pel, pix = python_steps(C, vals, dtype, g)
            steps = {"scale": (osc, psc), "element": (oel, pel), "index": (oix, pix)}
            bad = {k: int((a != b).sum()) for k, (a, b) in steps.items() if (a != b).any()}
            tag = f"{dtype:16s} x{scale:<6g}"
            if bad:
                failures += 1
                print(f"FAIL {tag} {bad}")
                for k, (a, b) in steps.items():
                    if (a != b).any():
                        r, c = np.argwhere(a != b)[0]
                        print(f"       first {k} divergence [{r},{c}]: oracle {a[r,c]} python {b[r,c]}")
            else:
                print(f"ok   {tag} scale/element/index all identical")

            # And the function the grader calls, end to end: the value the NEXT matmul multiplies
            # is the book entry the oracle's index names -- or, for the direct path, the decode of
            # the oracle's own element code.
            if dtype in LUT_FMTS:
                P, _ = requantize_chained(C, dtype=dtype, books=mxlut.pack_codebooks(vals, fmt=f))
                want = np.take_along_axis(vals[np.arange(M) >> g], oix.astype(np.intp), axis=1)
            else:
                P, _ = requantize_chained(C, dtype=dtype)
                want = mxwire.DECODERS[dtype](oix)
            if not np.array_equal(P.numpy().T, want.astype(np.float32)):
                failures += 1
                print(f"FAIL {tag} requantize_chained disagrees with the oracle's own indices")

    print(f"\n{'FAILED' if failures else 'PASS'}: {len(FMT_ID) * 3} cases, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
