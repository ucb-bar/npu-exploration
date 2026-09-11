"""The quantizer is MXQuant, and it agrees with the baremetal tests — byte for byte.

Two claims, checked against two independent artifacts rather than asserted:

1. **It is MXQuant.** ``app/mxq_golden.quantize_operand`` reaches ``quantize_mx_block32`` by import,
   never by transcription, so it cannot drift. The wire encoding on top of it is lossless by
   construction and ``golden()`` asserts that on every call.

2. **It is what the baremetal examples run.** The shipped
   ``gemmini-rocc-tests/include/matmul_fp8_*.h`` headers contain the exact operand bytes spike is
   tested against. This re-quantizes the same real TinyLlama tensors those headers were built from
   and requires **byte equality** with ``A_in`` / ``A_scales_row`` / ``B_in`` / ``B_scales_col``.

Claim 2 is the load-bearing one. It is what makes "the ELF our compiler emits" and "the ELF in
gemmini-rocc-tests" numerically the same program, which is the premise of the whole port
(``planning/merlin_glue_port_plan.md`` D1/D3).

Needs the captured tiles (``python3 -m app.capture_llama_tiles``); skips cleanly without them.

    .venv/bin/python tests/selftest_quantizer.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import mxformats                                    # noqa: E402
from app.mxq_golden import MXQ_ROOT, quantize_operand        # noqa: E402

#: The baremetal reference. D1: we depend on this tree for `gemmini.h` and read it for provenance;
#: nothing here is built against it.
ROCC = REPO.parent / "software" / "gemmini-rocc-tests"
DATA = MXQ_ROOT / "end_to_end_linear" / "systolic_simulation" / "data_evalrun_512"

#: (header, M, K, N, layer, projection) — mirrors gen_matmul_llama.SHAPES for the FP8 entries.
#: Every one of these headers backs at least one PASSing spike test.
CASES = [
    ("matmul_fp8_32x32x32.h",   32,  32,  32, "layer0", "mlp.gate_proj"),
    ("matmul_fp8_64x64.h",      64,  64,  64, "layer0", "mlp.gate_proj"),
    ("matmul_fp8_96x32x32.h",   96,  32,  32, "layer0", "mlp.up_proj"),
    ("matmul_fp8_64x96x64.h",   64,  64,  96, "layer0", "attn.q_proj"),
    ("matmul_fp8_96x96x64.h",   96,  64,  96, "layer0", "attn.o_proj"),
    ("matmul_fp8_128x128.h",   128, 128, 128, "layer0", "mlp.down_proj"),
    ("matmul_fp8_128x128x256.h", 128, 256, 128, "layer1", "mlp.gate_proj"),
]


def parse_array(header: str, name: str) -> np.ndarray:
    """Pull one `static const uintN_t NAME[..][..] = { ... };` out of a C header as an array."""
    m = re.search(rf"static const uint(\d+)_t {re.escape(name)}\s*\[([^\]]*)\]\s*\[([^\]]*)\]"
                  r"\s*=\s*\{(.*?)\n\};", header, re.S)
    if not m:
        raise KeyError(f"{name} not found in header")
    width, body = int(m.group(1)), m.group(4)
    vals = [int(t, 0) for t in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", body)]
    dt = {8: np.uint8, 16: np.uint16, 32: np.uint32}[width]
    return np.array(vals, dtype=dt)


def load_pair(M: int, K: int, N: int, layer: str, proj: str):
    """The same slice `gen_matmul_llama.load_pair` takes: one contiguous real (A, W) pair."""
    d = DATA / layer / proj
    with np.load(d / "A_square.npz") as z:
        A_full = z["data"].astype(np.float32)
    with np.load(d / "W_square.npz") as z:
        W_full = z["data"].astype(np.float32)
    A = np.ascontiguousarray(A_full[:M, :K])
    B = np.ascontiguousarray(W_full[:N, :K].T)        # [out][in] -> [K][N]
    return A, B


def main() -> int:
    if not DATA.is_dir():
        print(f"SKIP: captured tiles missing at {DATA}\n"
              f"      run: .venv/bin/python3 -m app.capture_llama_tiles")
        return 0
    if not (ROCC / "include").is_dir():
        print(f"SKIP: baremetal reference missing at {ROCC}")
        return 0

    checks = fails = 0

    # --- 1. the format table is self-consistent ------------------------------------------------
    for key, f in mxformats.FORMATS.items():
        assert f.name == key, f"{key}: name/key mismatch"
        assert f.bits in (4, 8), f"{key}: odd wire width {f.bits}"
        assert (f.entry_bits is not None) == f.lut, f"{key}: entry_bits must be set iff LUT-indexed"
        assert f.out_requant in ("mxquant", "model"), f"{key}: bad out_requant"
        checks += 1
    print(f"format table          {len(mxformats.FORMATS)} formats, "
          f"{sum(f.proven for f in mxformats.FORMATS.values())} proven")

    # --- 2. an unproven format fails closed ----------------------------------------------------
    # Picked dynamically: formats become proven as Step 5 lands them, and a hardcoded name here
    # would turn "we proved another format" into a test failure.
    unproven = next((f.name for f in mxformats.FORMATS.values() if not f.proven), None)
    if unproven is None:
        print("  (every format is proven -- nothing left to fail closed)")
    else:
        try:
            mxformats.get(unproven, where="selftest")
        except mxformats.MxFormatError:
            checks += 1
        else:
            print(f"FAIL: unproven format {unproven} was accepted"); fails += 1

    # --- 3. the elaboration gate actually gates ------------------------------------------------
    # fp4 on an E4M3-only build: spike would run it, the elaborated hardware could not.
    try:
        mxformats.check_elaboration(["fp4_e2m1"], "MxGemminiRocketConfig")
    except mxformats.MxFormatError:
        checks += 1
    else:
        print("FAIL: fp4 was accepted on an E4M3-only elaboration"); fails += 1
    mxformats.check_elaboration(["fp8_e4m3"], "MxGemminiRocketConfig")
    mxformats.check_elaboration(["fp4_e2m1"], "MxAllGemminiRocketConfig")
    checks += 2

    # --- 4. byte equality with the shipped baremetal headers -----------------------------------
    print(f"\n{'header':28s} {'shape':14s} {'A codes':>10s} {'A scales':>10s} "
          f"{'B codes':>10s} {'B scales':>10s}")
    for header, M, K, N, layer, proj in CASES:
        p = ROCC / "include" / header
        if not p.exists() or not (DATA / layer / proj).is_dir():
            print(f"{header:28s} {'(missing)':14s}")
            continue
        text = p.read_text()
        A, B = load_pair(M, K, N, layer, proj)
        a_codes, a_scales, _ = quantize_operand(A, side="a")
        b_codes, b_scales, _ = quantize_operand(B, side="b")

        want = {
            "A codes":  (a_codes.ravel(),  parse_array(text, "A_in")),
            "A scales": (a_scales.ravel(), parse_array(text, "A_scales_row")),
            "B codes":  (b_codes.ravel(),  parse_array(text, "B_in")),
            "B scales": (b_scales.ravel(), parse_array(text, "B_scales_col")),
        }
        cells = []
        for label, (mine, theirs) in want.items():
            if mine.shape != theirs.shape:
                cells.append(f"SHAPE {mine.size}/{theirs.size}"); fails += 1
                continue
            d = int((mine != theirs).sum())
            cells.append("== " if d == 0 else f"{d} DIFF")
            checks += 1
            if d:
                fails += 1
        print(f"{header:28s} {f'{M}x{K}x{N}':14s} " + " ".join(f"{c:>10s}" for c in cells))

    print(f"\n{checks} checks, {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
