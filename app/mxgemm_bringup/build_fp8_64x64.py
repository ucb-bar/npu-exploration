"""Bring-up experiment: the fp8 64x64x64 MX matmul, end to end on spike.

Builds a merlin command buffer for one weight-stationary MX GEMM, hands it to the
``mx_gemmini_rocket`` backend, and reports what came back.

This is the APP layer: it owns the operands, the shape, and the choice of experiment. The compiler
backend receives a command buffer and knows none of that.

Operands come from the gemmini-rocc-tests data header whose golden is already trusted on spike (plan
section 6) — the point is to prove the emitted kernel reproduces it. From the PyTorch step the same
command buffer is built from tensors instead, and nothing downstream changes.

    python app/mxgemm_bringup/build_fp8_64x64.py            # emit, build, run, report
    python app/mxgemm_bringup/build_fp8_64x64.py --emit-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from header_operands import load_mx_operands

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "compiler" / "targets" / "mx_gemmini_rocket"
DEFAULT_HEADER = (REPO.parent / "software" / "gemmini-rocc-tests" / "include"
                  / "matmul_fp8_64x64.h")
DEFAULT_WORKDIR = REPO / "out" / "build" / "fp8_64x64"

# The backend is a package (merlin loads it the same way, as merlin._oot_backends.<name>), so import
# it as one rather than putting its interior on sys.path — its siblings use relative imports.
sys.path.insert(0, str(TARGET))
import backend as mx  # noqa: E402


def build_command_buffer(m: int, n: int, k: int, mx_operands: dict) -> dict:
    """One RES_PACK -> MATMUL_RESIDENT -> COMMIT -> EVICT buffer for an mxfp8 GEMM."""
    return {
        "abi_version": "0.1",
        "target": "mx_gemmini_rocket",
        "backend": "spike_mx_gemmini",
        "tensors": {
            "A0": {"dtype": "mxfp8", "shape": [m, k]},
            "W": {"dtype": "mxfp8", "shape": [k, n]},
            "Y0": {"dtype": "bf16", "shape": [m, n]},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"}, "attributes": {}},
            {"opcode": "EVICT", "operands": {"handle": "W_res"}},
        ],
        "params": {},
        "resources": {"buffers": [], "handles": ["W_res", "acc0"]},
        "metrics_requested": ["cycles"],
        # MX side-channel: raw operand codes + E8M0 block scales. The decoded tensor table above
        # cannot carry these — see mxgemm_emit._mx_operands.
        "mx_operands": mx_operands,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--header", type=Path, default=DEFAULT_HEADER,
                    help="gemmini-rocc-tests data header supplying the operands + golden")
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--emit-only", action="store_true", help="write the C and stop")
    a = ap.parse_args()

    ops = load_mx_operands(a.header)
    m, k = len(ops["a_codes"]), len(ops["a_codes"][0])
    n = len(ops["b_codes"][0])
    print(f"operands  A[{m}][{k}] B[{k}][{n}]  scales A{len(ops['a_scales'])}x{m} "
          f"B{len(ops['b_scales'])}x{n}   <- {a.header.name}")

    cb = build_command_buffer(m, n, k, ops)

    if a.emit_only:
        a.workdir.mkdir(parents=True, exist_ok=True)
        out = a.workdir / "main.c"
        out.write_text(mx.generate_driver(cb), encoding="utf-8")
        print(f"emitted   {len(out.read_text().splitlines())} lines -> {out}")
        return 0

    if not mx.available("spike"):
        print("spike oracle NOT AVAILABLE — source the chipyard env.sh (needs $RISCV) and make "
              "sure software/libgemmini/libgemmini.so is built.", file=sys.stderr)
        return 2

    res = mx.run_command_buffer(cb, workdir=a.workdir)
    got = res["outputs"].get("Y0")
    oracle = res["oracle"]
    print(f"elf       {res['elf']}")
    print(f"oracle    {oracle['simulator']} ({oracle['kind']}, "
          f"derived_from_rtl={oracle['derived_from_rtl']})")
    print(f"OUT Y0    {len(got)}x{len(got[0])} bf16 patterns, first row[:4]={got[0][:4]}")
    print(f"METRIC    {res['metrics']}")

    # The kernel also self-checks against the baked golden; surface its verdict.
    verdict = [ln for ln in res["console"].splitlines() if "matmul test" in ln]
    for ln in verdict:
        print(f"selfcheck {ln.strip()}")
    return 0 if any("PASSED" in ln for ln in verdict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
