"""Bring-up experiment: the fp8 64x64x64 MX matmul, end to end.

Builds a merlin command buffer for one weight-stationary MX GEMM, hands it to the
``mx_gemmini_rocket`` backend, and writes the emitted C. Compile + run are driven by ``sim/``.

This is the APP layer: it owns the operands, the shape, and the choice of experiment. The compiler
backend receives a command buffer and knows none of that.

The operands come from the gemmini-rocc-tests data header whose golden is already trusted on spike
(plan section 6) — the point of this experiment is to prove the emitted kernel reproduces it. From
Step 8 the same command buffer is built from PyTorch tensors instead, and nothing downstream changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from header_operands import load_mx_operands

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "compiler" / "targets" / "mx_gemmini_rocket" / "backend"
DEFAULT_HEADER = (REPO.parent / "software" / "gemmini-rocc-tests" / "include"
                  / "matmul_fp8_64x64.h")
DEFAULT_OUT = REPO / "out" / "build" / "mx_gemm_fp8_64x64.c"

sys.path.insert(0, str(BACKEND))
from mxgemm_emit import generate_driver  # noqa: E402


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
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write the emitted C")
    a = ap.parse_args()

    ops = load_mx_operands(a.header)
    m, k = len(ops["a_codes"]), len(ops["a_codes"][0])
    n = len(ops["b_codes"][0])
    print(f"operands: A[{m}][{k}] B[{k}][{n}] "
          f"scales A{len(ops['a_scales'])}x{len(ops['a_scales'][0])} "
          f"B{len(ops['b_scales'])}x{len(ops['b_scales'][0])}  from {a.header.name}")

    src = generate_driver(build_command_buffer(m, n, k, ops))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(src, encoding="utf-8")
    print(f"emitted {len(src.splitlines())} lines -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
