"""PyTorch -> ELF: compile an ``nn.Linear`` into a spike-executable binary for the MX Gemmini.

This is the front end. A layer defined in PyTorch is quantized to MX (fp8 e4m3 codes + E8M0 block
scales), lowered through the ``mx_gemmini_rocket`` backend, and emitted as a bare-metal ELF. Running
it on spike is one more call — and spike is the reference for what the hardware computes, so this
script does not try to predict the numbers.

    .venv/bin/python app/torch_linear/run_linear.py                 # build the ELF and run it
    .venv/bin/python app/torch_linear/run_linear.py --build-only    # stop at the ELF
    .venv/bin/python app/torch_linear/run_linear.py --m 32 --k 64 --n 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "app"))
sys.path.insert(0, str(REPO / "compiler" / "targets" / "mx_gemmini_rocket"))

import backend as mx          # noqa: E402  the compiler target package
import mxcb                   # noqa: E402
import mxquant as q           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, default=64, help="batch rows")
    ap.add_argument("--k", type=int, default=64, help="in_features")
    ap.add_argument("--n", type=int, default=64, help="out_features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-only", action="store_true", help="emit the ELF, do not run it")
    ap.add_argument("--workdir", type=Path, default=REPO / "out" / "build" / "torch_linear")
    a = ap.parse_args()

    torch.manual_seed(a.seed)

    # ---- the model ------------------------------------------------------------------------------
    layer = nn.Linear(a.k, a.n, bias=False)
    x = torch.randn(a.m, a.k)
    print(f"model     nn.Linear({a.k} -> {a.n}, bias=False), input [{a.m}][{a.k}]")

    # ---- quantize to MX -------------------------------------------------------------------------
    # nn.Linear stores weight as [out_features][in_features] = [N][K] and computes x @ W.T, so the
    # matmul's B operand is W.T -> [K][N].
    ops = q.quantize_matmul_operands(
        x.detach().numpy().astype(np.float32),
        layer.weight.detach().numpy().T.astype(np.float32))
    print(f"quantize  mxfp8 e4m3 + E8M0 block scales (group {q.BLOCK}, "
          f"peak code 2^{q.TARGET_CODE_EXP})")

    # ---- lower + build --------------------------------------------------------------------------
    cb = mxcb.matmul_cb(a.m, a.n, a.k, ops)
    if not mx.available("spike"):
        print("toolchain NOT AVAILABLE — source the chipyard env.sh first.", file=sys.stderr)
        return 2
    elf = mx.compile_command_buffer(cb, a.workdir)
    print(f"elf       {elf}  ({elf.stat().st_size} bytes)")
    if a.build_only:
        return 0

    # ---- run ------------------------------------------------------------------------------------
    console = mx.run_elf(elf, simulator="spike")
    outputs, metrics = mx.parse_output(console)
    got = q.bf16_bits_to_float(np.array(outputs["Y0"], dtype=np.uint16))
    finite = int(np.isfinite(got).sum())
    print(f"spike     Y0 {got.shape} bf16   METRIC {metrics}")
    print(f"          finite {finite}/{got.size}   "
          f"range [{np.nanmin(got):.4g}, {np.nanmax(got):.4g}]")
    if finite != got.size:
        print("          WARNING: non-finite outputs — operands likely overflow the datapath's "
              "4-bit-exponent intermediate accumulator (see mxquant.TARGET_CODE_EXP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
