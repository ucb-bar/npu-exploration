"""PyTorch -> MX-Gemmini -> spike, graded. The single entry point.

A kernel is a chain of one or more MX matmuls, so one command covers both a single
layer and a stitched model:

    source scripts/env.sh
    .venv/bin/python run_kernel.py --list
    .venv/bin/python run_kernel.py --kernel linear
    .venv/bin/python run_kernel.py --kernel linear --m 32 --k 128 --n 96
    .venv/bin/python run_kernel.py --kernel mlp2 --h 128 --seam rescale --artifacts
    .venv/bin/python run_kernel.py --kernel linear --tol 0.01     # forced FAIL
    .venv/bin/python run_kernel.py --kernel linear --build-only   # stop at the ELF

Phase 1 grades against FP32, which measures the cost of the MX format. It does NOT
prove the hardware correct -- that needs the MX golden (phase 2).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from grade.pipeline import run
from grade.telemetry import Telemetry
from kernels.registry import build, list_kernels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list known kernels and exit")
    ap.add_argument("--kernel", default="linear", help="kernel name (see --list)")
    ap.add_argument("--m", type=int, default=64, help="batch rows")
    ap.add_argument("--k", type=int, default=64, help="in_features")
    ap.add_argument("--h", type=int, default=64, help="hidden width (chained kernels)")
    ap.add_argument("--n", type=int, default=64, help="out_features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seam", choices=("weight", "rescale"), default="weight",
                    help="how the requantized intermediate is made safe for the next stage")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="pass threshold on relative Frobenius error vs fp32")
    ap.add_argument("--simulator", default="spike")
    ap.add_argument("--artifacts", action="store_true",
                    help="write the RTL-replay bundle (interface MLIR + C + operands.npz)")
    ap.add_argument("--build-only", action="store_true", help="emit the ELF, do not run it")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    a = ap.parse_args()

    if a.list:
        print("kernels:")
        for name, desc in list_kernels().items():
            print(f"  {name:10s} {desc}")
        return 0

    tel = Telemetry()
    try:
        spec = build(a.kernel, m=a.m, k=a.k, h=a.h, n=a.n, seed=a.seed)
        res = run(spec, tol=a.tol, simulator=a.simulator, seam=a.seam,
                  build_only=a.build_only, artifacts=a.artifacts,
                  workdir=a.workdir, results_dir=a.results_dir, telemetry=tel)
    except Exception as exc:
        tel.log("error", f"{type(exc).__name__}: {exc}")
        return 2

    if res["metrics"] is None:
        return 0
    m = res["metrics"]
    print(f"\nVERDICT  {'PASS' if m['pass'] else 'FAIL'}  "
          f"(tier={m['tier']}, rel_fro={m['accuracy_vs_fp32_reference']['rel_fro']:.4%}, "
          f"tol={a.tol:.2%}, cycles={m['total_cycles']})")
    print(f"RESULTS  {res['run_dir']}")
    return 0 if m["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
