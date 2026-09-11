"""PyTorch -> MX-Gemmini -> spike, graded. The single entry point.

A kernel is a chain of one or more MX matmuls, so one command covers both a single
layer and a stitched model:

    source scripts/env.sh
    .venv/bin/python run_kernel.py --list
    .venv/bin/python run_kernel.py --kernel linear
    .venv/bin/python run_kernel.py --kernel linear --m 32 --k 128 --n 96
    .venv/bin/python run_kernel.py --kernel linear --dtype fp4_e2m1
    .venv/bin/python run_kernel.py --kernel mlp2 --h 128 --artifacts
    .venv/bin/python run_kernel.py --kernel linear --tol 0.01     # forced FAIL
    .venv/bin/python run_kernel.py --kernel linear --build-only   # stop at the ELF

Phase 1 grades against FP32, which measures the cost of the MX format. It does NOT
prove the hardware correct -- that needs the MX golden (phase 2).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config.recipe import RecipeError, list_recipes
from config.recipe import load as load_recipe
from grade.pipeline import run
from grade.telemetry import Telemetry
from kernels.registry import build, list_kernels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list known kernels and hardware recipes, then exit")
    ap.add_argument("--kernel", default="linear", help="kernel name (see --list)")
    ap.add_argument("--config", default="baseline",
                    help="hardware recipe: a name in config/recipes/ or a path to a .json. "
                         "Defines the MX-Gemmini the kernel runs on, and is the single "
                         "definition shared by the software model, spike and Verilator")
    ap.add_argument("--m", type=int, default=64, help="batch rows")
    ap.add_argument("--k", type=int, default=64, help="in_features")
    ap.add_argument("--h", type=int, default=64, help="hidden width (chained kernels)")
    ap.add_argument("--n", type=int, default=64, help="out_features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seam", choices=("weight", "rescale"), default=None,
                    help="how the requantized intermediate is made safe for the next stage "
                         "(default: the recipe's software.seam)")
    ap.add_argument("--dtype", default="fp8_e4m3",
                    help="MX operand format (see app/mxformats.py; unproven formats are refused)")
    ap.add_argument("--allow-lossy-chain", action="store_true",
                    help="chain a format whose requant range exceeds what the nearest-entry finder "
                         "can represent (fp6_e3m2, fp8_e5m2). The reference has the same behaviour; "
                         "see app/mxformats.chain_refusal")
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
        print("\nhardware recipes (--config):")
        for name, desc in list_recipes().items():
            print(f"  {name:14s} {desc}")
        return 0

    tel = Telemetry()
    try:
        recipe = load_recipe(a.config)
        spec = build(a.kernel, m=a.m, k=a.k, h=a.h, n=a.n, seed=a.seed)
        res = run(spec, recipe=recipe, tol=a.tol, simulator=a.simulator, seam=a.seam,
                  dtype=a.dtype, allow_lossy_chain=a.allow_lossy_chain,
                  build_only=a.build_only, artifacts=a.artifacts,
                  workdir=a.workdir, results_dir=a.results_dir, telemetry=tel)
    except RecipeError as exc:
        tel.log("error", f"bad recipe: {exc}")
        return 2
    except Exception as exc:
        tel.log("error", f"{type(exc).__name__}: {exc}")
        return 2

    if res["metrics"] is None:
        return 0
    m = res["metrics"]
    ok = "PASS" if m["pass"] else "FAIL"
    corr = m.get("correctness_vs_golden_model")
    if corr is not None:
        # The verdict is bit-identity against MXQuant/rtl_exact. No tolerance appears in it.
        gap = (m.get("delta_vs_mxquant_as_shipped") or {}).get("rel_fro")
        print(f"\nVERDICT  {ok}  hardware {'==' if corr['bit_exact'] else '!='} MXQuant/rtl_exact"
              f"  ({corr['total_elements'] - corr['n_mismatch']}/{corr['total_elements']}"
              f" identical, max|d| {corr['max_abs_diff']:g})")
        if gap is not None:
            print(f"         MXQuant as shipped is {gap:.2%} from the hardware "
                  f"-- the model-vs-silicon gap")
        print(f"         fp32 {m['accuracy_vs_fp32_reference']['rel_fro']:.4%} (context), "
              f"cycles {m['total_cycles']}")
    else:
        print(f"\nVERDICT  {ok}  "
              f"(tier={m['tier']}, rel_fro={m['accuracy_vs_fp32_reference']['rel_fro']:.4%}, "
              f"tol={a.tol:.2%}, cycles={m['total_cycles']})")
    print(f"RESULTS  {res['run_dir']}")
    return 0 if m["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
