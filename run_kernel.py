"""PyTorch -> MX-Gemmini -> spike, graded. The single entry point of exploration mode.

A kernel is a chain of one or more MX matmuls, so one command covers both a single
layer and a stitched model. ``--models`` picks which models run on the recipe's machine:

    source scripts/env.sh
    .venv/bin/python run_kernel.py --list
    .venv/bin/python run_kernel.py --kernel linear                         # default: reference mxquant spike ppa perf
    .venv/bin/python run_kernel.py --kernel linear --hw wide_acc
    .venv/bin/python run_kernel.py --kernel linear --m 32 --k 128 --n 96
    .venv/bin/python run_kernel.py --kernel linear --run fp4_e2m1
    .venv/bin/python run_kernel.py --kernel mlp2 --h 128 --artifacts
    .venv/bin/python run_kernel.py --kernel linear --build-only            # stop at the ELF
    .venv/bin/python run_kernel.py --kernel linear --models mxquant        # the model alone, no spike, seconds
    .venv/bin/python -m models.mxquant --workload tinyllama --hw wide_acc --gpus 0,1,2,3   # perplexity (minutes)

The hardware recipe (--hw, config/hardware/) is the machine; the run recipe (--run, config/run/) the operand
format and grading tolerance. VERDICT is bit-identity between spike and the mxquant model (models/mxquant).
The fp32 reference is context: it measures the cost of the MX format, not the correctness of the
hardware. Without spike there is no verdict; the run prints the model's own numbers and NO VERDICT.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import models
from config.recipe import RecipeError, list_hardware, list_runs, load_hardware, load_run, removed_flag
from grade.pipeline import run
from grade.telemetry import Telemetry
from kernels.registry import build, list_kernels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list known kernels, hardware recipes and models, then exit")
    ap.add_argument("--kernel", default="linear", help="kernel name (see --list)")
    ap.add_argument("--hw", "--config", dest="hw", default="baseline",
                    help="hardware recipe: a name in config/hardware/ or a path to a .json. "
                         "Defines the MX-Gemmini the kernel runs on, for every model and spike")
    ap.add_argument("--run", default="default",
                    help="run recipe: a name in config/run/ or a path to a .json "
                         "(operand format, lossy-chain switch, fp32 tolerance)")
    ap.add_argument("--models", default="default",
                    help="which models run: 'default' (%s), 'all' (the same today), or a comma list of "
                         "%s" % (",".join(models.DEFAULT), ", ".join(models.NAMES)))
    ap.add_argument("--m", type=int, default=64, help="batch rows")
    ap.add_argument("--k", type=int, default=64, help="in_features")
    ap.add_argument("--h", type=int, default=64, help="hidden width (chained kernels)")
    ap.add_argument("--n", type=int, default=64, help="out_features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--simulator", default="spike")
    ap.add_argument("--artifacts", action="store_true",
                    help="write the RTL-replay bundle (interface MLIR + C + operands.npz)")
    ap.add_argument("--build-only", action="store_true", help="emit the ELF, do not run it")
    ap.add_argument("--per-stage-elf", action="store_true",
                    help="one ELF per matmul, intermediates carried by the host (the differential-debugging "
                         "path); the default fuses a chain or emits a graph as ONE ELF")
    ap.add_argument("--legacy-mxquant", action="store_true",
                    help="grade against the legacy implementation (grade/mxquant_ref.py: MXQuant's "
                         "simulator patched at runtime; needs the MXQuant clone; recipe-blind). "
                         "For the equivalence test only; removed in the next PR")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    gone = removed_flag(sys.argv[1:])
    if gone:
        print(f"[error   ] {gone}", file=sys.stderr)
        return 2
    a = ap.parse_args()

    if a.list:
        print("kernels:")
        for name, desc in list_kernels().items():
            print(f"  {name:10s} {desc}")
        print("\nhardware recipes (--hw):")
        for name, desc in list_hardware().items():
            print(f"  {name:14s} {desc}")
        print("\nrun recipes (--run):")
        for name, desc in list_runs().items():
            print(f"  {name:14s} {desc}")
        print("\nmodels (--models): " + ", ".join(models.NAMES)
              + f"   groups: default = {','.join(models.DEFAULT)}; all = every model")
        return 0

    tel = Telemetry()
    try:
        selected = models.select(a.models)
        recipe, run_recipe = load_hardware(a.hw), load_run(a.run)
        spec = build(a.kernel, m=a.m, k=a.k, h=a.h, n=a.n, seed=a.seed)
        res = run(spec, recipe=recipe, run_recipe=run_recipe, simulator=a.simulator,
                  build_only=a.build_only, artifacts=a.artifacts, per_stage_elf=a.per_stage_elf,
                  workdir=a.workdir, results_dir=a.results_dir, telemetry=tel,
                  models=selected, legacy_mxquant=a.legacy_mxquant)
    except RecipeError as exc:
        tel.log("error", f"bad recipe: {exc}")
        return 2
    except ValueError as exc:
        tel.log("error", str(exc))
        return 2
    except Exception as exc:
        tel.log("error", f"{type(exc).__name__}: {exc}")
        return 2

    if res["metrics"] is None:
        return 0
    m = res["metrics"]

    # Each model prints its own line. VERDICT belongs to the pair spike + mxquant; with spike alone
    # it is the fp32 tier; with mxquant alone there is no verdict, only the model's numbers.
    if m.get("correctness_vs_mxquant") is not None or "spike" not in selected:
        from models import mxquant
        print(mxquant.line(m))
    else:
        from models import reference
        print(reference.line(m, tol=run_recipe.fp32_tol))
    if m.get("ppa"):
        from models.ppa import ppa
        print(ppa.line(m["ppa"]))
    if m.get("perf"):
        from models.perf import perf
        print(perf.line(m["perf"]))
    print(f"RESULTS  {res['run_dir']}")
    return {True: 0, False: 1, None: 0}[m["pass"]]


if __name__ == "__main__":
    raise SystemExit(main())
