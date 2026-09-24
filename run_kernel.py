"""PyTorch -> MX-Gemmini -> spike, graded. The single entry point of exploration mode.

A kernel is a chain of one or more MX matmuls, so one command covers both a single
layer and a stitched model. ``--models`` picks which models run on the recipe's machine:

    source scripts/env.sh
    .venv/bin/python run_kernel.py --list
    .venv/bin/python run_kernel.py --kernel linear                         # default: reference mxquant spike ppa perf
    .venv/bin/python run_kernel.py --kernel linear --config wide_acc
    .venv/bin/python run_kernel.py --kernel linear --m 32 --k 128 --n 96
    .venv/bin/python run_kernel.py --kernel linear --dtype fp4_e2m1
    .venv/bin/python run_kernel.py --kernel mlp2 --h 128 --artifacts
    .venv/bin/python run_kernel.py --kernel linear --build-only            # stop at the ELF
    .venv/bin/python run_kernel.py --kernel linear --models mxquant        # the model alone, no spike, seconds
    .venv/bin/python run_kernel.py --kernel linear --models all --gpus 0,1,2,3   # + TinyLlama perplexity (minutes)

VERDICT is bit-identity between spike and the mxquant model (models/mxquant), built from the recipe.
The fp32 reference is context: it measures the cost of the MX format, not the correctness of the
hardware. Without spike there is no verdict; the run prints the model's own numbers and NO VERDICT.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import models
from config.recipe import RecipeError, list_recipes
from config.recipe import load as load_recipe
from grade.pipeline import run
from grade.telemetry import Telemetry
from kernels.registry import build, list_kernels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list known kernels, hardware recipes and models, then exit")
    ap.add_argument("--kernel", default="linear", help="kernel name (see --list)")
    ap.add_argument("--config", default="baseline",
                    help="hardware recipe: a name in config/recipes/ or a path to a .json. "
                         "Defines the MX-Gemmini the kernel runs on, and is the single "
                         "definition shared by every model, spike and Verilator")
    ap.add_argument("--models", default="default",
                    help="which models run: 'default' (%s), 'all' (adds accuracy), or a comma list of "
                         "%s" % (",".join(models.DEFAULT), ", ".join(models.NAMES)))
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
                    help="pass threshold on relative Frobenius error vs fp32 (fp32 tier only)")
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
    ap.add_argument("--gpus", default=None, help="accuracy model: GPUs to split the samples over, e.g. 0,1,2,3")
    ap.add_argument("--nsamples", type=int, default=16, help="accuracy model: WikiText-2 samples of 2048 tokens")
    ap.add_argument("--model-id", default=None, help="accuracy model: HF model id (default TinyLlama-1.1B-Chat)")
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
        print("\nmodels (--models): " + ", ".join(models.NAMES)
              + f"   groups: default = {','.join(models.DEFAULT)}; all = every model")
        return 0

    tel = Telemetry()
    try:
        selected = models.select(a.models)
        recipe = load_recipe(a.config)
        spec = build(a.kernel, m=a.m, k=a.k, h=a.h, n=a.n, seed=a.seed)
        acc_args = {"nsamples": a.nsamples, "gpus": a.gpus}
        if a.model_id:
            acc_args["model_id"] = a.model_id
        res = run(spec, recipe=recipe, tol=a.tol, simulator=a.simulator, seam=a.seam,
                  dtype=a.dtype, allow_lossy_chain=a.allow_lossy_chain,
                  build_only=a.build_only, artifacts=a.artifacts, per_stage_elf=a.per_stage_elf,
                  workdir=a.workdir, results_dir=a.results_dir, telemetry=tel,
                  models=selected, legacy_mxquant=a.legacy_mxquant, accuracy_args=acc_args)
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
        print(reference.line(m, tol=a.tol))
    if m.get("ppa"):
        from models.ppa import ppa
        print(ppa.line(m["ppa"]))
    if m.get("perf"):
        from models.perf import perf
        print(perf.line(m["perf"]))
    if m.get("accuracy"):
        from models.accuracy import accuracy
        print(accuracy.line(m["accuracy"]))
    print(f"RESULTS  {res['run_dir']}")
    return {True: 0, False: 1, None: 0}[m["pass"]]


if __name__ == "__main__":
    raise SystemExit(main())
