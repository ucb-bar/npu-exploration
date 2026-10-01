"""The mxquant model on a workload: perplexity of one language model on one hardware recipe, driven by a run recipe.

    .venv/bin/python -m models.mxquant --list                                           # workloads, hardware and run recipes
    .venv/bin/python -m models.mxquant --workload tinyllama --hw baseline --dry-run      # which layers get the Scheme
    .venv/bin/python -m models.mxquant --workload tinyllama --hw baseline --gpus 0,1,2,3 # run recipe "default"; ~6 min on 4 L40S, then cached
    .venv/bin/python -m models.mxquant --workload tinyllama --hw wide_acc --run fp4_e2m1 --gpus 0,1 --nsamples 4
    .venv/bin/python -m models.mxquant --workload tinyllama --hw baseline --run exact --gpus 0,1,2,3        # the format's cost alone
    .venv/bin/python -m models.mxquant --workload tinyllama --hw baseline --run bf16_tiles --gpus 0,1,2,3   # + bf16 across blocks
    .venv/bin/python -m models.mxquant --workload tinyllama --hw none --gpus 0,1,2,3 --nsamples 0          # bf16 model, whole split

Any other operand format, rounding, scale floor or reducer is a run recipe: copy config/run/default.json and
pass --run <file>. For one kernel's bits (the same model's other path) use run_kernel.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config.recipe import RecipeError, list_hardware, list_runs, load_hardware, load_run, removed_flag  # noqa: E402
from grade.telemetry import Telemetry  # noqa: E402
from models.mxquant import workload, workloads  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="the registered workloads and the recipes")
    ap.add_argument("--workload", default="tinyllama", help="a registered workload (--list)")
    ap.add_argument("--hw", "--config", dest="hw", default="baseline",
                    help="hardware recipe: a name in config/hardware/ or a .json path; 'none' = the bf16 model alone")
    ap.add_argument("--run", default="default", help="run recipe: a name in config/run/ or a .json path")
    ap.add_argument("--gpus", default=None, help="GPUs to split the samples over, e.g. 0,1,2,3 (default: one worker)")
    ap.add_argument("--nsamples", type=int, default=None, help="override the workload's sample count; 0 = the whole test split")
    ap.add_argument("--seqlen", type=int, default=None, help="override the workload's tokens per sample")
    ap.add_argument("--seed", type=int, default=None, help="override the workload's seed")
    ap.add_argument("--sequential", action="store_true", help="the first nsamples in order instead of seeded ones")
    ap.add_argument("--rules", default=None, help="override which linear layers: mxquant_layers | all_linear | linears_no_head")
    ap.add_argument("--model-id", default=None, help="override the workload's HF model")
    ap.add_argument("--no-compiled", action="store_true", help="skip torch.compile (5-7x slower, same bits)")
    ap.add_argument("--force", action="store_true", help="measure again even if cached")
    ap.add_argument("--dry-run", action="store_true", help="print the patch table, run nothing")
    ap.add_argument("--results-dir", type=Path, default=workload.RESULTS, help="cache directory")
    ap.add_argument("--json", action="store_true", help="print the full record")
    gone = removed_flag(sys.argv[1:])
    if gone:
        print(f"[error   ] {gone}", file=sys.stderr)
        return 2
    a = ap.parse_args()

    if a.list:
        print("workloads (--workload):")
        for w in workloads.WORKLOADS.values():
            print(f"  {w.name:12s} {w.model_id}  {w.nsamples}x{w.seqlen} seed {w.seed}  rules {w.rules}")
        print("\nhardware recipes (--hw):")
        for name, desc in list_hardware().items():
            print(f"  {name:14s} {desc}")
        print("\nrun recipes (--run):")
        for name, desc in list_runs().items():
            print(f"  {name:14s} {desc}")
        return 0
    overrides = {k: v for k, v in (("nsamples", a.nsamples), ("seqlen", a.seqlen), ("seed", a.seed),
                                   ("rules", a.rules), ("model_id", a.model_id)) if v is not None}
    if a.sequential:
        overrides["seed"] = None
    tel = Telemetry()
    try:
        if a.hw == "none":
            recipe_only = [f for f, on in (("--run", a.run != "default"), ("--rules", a.rules is not None),
                                           ("--no-compiled", a.no_compiled)) if on]
            if recipe_only:
                tel.log("error", f"--hw none is the bf16 model alone, so {' '.join(recipe_only)} cannot apply")
                return 2
            if a.dry_run:
                print("no recipe: the bf16 model, nothing patched")
                return 0
            m = workload.bf16(a.workload, gpus=a.gpus, results_dir=a.results_dir, force=a.force, tel=tel, **overrides)
            if a.json:
                import json
                print(json.dumps(m, indent=1))
            print(workload.line(m))
            print(f"         {m['perplexity']!r}   bf16 model, no recipe   mxq {m['mxq_commit']}")
            print(f"RESULTS  {m['path']}")
            return 0
        recipe, run = load_hardware(a.hw), load_run(a.run)
        if a.dry_run:
            return workload.dry_run(a.workload, recipe, run, **overrides)
        m = workload.evaluate(a.workload, recipe, run, gpus=a.gpus, compiled=not a.no_compiled,
                              results_dir=a.results_dir, force=a.force, tel=tel, **overrides)
    except RecipeError as exc:
        tel.log("error", f"bad recipe: {exc}")
        return 2
    except Exception as exc:
        tel.log("error", f"{type(exc).__name__}: {exc}")
        return 2
    if a.json:
        import json
        print(json.dumps(m, indent=1))
    print(workload.line(m))
    print(f"         {m['perplexity']!r}   (bf16 {m['bf16_perplexity']!r})   hardware {m['recipe']} {m['build_id']}  "
          f"run {m['run']} {m['run_id']}: {m['dtype']} {m['rounding_mode']} floor {m['scale_floor']:g} "
          f"reduce {m['reduce']}  mxq {m['mxq_commit']}")
    print(f"RESULTS  {m['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
