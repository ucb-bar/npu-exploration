"""The mxquant model on a workload: perplexity of one language model on one recipe machine.

    .venv/bin/python -m models.mxquant --list                                          # registered workloads
    .venv/bin/python -m models.mxquant --workload tinyllama --config baseline --dry-run # which layers get the Scheme
    .venv/bin/python -m models.mxquant --workload tinyllama --config baseline --gpus 0,1,2,3   # ~12 min on 4 L40S, then cached
    .venv/bin/python -m models.mxquant --workload tinyllama --config wide_acc --dtype fp4_e2m1 --gpus 0,1 --nsamples 4
    .venv/bin/python -m models.mxquant --workload tinyllama --config baseline --gpus 0,1,2,3 --rounding-mode ties_away --scale-floor 1e-38

For one kernel's bits (the same model's other path) use run_kernel.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import scheme  # noqa: E402
from config.recipe import RecipeError, load  # noqa: E402
from grade.telemetry import Telemetry  # noqa: E402
from models.mxquant import workload, workloads  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="the registered workloads")
    ap.add_argument("--workload", default="tinyllama", help="a registered workload (--list)")
    ap.add_argument("--config", default="baseline", help="recipe name in config/recipes/ or a .json path")
    ap.add_argument("--dtype", default=None, help=f"operand format (default: the recipe's): {', '.join(scheme.MXQ_FORMAT)}")
    ap.add_argument("--gpus", default=None, help="GPUs to split the samples over, e.g. 0,1,2,3 (default: one worker)")
    ap.add_argument("--nsamples", type=int, default=None, help="override the workload's sample count")
    ap.add_argument("--seqlen", type=int, default=None, help="override the workload's tokens per sample")
    ap.add_argument("--seed", type=int, default=None, help="override the workload's seed")
    ap.add_argument("--rules", default=None, help="override which linear layers: mxquant_layers | all_linear")
    ap.add_argument("--model-id", default=None, help="override the workload's HF model")
    ap.add_argument("--rounding-mode", default=scheme.ROUNDING, help="operand rounding: rne (hardware) | ties_away")
    ap.add_argument("--scale-floor", type=float, default=None, help="block-max floor (default 2^-23, the hardware's)")
    ap.add_argument("--no-compiled", action="store_true", help="skip torch.compile (5-7x slower, same bits)")
    ap.add_argument("--force", action="store_true", help="measure again even if cached")
    ap.add_argument("--dry-run", action="store_true", help="print the patch table, run nothing")
    ap.add_argument("--results-dir", type=Path, default=workload.RESULTS, help="cache directory")
    ap.add_argument("--json", action="store_true", help="print the full record")
    a = ap.parse_args()

    if a.list:
        for w in workloads.WORKLOADS.values():
            print(f"{w.name:12s} {w.model_id}  {w.nsamples}x{w.seqlen} seed {w.seed}  rules {w.rules}")
        return 0
    overrides = {k: v for k, v in (("nsamples", a.nsamples), ("seqlen", a.seqlen), ("seed", a.seed),
                                   ("rules", a.rules), ("model_id", a.model_id)) if v is not None}
    tel = Telemetry()
    try:
        recipe = load(a.config)
        if a.dry_run:
            return workload.dry_run(a.workload, recipe, dtype=a.dtype, rounding_mode=a.rounding_mode,
                                    scale_floor=a.scale_floor, **overrides)
        m = workload.evaluate(a.workload, recipe, dtype=a.dtype, gpus=a.gpus, rounding_mode=a.rounding_mode,
                              scale_floor=a.scale_floor, compiled=not a.no_compiled, results_dir=a.results_dir,
                              force=a.force, tel=tel, **overrides)
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
    print(f"         {m['perplexity']!r}   (bf16 {m['bf16_perplexity']!r})   recipe {m['recipe']} build_id {m['build_id']}  "
          f"{m['dtype']} {m['rounding_mode']} floor {m['scale_floor']:g}  mxq {m['mxq_commit']}")
    print(f"RESULTS  {m['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
