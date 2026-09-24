"""The accuracy model on its own: TinyLlama perplexity for one recipe.

    .venv/bin/python -m models.accuracy --config baseline --dry-run              # which layers get the Scheme
    .venv/bin/python -m models.accuracy --config baseline --gpus 0,1,2,3           # ~12 min on 4 L40S, then cached
    .venv/bin/python -m models.accuracy --config wide_acc --gpus 0,1 --nsamples 4
    .venv/bin/python -m models.accuracy --config baseline --gpus 0,1,2,3 --rounding-mode ties_away --scale-floor 1e-38
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
from models.accuracy import accuracy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="baseline", help="recipe name in config/recipes/ or a .json path")
    ap.add_argument("--gpus", default=None, help="GPUs to split the samples over, e.g. 0,1,2,3 (default: one worker)")
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules", default="mxquant_layers", help="which linear layers: mxquant_layers | all_linear")
    ap.add_argument("--model-id", default=accuracy.MODEL_ID)
    ap.add_argument("--rounding-mode", default=scheme.ROUNDING, help="operand rounding: rne (hardware) | ties_away")
    ap.add_argument("--scale-floor", type=float, default=None, help="block-max floor (default 2^-23, the hardware's)")
    ap.add_argument("--no-compiled", action="store_true", help="skip torch.compile (5-7x slower, same bits)")
    ap.add_argument("--force", action="store_true", help="measure again even if cached")
    ap.add_argument("--dry-run", action="store_true", help="print the patch table, run nothing")
    ap.add_argument("--results-dir", type=Path, default=accuracy.RESULTS, help="cache directory")
    ap.add_argument("--json", action="store_true", help="print the full record")
    a = ap.parse_args()

    tel = Telemetry()
    try:
        recipe = load(a.config)
        if a.dry_run:
            return accuracy.dry_run(recipe, rules=a.rules, rounding_mode=a.rounding_mode,
                                    scale_floor=a.scale_floor, model_id=a.model_id, seqlen=a.seqlen)
        m = accuracy.run(recipe, model_id=a.model_id, nsamples=a.nsamples, seqlen=a.seqlen, seed=a.seed,
                         gpus=a.gpus, rules=a.rules, rounding_mode=a.rounding_mode, scale_floor=a.scale_floor,
                         compiled=not a.no_compiled, results_dir=a.results_dir, force=a.force, tel=tel)
    except RecipeError as exc:
        tel.log("error", f"bad recipe: {exc}")
        return 2
    except Exception as exc:
        tel.log("error", f"{type(exc).__name__}: {exc}")
        return 2
    if a.json:
        import json
        print(json.dumps(m, indent=1))
    print(accuracy.line(m))
    print(f"         {m['perplexity']!r}   (bf16 {m['bf16_perplexity']!r})   recipe {m['recipe']} build_id {m['build_id']}  "
          f"{m['rounding_mode']} floor {m['scale_floor']:g}  mxq {m['mxq_commit']}")
    print(f"RESULTS  {m['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
