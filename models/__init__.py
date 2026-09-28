"""models -- one folder per model, each answering one question about the machine a recipe describes.

    reference   fp32 torch: what the kernel computes with no quantization        (context only)
    mxquant     the bits the accelerator must produce, from the recipe, on mxq   (VERDICT: spike == mxquant)
    spike       the functional device model: a compiled ELF run on spike         (the thing being graded)
    ppa         silicon cost of the machine: area, power, pJ/op                  (post-synthesis model)
    perf        predicted timeline of this kernel on that machine                (RTL-calibrated model)

mxquant has a second path, ``evaluate(workload, recipe)``: a whole language model with the same
arithmetic in its linear layers, measured as perplexity (GPU, minutes; ``python -m models.mxquant``).

Each folder exposes ``run(...) -> dict`` and ``line(metrics) -> str``, the one terminal line it prints.
``run_kernel.py --models`` picks which ones run; all five take seconds.

The mxq library (git submodule ``microscaling-quant/``) is put on ``sys.path`` HERE and nowhere else.
The directory is hyphenated on purpose: a directory called ``mxq`` at the repo root would shadow the
package, because the repo root is on ``sys.path`` too.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MXQ_ROOT = REPO / "microscaling-quant"

NAMES = ("reference", "mxquant", "spike", "ppa", "perf")
DEFAULT = NAMES
GROUPS = {"default": DEFAULT, "all": NAMES}


def paths() -> bool:
    """Put the mxq submodule on sys.path. Idempotent. False, not an error, when it is not checked
    out: the models that need it report UNAVAILABLE with the fix, and the rest of a run proceeds."""
    if not (MXQ_ROOT / "mxq" / "__init__.py").exists():
        return False
    s = str(MXQ_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)
    return True


def mxq_missing() -> str:
    """The message every model prints when mxq is absent."""
    return (f"mxq submodule not checked out at {MXQ_ROOT} -- "
            "run: git submodule update --init microscaling-quant")


def mxq_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(MXQ_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def select(arg: str) -> tuple[str, ...]:
    """``"default"`` | ``"all"`` | a comma list of model names or group names -> canonical order."""
    chosen: set[str] = set()
    for tok in (t.strip() for t in arg.split(",")):
        if not tok:
            continue
        if tok in GROUPS:
            chosen.update(GROUPS[tok])
        elif tok in NAMES:
            chosen.add(tok)
        else:
            raise ValueError(f"unknown model {tok!r}; choose from {', '.join(NAMES)} "
                             f"or a group: {', '.join(GROUPS)}")
    if not chosen:
        raise ValueError("no model selected")
    return tuple(n for n in NAMES if n in chosen)


paths()
