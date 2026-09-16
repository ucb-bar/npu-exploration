"""Silicon cost of a recipe: adapter to the MxGemmini area/power model.

The model lives OUTSIDE this repo (Amanda Shi's MxGemmini-workspace/ppa) and is
consumed as a black box, the same way mxq_golden consumes MXQuant: we call it,
never reimplement it, so it cannot drift from its own calibration. It is
analytical -- table lookups + composition, no EDA tools, milliseconds -- and
needs nothing from a run: only the recipe. It therefore evaluates in parallel
with the spike path, as a fourth consumer of the recipe.

Numbers are POST-SYNTHESIS (tstech16c, tt0p8v25c, 2.0 ns) and calibrated at a
16x16 mesh; any other dim is a structural extrapolation the model's own README
warns about, so results carry ``model.calibrated`` and every run records the
exact invocation and the workspace git head.

Root discovery: ``$MX_PPA_ROOT``, else ``<repo>/../MxGemmini-workspace/ppa``.
Absence is not an error here -- callers treat :class:`PpaError` as "skip, log".
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Defaults matching the model's own tapeout-baseline invocation (ppa/README.md
#: "Quick start"): --util 0.965 --blocks-variant new --calib new at 2.0 ns.
DEFAULT_UTIL = 0.965
DEFAULT_CLOCK_NS = 2.0
CALIBRATED_DIM = 16
TECH, CORNER = "tstech16c", "tt0p8v25c"


class PpaError(RuntimeError):
    """The PPA model is unavailable or its output did not parse. Callers skip."""


def ppa_root() -> Path:
    env = os.environ.get("MX_PPA_ROOT")
    root = Path(env) if env else REPO.parent / "MxGemmini-workspace" / "ppa"
    if not (root / "compose_gemmini.py").exists():
        raise PpaError(f"compose_gemmini.py not found at {root} "
                       "(clone Rakanic/MxGemmini-workspace beside this repo, "
                       "or set MX_PPA_ROOT)")
    return root


def _run_length(pairs) -> str:
    """[(e,s)]*16 -> the model's --rows spelling, e.g. '8x4,5 2x4,6 5x4,7 1x8,8'."""
    groups: list[tuple[int, tuple[int, int]]] = []
    for p in pairs:
        if groups and groups[-1][1] == p:
            groups[-1] = (groups[-1][0] + 1, p)
        else:
            groups.append((1, p))
    return " ".join(f"{n}x{e},{s}" for n, (e, s) in groups)


def ppa_args(recipe, *, util: float = DEFAULT_UTIL,
             clock_ns: float = DEFAULT_CLOCK_NS) -> list[str]:
    """Map a recipe onto compose_gemmini's CLI.

    The model speaks (expWidth, sigWidth); the recipe's properties speak
    (e, m = sig-1), so sig = m + 1 here. The acc ladder is run-length encoded
    per the model's --rows grammar; baseline must come out exactly as the
    model's documented tapeout invocation (tests/selftest_ppa.py pins this).
    """
    rows = _run_length([(e, m + 1) for e, m in zip(recipe.acc_e, recipe.acc_m)])
    return ["--rows", rows,
            "--prod", f"{recipe.prod_e},{recipe.prod_m + 1}",
            "--cols", str(recipe.dim),
            "--stim", recipe.operand_fmt,
            "--fmtset", "mxgemmini",
            "--calib", "new", "--blocks-variant", "new",
            "--util", str(util), "--clock-ns", str(clock_ns)]
    # --lut is left at the model's default (fp6): it selects which QuantLut
    # projection the requantizer block was synthesized with, and fp6 is the
    # hardware as built. Revisit when recipes grow a LUT-format field.


_ROW = re.compile(r"^(?P<name>\S.*?)\s{2,}(?P<area>\d+(?:\.\d+)?)k\s+"
                  r"(?P<power>\d+(?:\.\d+)?)\s*(?:\(.*)?$")
_SHARE = re.compile(r"mesh share:\s*(\d+)% of area,\s*(\d+)% of power")
_ENERGY = re.compile(r"([\d.]+)\s*Gop/s;\s*energy\s+([\d.]+)\s+pJ per op \(Gemmini\),"
                     r"\s+([\d.]+)\s+pJ per op \(mesh\)")


def _parse(stdout: str) -> dict:
    blocks: dict[str, dict] = {}
    total = None
    for line in stdout.splitlines():
        m = _ROW.match(line.rstrip())
        if not m:
            continue
        name = m["name"].split(" (")[0].strip()
        entry = {"area_um2": float(m["area"]) * 1e3, "power_mw": float(m["power"])}
        if name == "Gemmini total":
            total = entry
        else:
            blocks[name] = entry
    share = _SHARE.search(stdout)
    energy = _ENERGY.search(stdout)
    if total is None or not blocks or energy is None:
        raise PpaError("could not parse compose_gemmini output -- its format "
                       f"changed?\n--- stdout ---\n{stdout[-1500:]}")
    out = {"area_um2": total["area_um2"], "power_mw": total["power_mw"],
           "pj_per_op": float(energy.group(2)),
           "throughput_gops": float(energy.group(1)),
           "pj_per_op_mesh": float(energy.group(3)),
           "blocks": blocks}
    if share:
        out["mesh_share"] = {"area_pct": int(share.group(1)),
                             "power_pct": int(share.group(2))}
    return out


def _workspace_head(root: Path) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def run_ppa(recipe, *, util: float = DEFAULT_UTIL,
            clock_ns: float = DEFAULT_CLOCK_NS) -> dict:
    """Area/power/energy for the machine this recipe describes. Raises PpaError."""
    root = ppa_root()
    args = ppa_args(recipe, util=util, clock_ns=clock_ns)
    cmd = [sys.executable, str(root / "compose_gemmini.py"), *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=root)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PpaError(f"compose_gemmini failed to run: {exc}") from exc
    if r.returncode != 0:
        raise PpaError(f"compose_gemmini exited {r.returncode}:\n{r.stderr[-1500:]}")
    out = _parse(r.stdout)
    out["model"] = {
        "source": "MxGemmini-workspace/ppa/compose_gemmini.py",
        "args": " ".join(args),
        "tech": TECH, "corner": CORNER,
        "clock_ns": clock_ns, "util": util,
        "post_synthesis": True,
        "calibrated_dim": CALIBRATED_DIM,
        "calibrated": recipe.dim == CALIBRATED_DIM,
        "workspace_head": _workspace_head(root),
    }
    return out


def main() -> int:
    import argparse
    import json as _json
    from config.recipe import load
    ap = argparse.ArgumentParser(description="Silicon cost of a recipe (PPA model)")
    ap.add_argument("--config", default="baseline")
    ap.add_argument("--util", type=float, default=DEFAULT_UTIL)
    ap.add_argument("--clock-ns", type=float, default=DEFAULT_CLOCK_NS)
    ap.add_argument("--json", action="store_true", help="machine-readable output only")
    a = ap.parse_args()
    try:
        res = run_ppa(load(a.config), util=a.util, clock_ns=a.clock_ns)
    except PpaError as exc:
        print(f"ppa: {exc}", file=sys.stderr)
        return 2
    if a.json:
        print(_json.dumps(res, indent=1))
    else:
        print(f"{a.config}: {res['area_um2']/1e3:.1f}k um2  {res['power_mw']:.1f} mW  "
              f"{res['pj_per_op']:.2f} pJ/op  ({res['throughput_gops']:.1f} Gop/s; "
              f"post-syn {TECH} model, calibrated={res['model']['calibrated']})")
        for name, b in res["blocks"].items():
            print(f"  {name:44s} {b['area_um2']/1e3:9.1f}k  {b['power_mw']:7.1f} mW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
