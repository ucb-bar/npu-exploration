"""Predicted kernel timeline: adapter to the MxGemmini performance model.

Where ``models/ppa/ppa.py`` prices the MACHINE (area/power, kernel-independent),
this prices the RUN: cycles, wall time, utilization and energy of each GEMM
stage on the machine the recipe describes. The model is Amanda Shi's
``MxGemmini-workspace/ppa/perf/perf_model.py`` -- RTL-FSDB-calibrated
(validated within +1.0/-2.2/-2.2 % on three real kernels) -- and, like the
PPA model, it is consumed as a black box: called, never reimplemented.

Two numbers in the results file look alike and are NOT comparable, on purpose:

* ``perf.total_cycles_predicted`` -- this model's RTL-calibrated timeline,
  including setup, memory, LUT loads and drains;
* the spike stage cycles -- the functional model's op counter, with no memory
  system and no host in it.

Both are recorded, labelled, and never graded against each other.

``--as-measured`` (the default here) reproduces today's kernels -- DRAM-bound
mvins, GPU-written scales, no overlap -- which is the configuration the FSDB
validation covered. ``--energy`` re-runs compose_gemmini at the ACHIEVED
utilization, giving energy per stage and pJ/op including idle.

Root discovery reuses ``models.ppa.ppa.ppa_root()`` (``$MX_PPA_ROOT`` override);
absence raises :class:`PerfError`, which callers treat as "skip, log".
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from models.ppa.ppa import DEFAULT_CLOCK_NS, PpaError, ppa_root, _workspace_head

REPO = Path(__file__).resolve().parents[2]


class PerfError(RuntimeError):
    """The performance model is unavailable or its output did not parse."""


def perf_model_path() -> Path:
    try:
        root = ppa_root()
    except PpaError as exc:
        raise PerfError(str(exc)) from exc
    p = root / "perf" / "perf_model.py"
    if not p.exists():
        raise PerfError(f"perf_model.py not found at {p} "
                        "(pull MxGemmini-workspace; the perf model landed 2026-09-15)")
    return p


#: recipe.operand_fmt (fp8 | fp6 | fp4 families) -> the model's format token.
#: The recipe does not yet distinguish e5m2/e2m3 variants; when it grows that
#: field, map them to the model's fp8e5m2 / fp6e2m3 here.
_OPERAND_TOK = {"fp8": "fp8", "fp6": "fp6", "fp4": "fp4"}

#: stage out_dtype -> --out-fmt ("bf16" means no requant projection).
_OUT_TOK = {"bf16": "bf16", "f8E4M3FN": "fp8", "f8E5M2": "fp8e5m2"}


def _lut_on(recipe) -> bool:
    raw = getattr(recipe, "raw", None) or {}
    return bool(raw.get("runtime", {}).get("use_lut")
                or raw.get("mx", {}).get("enable_lut"))


def perf_args(recipe, m: int, n: int, k: int, out_fmt: str, *,
              as_measured: bool = True, energy: bool = False,
              clock_ns: float = DEFAULT_CLOCK_NS) -> list[str]:
    """Map recipe + one GEMM stage onto perf_model's CLI."""
    tok = _OPERAND_TOK.get(recipe.operand_fmt)
    if tok is None:
        raise PerfError(f"operand format {recipe.operand_fmt!r} has no perf-model token")
    out = _OUT_TOK.get(out_fmt, tok)
    args = ["--M", str(m), "--N", str(n), "--K", str(k),
            "--rows", str(recipe.dim), "--cols", str(recipe.dim),
            "--act", tok, "--wei", tok, "--out-fmt", out,
            "--clock-ns", str(clock_ns)]
    if _lut_on(recipe):
        args.append("--lut")   # 16x16 sharing defaults = one codebook per tile, today's kernels
    if as_measured:
        args.append("--as-measured")
    if energy:
        args.append("--energy")
    return args


_PHASES = re.compile(
    r"cycles: kernel fixed (\d+) \+ first-tile load (\d+) \([^)]*\) \+ "
    r"tile setup exposed (\d+) \([^)]*\) \+ compute (\d+) \+ "
    r"requantizer drain (\d+) \([^)]*\) \+ LUT (\d+)")
_TOTAL = re.compile(
    r"total (\d+) cycles = ([\d.]+) us;\s+([\d.]+) M ops;.*"
    r"utilization ([\d.]+) %;\s+([\d.]+) Gop/s")
_ENERGY = re.compile(
    r"energy: .*?([\d.]+) mW at .*?->\s*([\d.]+) uJ for the GEMM = "
    r"([\d.]+) pJ per op")


def _parse(stdout: str) -> dict:
    ph, tot = _PHASES.search(stdout), _TOTAL.search(stdout)
    if ph is None or tot is None:
        raise PerfError("could not parse perf_model output -- its format changed?"
                        f"\n--- stdout ---\n{stdout[-1500:]}")
    out = {"cycles_predicted": int(tot.group(1)),
           "us": float(tot.group(2)),
           "m_ops": float(tot.group(3)),
           "utilization_pct": float(tot.group(4)),
           "gops": float(tot.group(5)),
           "phases": {"kernel_fixed": int(ph.group(1)),
                      "first_tile_load": int(ph.group(2)),
                      "tile_setup_exposed": int(ph.group(3)),
                      "compute": int(ph.group(4)),
                      "requant_drain": int(ph.group(5)),
                      "lut": int(ph.group(6))}}
    en = _ENERGY.search(stdout)
    if en:
        out["energy"] = {"power_mw_at_util": float(en.group(1)),
                         "uj": float(en.group(2)),
                         "pj_per_op_achieved": float(en.group(3))}
    return out


def _run_one(recipe, m: int, n: int, k: int, out_fmt: str, *,
             as_measured: bool, energy: bool, clock_ns: float) -> dict:
    script = perf_model_path()
    args = perf_args(recipe, m, n, k, out_fmt,
                     as_measured=as_measured, energy=energy, clock_ns=clock_ns)
    cmd = [sys.executable, str(script), *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           cwd=script.parent)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PerfError(f"perf_model failed to run: {exc}") from exc
    if r.returncode != 0:
        raise PerfError(f"perf_model exited {r.returncode}:\n{r.stderr[-1500:]}")
    res = _parse(r.stdout)
    res["args"] = " ".join(args)
    return res


def run_perf(recipe, stages, *, as_measured: bool = True, energy: bool = True,
             clock_ns: float = DEFAULT_CLOCK_NS) -> dict:
    """Predicted timeline for every mesh stage, plus kernel totals.

    ``stages`` is the pipeline's per-stage record: dicts carrying ``m``, ``k``,
    ``n`` and ``out_dtype`` (exactly what lands in metrics.json). Raises
    :class:`PerfError`; callers skip and log.
    """
    script = perf_model_path()   # fail before any work if the model is absent
    per_stage = []
    for s in stages:
        if s.get("where", "mesh") != "mesh":       # host stages run on Rocket; the model prices the mesh
            continue
        res = _run_one(recipe, int(s["m"]), int(s["n"]), int(s["k"]),
                       str(s.get("out_dtype", "bf16")),
                       as_measured=as_measured, energy=energy, clock_ns=clock_ns)
        res["stage"] = s.get("stage", len(per_stage))
        res["gemm"] = f"{s['m']}x{s['k']}x{s['n']}"
        res["out_fmt"] = str(s.get("out_dtype", "bf16"))
        per_stage.append(res)
    if not per_stage:
        raise PerfError("no mesh stages to model")
    out = {
        "stages": per_stage,
        "total_cycles_predicted": sum(p["cycles_predicted"] for p in per_stage),
        "total_us": round(sum(p["us"] for p in per_stage), 2),
        "utilization_pct_min": min(p["utilization_pct"] for p in per_stage),
    }
    if all("energy" in p for p in per_stage):
        uj = sum(p["energy"]["uj"] for p in per_stage)
        ops = sum(p["m_ops"] for p in per_stage) * 1e6
        out["energy"] = {"uj_kernel": round(uj, 3),
                         "pj_per_op_achieved": round(uj * 1e6 / ops, 3) if ops else None}
    out["model"] = {
        "source": "MxGemmini-workspace/ppa/perf/perf_model.py",
        "mode": "as_measured" if as_measured else "ideal",
        "clock_ns": clock_ns,
        "validated": "+1.0/-2.2/-2.2 % total vs three kernel FSDBs (workspace README)",
        "workspace_head": _workspace_head(script.parents[1]),
    }
    return out


def line(perf: dict) -> str:
    """The PERF line run_kernel.py prints: the predicted timeline of this kernel on that machine."""
    e = perf.get("energy")
    return (f"PERF     {perf['total_cycles_predicted']} cycles predicted   "
            f"{perf['total_us']:.1f} us   util {perf['utilization_pct_min']:.1f}%"
            + (f"   {e['uj_kernel']:.2f} uJ ({e['pj_per_op_achieved']:.1f} pJ/op achieved)" if e else "")
            + f"   [spike functional count: {perf.get('spike_functional_cycles')}]")


def main() -> int:
    import argparse
    import json as _json
    from config.recipe import load
    ap = argparse.ArgumentParser(
        description="Predicted GEMM timeline on a recipe's machine (perf model)")
    ap.add_argument("--config", default="baseline")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out-fmt", default="bf16")
    ap.add_argument("--ideal", action="store_true",
                    help="drop --as-measured: overlapped setup, hw-issued mvins")
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--clock-ns", type=float, default=DEFAULT_CLOCK_NS)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    stage = {"stage": 0, "m": a.m, "k": a.k, "n": a.n, "out_dtype": a.out_fmt}
    try:
        res = run_perf(load(a.config), [stage], as_measured=not a.ideal,
                       energy=not a.no_energy, clock_ns=a.clock_ns)
    except PerfError as exc:
        print(f"perf: {exc}", file=sys.stderr)
        return 2
    if a.json:
        print(_json.dumps(res, indent=1))
    else:
        s = res["stages"][0]
        e = s.get("energy")
        print(f"{a.config}: {a.m}x{a.k}x{a.n} -> {s['out_fmt']}: "
              f"{s['cycles_predicted']} cycles = {s['us']} us, "
              f"util {s['utilization_pct']}%, {s['gops']} Gop/s"
              + (f", {e['uj']} uJ ({e['pj_per_op_achieved']} pJ/op achieved)" if e else ""))
        for name, c in s["phases"].items():
            print(f"  {name:20s} {c:8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
