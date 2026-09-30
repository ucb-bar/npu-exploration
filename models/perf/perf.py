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

from models.ppa.ppa import QRT_TABLE, PpaError, acc_rows, format_tokens, ppa_root, uses_lut, _workspace_head

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


#: stage out_dtype -> --out-fmt ("bf16" means no requant projection).
_OUT_TOK = {"bf16": "bf16", "f8E4M3FN": "fp8", "f8E5M2": "fp8e5m2"}

#: One LUT per 2**G rows of A / columns of W / rows of C: config.recipe.KERNEL_LUT_GROUP, the compiler's
#: LUT_GRANULARITY. A run recipe's lut.group overrides it.
DEFAULT_LUT_GROUP = 1


def perf_args(recipe, dtype: str, m: int, n: int, k: int, out_fmt: str, *,
              as_measured: bool = True, energy: bool = False, lut_group: int | None = None,
              tiles: tuple[int, int, int] | None = None, dma_bw: float | None = None,
              spad_kb: float | None = None, memory: bool = False) -> list[str]:
    """Map a hardware recipe, the run's operand format and one GEMM stage onto perf_model's CLI.

    A LUT format (models.ppa.ppa.uses_lut) runs with --lut and the chip's LUT layout: one LUT per 2**G rows of
    A, 2**G columns of W and 2**G rows of C, across the whole other dimension (the model's own default is one
    per 128x128 block). Each load moves only the tables the stage needs, as our emitter issues them
    (mxgemm_emit._emit_load_luts: N/2**G, M/2**G, M/2**G), not the full 64-table set per port the workspace's
    own kernels loaded (its --lut-full-set). The emitter also loads a C LUT for a bf16 output, which the model
    does not count: M/2**G tables, noted in the record.
    With --energy the power model gets the recipe's accumulator ladder, not the tapeout default.
    ``tiles``, ``dma_bw`` and ``spad_kb`` describe a production GEMM and need ``as_measured=False``.
    """
    try:
        _, _, _, tok = format_tokens(dtype)
    except PpaError as exc:
        raise PerfError(str(exc)) from exc
    out = _OUT_TOK.get(out_fmt, tok)
    args = ["--M", str(m), "--N", str(n), "--K", str(k),
            "--rows", str(recipe.dim), "--cols", str(recipe.dim),
            "--act", tok, "--wei", tok, "--out-fmt", out,
            "--clock-ns", str(recipe.clock_ns)]
    if uses_lut(dtype):
        g = 1 << (DEFAULT_LUT_GROUP if lut_group is None else lut_group)
        args += ["--lut", "--lut-a", str(g), str(k), "--lut-w", str(k), str(g), "--lut-c", str(g), str(n)]
    if as_measured:
        if tiles or dma_bw or spad_kb:
            raise PerfError("tiles / dma_bw / spad_kb describe a production GEMM; use as_measured=False (--ideal)")
        args.append("--as-measured")
    else:
        if tiles:
            args += ["--tm", str(tiles[0]), "--tn", str(tiles[1]), "--tk", str(tiles[2])]
        if dma_bw:
            args += ["--dma-bw", str(dma_bw)]
        if spad_kb:
            args += ["--spad-kb", str(spad_kb)]
    if energy:
        args += ["--energy", "--acc-rows", acc_rows(recipe)]
    if memory:
        args.append("--mem")
    return args


def memory_available() -> str | None:
    """None when perf_model --mem can run; else why not (the SRAM compiler tables are PDK data, not in the repo)."""
    try:
        root = ppa_root()
    except PpaError as exc:
        return str(exc)
    return None if (root / QRT_TABLE).exists() else f"no SRAM compiler tables at {root / QRT_TABLE}"


def spad_kb(recipe) -> float:
    """The hardware recipe's scratchpad capacity: banks x rows x dim bytes."""
    return recipe.banks * recipe.rows * recipe.dim / 1024


_PHASES = re.compile(
    r"cycles: kernel fixed (\d+) \+ first-tile load (\d+) \([^)]*\) \+ "
    r"tile setup exposed (\d+) \([^)]*\) \+ compute (\d+) \+ "
    r"requantizer drain (\d+) \([^)]*\) \+ LUT (\d+)")
_TOTAL = re.compile(
    r"total (\d+) cycles = ([\d.]+) us;\s+([\d.]+) M ops;.*"
    r"utilization ([\d.]+) %;\s+([\d.]+) Gop/s")
_TABLES = re.compile(r"LUT \d+ \((\d+) loads, (\d+) tables")
_MODE = re.compile(r"mode m(\d+) (\d+) ops/PE/cycle")
# --mem (memory_model.report with access counts): one row per memory, then the bandwidth and energy lines.
_MEM_ROW = re.compile(r"^(\S+)\s+(.+?)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
                      r"(?:\s+~\S*)?\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%")
_MEM_TIMING = re.compile(r"timing: clock [\d.]+ ns -> (.*)")
_MEM_BW = re.compile(r"scratchpad bandwidth: demand ([\d.]+) B/cycle during compute vs supply (\d+) B/cycle \((ok|STALL x[\d.]+)")
_MEM_TOTAL = re.compile(r"memory energy ([\d.]+) uJ macros \+ ([\d.]+) uJ accumulate logic = ([\d.]+) pJ per op "
                        r"\(([\d.]+) mW over the GEMM\)")


def _parse_memory(stdout: str) -> dict:
    start = stdout.find("memory (QRT model")
    tot = _MEM_TOTAL.search(stdout)
    if start < 0 or tot is None:
        raise PerfError(f"could not parse perf_model --mem output\n--- stdout ---\n{stdout[-1500:]}")
    per = {}
    for ln in stdout[start:].splitlines()[2:]:
        m = _MEM_ROW.match(ln.strip())
        if not m:
            break
        per[m.group(1)] = {"macro": m.group(2).strip(), "count": int(m.group(3)), "reads": int(m.group(11)),
                           "writes": int(m.group(12)), "dyn_uj": float(m.group(13)), "mw": float(m.group(14)),
                           "util_pct": float(m.group(15))}
    if not per:
        raise PerfError(f"perf_model --mem printed no memory rows\n--- stdout ---\n{stdout[start:start + 1500]}")
    bw, tm = _MEM_BW.search(stdout), _MEM_TIMING.search(stdout)
    return {"memories": per, "uj_macros": float(tot.group(1)), "uj_acc_logic": float(tot.group(2)),
            "pj_per_op": float(tot.group(3)), "mw": float(tot.group(4)),
            "timing": tm.group(1).strip() if tm else None,
            "spad_bandwidth": None if bw is None else {"demand_b_per_cycle": float(bw.group(1)),
                                                      "supply_b_per_cycle": int(bw.group(2)), "verdict": bw.group(3)}}
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
    tb = _TABLES.search(stdout)
    if tb:
        out["lut_loads"], out["lut_tables"] = int(tb.group(1)), int(tb.group(2))
    md = _MODE.search(stdout)
    if md:
        out["pe_mode"], out["ops_per_pe_cycle"] = int(md.group(1)), int(md.group(2))
    en = _ENERGY.search(stdout)
    if en:
        out["energy"] = {"power_mw_at_util": float(en.group(1)),
                         "uj": float(en.group(2)),
                         "pj_per_op_achieved": float(en.group(3))}
    if "memory (QRT model" in stdout:
        out["memory"] = _parse_memory(stdout)
    return out


def _run_one(recipe, dtype: str, m: int, n: int, k: int, out_fmt: str, *,
             as_measured: bool, energy: bool, **opts) -> dict:
    script = perf_model_path()
    args = perf_args(recipe, dtype, m, n, k, out_fmt, as_measured=as_measured, energy=energy, **opts)
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


def run_perf(recipe, dtype: str, stages, *, as_measured: bool = True, energy: bool = True, **opts) -> dict:
    """Predicted timeline for every mesh stage, plus kernel totals.

    ``stages`` is the pipeline's per-stage record: dicts carrying ``m``, ``k``,
    ``n`` and ``out_dtype`` (exactly what lands in metrics.json). ``opts`` go to
    :func:`perf_args` (lut_group, tiles, dma_bw, spad_kb). Raises
    :class:`PerfError`; callers skip and log.
    """
    script = perf_model_path()   # fail before any work if the model is absent
    no_memory = memory_available()
    per_stage = []
    for s in stages:
        if s.get("where", "mesh") != "mesh":       # host stages run on Rocket; the model prices the mesh
            continue
        res = _run_one(recipe, dtype, int(s["m"]), int(s["n"]), int(s["k"]),
                       str(s.get("out_dtype", "bf16")), as_measured=as_measured, energy=energy,
                       memory=no_memory is None, **opts)
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
    # Reported beside energy, never added to it: --energy already carries the measured Scratchpad block power
    # (memory_model / perf_model: "use one or the other for the memory part").
    if no_memory is not None:
        out["memory"] = {"available": False, "why": no_memory}
    else:
        out["memory"] = {"available": True,
                         "system": "radiance (as measured: the validated kernels ran there)" if as_measured else "rocket",
                         "uj_macros": round(sum(p["memory"]["uj_macros"] for p in per_stage), 3),
                         "uj_acc_logic": round(sum(p["memory"]["uj_acc_logic"] for p in per_stage), 3),
                         "timing": sorted({p["memory"]["timing"] for p in per_stage if p["memory"]["timing"]}),
                         "note": "separate from energy: that already contains the measured Scratchpad block"}
    out["model"] = {
        "source": "MxGemmini-workspace/ppa/perf/perf_model.py",
        "mode": "as_measured" if as_measured else "ideal",
        "clock_ns": recipe.clock_ns,
        "lut": uses_lut(dtype),
        "enable_lut": recipe.enable_lut,
        "validated": "+1.0/-2.2/-2.2 % total vs three kernel FSDBs (workspace README)",
        "workspace_head": _workspace_head(script.parents[1]),
    }
    if uses_lut(dtype) and any(str(s.get("out_dtype", "bf16")) == "bf16" for s in stages
                               if s.get("where", "mesh") == "mesh"):
        out["model"]["lut_note"] = ("a bf16-output stage: the emitter still loads its C LUT (M/2**G tables), "
                                    "which the model does not count")
    if energy and (recipe.prod_e, recipe.prod_m) != (4, 3):
        out["model"]["note"] = (f"perf_model has no --prod: its energy uses the e4m3 product, not the recipe's "
                                f"e{recipe.prod_e}m{recipe.prod_m} (models/ppa prices the recipe's own)")
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
    from config.recipe import RecipeError, load_hardware, load_run
    ap = argparse.ArgumentParser(
        description="Predicted GEMM timeline on a hardware recipe's machine (perf model)")
    ap.add_argument("--hw", "--config", dest="hw", default="baseline", help="hardware recipe name or .json path")
    ap.add_argument("--run", default="default", help="run recipe name or .json path (the operand format)")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out-fmt", default="bf16")
    ap.add_argument("--ideal", action="store_true",
                    help="drop --as-measured: overlapped setup, hw-issued mvins")
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--tiles", type=int, nargs=3, metavar=("TM", "TN", "TK"), help="with --ideal: tile sizes")
    ap.add_argument("--dma-bw", type=float, help="with --ideal: bytes per cycle the DMA delivers into the scratchpad")
    ap.add_argument("--recipe-spad", action="store_true",
                    help="with --ideal: the hardware recipe's scratchpad (banks x rows x dim) instead of the model's preset")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    stage = {"stage": 0, "m": a.m, "k": a.k, "n": a.n, "out_dtype": a.out_fmt}
    try:
        hw, run = load_hardware(a.hw), load_run(a.run)
        opts = {"lut_group": run.lut.group if run.lut else None}
        if a.ideal:
            opts.update(tiles=tuple(a.tiles) if a.tiles else None, dma_bw=a.dma_bw,
                        spad_kb=spad_kb(hw) if a.recipe_spad else None)
        elif a.tiles or a.dma_bw or a.recipe_spad:
            raise PerfError("--tiles / --dma-bw / --recipe-spad need --ideal")
        res = run_perf(hw, run.operand_fmt, [stage], as_measured=not a.ideal, energy=not a.no_energy, **opts)
    except (PerfError, RecipeError) as exc:
        print(f"perf: {exc}", file=sys.stderr)
        return 2
    if a.json:
        print(_json.dumps(res, indent=1))
    else:
        s = res["stages"][0]
        e = s.get("energy")
        print(f"{a.hw}: {a.m}x{a.k}x{a.n} -> {s['out_fmt']}: "
              f"{s['cycles_predicted']} cycles = {s['us']} us, "
              f"util {s['utilization_pct']}%, {s['gops']} Gop/s"
              + (f", {e['uj']} uJ ({e['pj_per_op_achieved']} pJ/op achieved)" if e else ""))
        for name, c in s["phases"].items():
            print(f"  {name:20s} {c:8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
