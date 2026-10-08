"""Predicted kernel timeline: adapter to the MxGemmini performance model.

Where ``models/ppa/ppa.py`` prices the MACHINE (area/power, kernel-independent),
this prices the RUN: cycles, wall time, utilization and energy of each GEMM
stage on the machine the recipe describes. The model is Amanda Shi's
``MxGemmini-workspace/ppa/perf/perf_model.py`` -- RTL-FSDB-calibrated
(validated within +1.0/-2.2/-2.2 % on three real kernels) -- and, like the
PPA model, it is consumed as a black box: called, never reimplemented.

Two sources of time meet in the record, and :func:`merge_measured` keeps them apart:

* the top level of ``perf`` (``cycles``, ``us``, ``gops``, ``utilization_pct``) is MEASURED: spike's
  cycle model (libgemmini ``perf/``, run beside the bits, ``metrics.timing``) timing the ELF our
  emitter wrote, host code included;
* ``perf.estimate`` is this model's timeline for the same GEMM shapes (``cycles_predicted``, ``us``,
  ``gops``, ``utilization_pct``, ``phases``), calibrated on the workspace's own kernels.

What only this model gives stays at the top level: ``energy`` (computed on ITS timeline and
utilization, which ``energy.basis_cycles`` names), ``pe_mode`` / ``ops_per_pe_cycle``, the LUT
layout, ``memory``. Without a spike run the measured fields are ``None`` and ``cycles_source`` says
why. The standalone CLI below prints this model's numbers alone.

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

def perf_args(recipe, dtype: str, m: int, n: int, k: int, out_fmt: str, *,
              as_measured: bool = True, energy: bool = False, lut_group: int | None = None,
              tiles: tuple[int, int, int] | None = None, dma_bw: float | None = None,
              spad_kb: float | None = None, memory: bool = False) -> list[str]:
    """Map a hardware recipe, the run's operand format and one GEMM stage onto perf_model's CLI.

    A LUT format (models.ppa.ppa.uses_lut) runs with --lut and the chip's LUT layout: one LUT per 2**G rows of
    A, 2**G columns of W and 2**G rows of C, across the whole other dimension, G being ``lut_group`` (the run
    recipe's lut.group, required for a LUT format) (the model's own default is one
    per 128x128 block). Each load moves only the tables the stage needs, as our emitter issues them
    (mxgemm_emit._emit_load_luts: N/2**G, M/2**G, M/2**G), not the full 64-table set per port the workspace's
    own kernels loaded (its --lut-full-set). The emitter also loads a C LUT for a bf16 output, which the model
    does not count: M/2**G tables, noted in the record.
    As measured, a span that is not a multiple of 2**G or needs more LUTs than the build's mx.lut.numEntries is
    refused, as the emitter refuses it; ``as_measured=False`` models it anyway (a production GEMM).
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
        if lut_group is None:
            raise PerfError(f"{dtype} is a LUT format: lut_group (the run recipe's lut.group) is required")
        g = 1 << lut_group
        if as_measured:                     # the kernel the emitter would build: whole groups, within the tables
            for table, sel, span, axis in (("B", 0, n, "N"), ("A", 1, m, "M"), ("C", 2, m, "M")):
                if span % g:
                    raise PerfError(f"{axis}={span} is not a multiple of 2**G = {g} (lut.group {lut_group})")
                if recipe.lut is not None and span >> lut_group > recipe.lut.tables[sel]:
                    raise PerfError(f"{table} table: {axis}={span} at G={lut_group} needs {span >> lut_group} "
                                    f"LUTs, and {recipe.name} holds {recipe.lut.tables[sel]} (mx.lut.numEntries)")
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
    else:
        # perf_model's --energy runs compose_gemmini itself and prints why when that fails (workspace
        # >= 3d28713: its default --system rocket wants the PDK SRAM table); the record keeps the reason.
        fail = re.search(r"\[energy\] power model failed: (.*)", stdout)
        if fail:
            out["energy_error"] = fail.group(1).strip()
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
        res["ops"] = int(s["m"]) * int(s["k"]) * int(s["n"])        # exact; perf_model prints m_ops to 2 decimals
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
    energy_errors = [p.pop("energy_error") for p in per_stage if "energy_error" in p]
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
    if energy and "energy" not in out:
        out["model"]["energy_note"] = ("perf_model --energy gave no energy: its own compose_gemmini call passes no "
                                       "--system, and the workspace's default (rocket, since 3d28713) needs the PDK "
                                       "SRAM table")
        if energy_errors:                 # compose's last stdout line, as perf_model relays it
            out["model"]["energy_error"] = " ".join(energy_errors[0].split())
    return out


#: Amanda's timeline fields that the cycle model's measurement replaces; they move under ``estimate``.
_ESTIMATE_TOTAL = ("total_cycles_predicted", "total_us", "utilization_pct_min")
_ESTIMATE_STAGE = ("cycles_predicted", "us", "gops", "utilization_pct", "phases", "args", "energy")
MEASURED_SOURCE = ("libgemmini cycle model (GEMMINI_MODE=both), timing the ELF the emitter wrote; "
                   "metrics.timing has its counters")


def merge_measured(perf: dict, timing: dict | None, *, clock_ns: float, recipe=None,
                   dtype: str = "fp8_e4m3") -> dict:
    """Put spike's measured time at the top of the ``perf`` record, this model's timeline under
    ``estimate``, and the kernel's energy on the measured window. In place; returns ``perf``.

    ``timing`` is ``metrics["timing"]`` (grade.pipeline): ``cycles`` is the ELF's whole measured window,
    ``stage_cycles`` the per-stage windows where the ELF reports them, ``summary.mesh_busy_cycles`` the
    cycles the array computed. Utilization is mesh busy over the window; ``gops`` the kernel's ops over the
    window at the recipe's clock. A stage without its own window keeps ``cycles: None``. Without
    ``timing`` (spike did not run) every measured field is ``None`` and ``cycles_source`` says so.

    ``energy`` is ``models.ppa.ppa.run_energy`` on ``recipe``'s machine: compose_gemmini's power at the
    measured utilization times the measured window, pJ/op over the kernel's ops. Without ``recipe``
    (standalone use), without a window or without a mesh-busy count it is ``None`` and
    ``model.measured.energy_note`` says why. perf_model's own ``--energy`` figure, when it ran, is on its
    predicted timeline and moves under ``estimate.energy`` (per stage too).
    Keys that only this model produces (``m_ops``, ``pe_mode``, LUT layout, ``memory``) stay where they were.
    """
    if "estimate" in perf:
        return perf                                  # already merged
    estimate = {k: perf.pop(k) for k in _ESTIMATE_TOTAL if k in perf}
    estimate["stages"] = [{"stage": s.get("stage"), **{k: s.pop(k) for k in _ESTIMATE_STAGE if k in s}}
                          for s in perf["stages"]]
    perf.pop("spike_cycles", None)
    perf.pop("spike_functional_cycles", None)
    ops = sum(float(s["ops"]) if s.get("ops") else float(s.get("m_ops") or 0) * 1e6 for s in perf["stages"])
    if "energy" in perf:                          # perf_model --energy: on its own timeline
        estimate["energy"] = {**perf.pop("energy"), "basis_cycles": estimate.get("total_cycles_predicted"),
                              "basis": "perf_model's own timeline and utilization, not the measured window"}
    energy, energy_note = None, None
    if timing is None or timing.get("cycles") is None:
        cyc = None
        for s in perf["stages"]:
            s["cycles"] = s["us"] = None
        measured = {"cycles": None, "us": None, "gops": None, "utilization_pct": None,
                    "cycles_source": "none: spike did not run; estimate holds perf_model's timeline"}
        energy_note = "no measured window (spike did not run)"
    else:
        cyc = int(timing["cycles"])
        busy = (timing.get("summary") or {}).get("mesh_busy_cycles")
        seconds = cyc * clock_ns * 1e-9
        measured = {"cycles": cyc,
                    "us": round(cyc * clock_ns / 1e3, 2),
                    "gops": round(ops / seconds / 1e9, 1) if seconds else None,
                    "utilization_pct": round(100.0 * busy / cyc, 1) if (busy is not None and cyc) else None,
                    "cycles_source": MEASURED_SOURCE}
        stage_cycles = timing.get("stage_cycles") or {}
        for s in perf["stages"]:
            sc = stage_cycles.get(str(s.get("stage")))
            s["cycles"] = sc
            s["us"] = round(sc * clock_ns / 1e3, 2) if sc is not None else None
        if recipe is None:
            energy_note = "no recipe given (standalone merge): run_energy needs the machine"
        elif not busy:
            energy_note = "the cycle model reported no mesh busy cycles: no utilization to price power at"
        elif not ops:
            energy_note = "the stage records carry no ops"
        else:
            from models.ppa.ppa import PpaError, run_energy
            try:
                energy = run_energy(recipe, dtype, util=busy / cyc, cycles=cyc, ops=ops)
            except PpaError as exc:
                energy_note = str(exc)
    model = perf.pop("model")
    measured_model = {"source": MEASURED_SOURCE, "clock_ns": clock_ns}
    if energy_note:
        measured_model["energy_note"] = energy_note
    ordered = {**measured, "energy": energy, "stages": perf.pop("stages"), **perf, "estimate": estimate,
               "model": {"measured": measured_model, "estimate": model}}
    perf.clear()
    perf.update(ordered)
    return perf


def line(perf: dict) -> str:
    """The PERF line run_kernel.py prints: measured time first, this model's estimate and energy after."""
    e = perf.get("energy")
    est = perf.get("estimate")
    if est is None:                                   # a record before merge_measured (standalone use)
        return (f"PERF     {perf['total_cycles_predicted']} cycles predicted   "
                f"{perf['total_us']:.1f} us   util {perf['utilization_pct_min']:.1f}%"
                + (f"   {e['uj_kernel']:.2f} uJ ({e['pj_per_op_achieved']:.1f} pJ/op achieved)" if e else ""))
    if perf.get("cycles") is None:
        head = "PERF     no measured cycles (spike did not run)"
    else:
        head = (f"PERF     {perf['cycles']} cycles (spike cycle model)   {perf['us']:.1f} us   "
                f"mesh util {perf['utilization_pct']:.1f}%"
                + (f"   {e['uj_kernel']:.2f} uJ ({e['pj_per_op']:.2f} pJ/op)" if e else ""))
    ee = est.get("energy")
    return (head + f"   | estimate {est['total_cycles_predicted']} cycles"
            + (f", {ee['uj_kernel']:.2f} uJ ({ee['pj_per_op_achieved']:.1f} pJ/op, on the estimate)" if ee else ""))


def main() -> int:
    import argparse
    import json as _json
    from config.recipe import RecipeError, check, load_hardware, load_run
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
        check(hw, run, "perplexity")        # the format, reducer and LUT unit agree; not the kernel path's limits
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
