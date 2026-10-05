"""Silicon cost of a recipe: adapter to the MxGemmini area/power model.

The model lives OUTSIDE this repo (Amanda Shi's MxGemmini-workspace/ppa) and is
consumed as a black box, the same way compiler/operands.py consumes mxq: we call it,
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

Two additions, reported beside the totals and never folded into them:

* ``memory``: the on-chip memory inventory from the workspace's ``memory_model.py`` (SRAM macros per
  memory, area, leakage, access energy, access / cycle time against the recipe's clock). It needs the
  SRAM compiler tables (``tech/sram_qrt/qrt_table.csv``, PDK data kept out of the workspace repo); without
  them the record says so. The Gemmini total already contains the MEASURED Scratchpad block, so the
  inventory is not added to ``area_um2`` / ``power_mw``.
* ``pe``: the PE mode, ops per PE per cycle and RTL-test status of the run's format, from the workspace's
  ``pair_modes.spec`` (mxgen ``requiredPEMode``).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The model's calibration, not a recipe knob: it was fitted to one post-synthesis run (tstech16c,
#: tt0p8v25c) of the 16x16 mesh with --calib new --blocks-variant new --fmtset mxgemmini (ppa/README.md
#: "Quick start"). Clock and utilization come from the hardware recipe's implementation section.
CALIBRATED_DIM = 16
TECH, CORNER = "tstech16c", "tt0p8v25c"
CALIBRATION = ("--fmtset", "mxgemmini", "--calib", "new", "--blocks-variant", "new")
#: The MxAll machine as the workspace measured it (perf/perf_model.py energy(), README's quad line): the PE with
#: the quad arm, its calibration, and its Scratchpad and MxRequantizer (an FP8 E4M3 QuantLut with all finders).
CALIBRATION_LUT = ("--fmtset", "mxgemmini-all", "--calib", "all", "--blocks-variant", "all", "--lut", "fp8")

#: The hardware recipe's LUT unit (mx.lut.projFormat) -> the whole machine the workspace measured with that unit
#: (blocks/blocks.yaml: requantizer_new is "default tapeout config, FP6 LUT", requantizer_all "MxAll config, FP8
#: E4M3 codebook LUT with all deproject finders"). The build decides what is priced; the run's format decides
#: only the stimulus. A LUT unit the workspace has not measured on the current RTL, or no LUT unit, has no
#: price: ppa_args raises PpaError, which callers skip.
MACHINES = {
    "LutFP6E3M2": CALIBRATION,
    "LutFP8E4M3": CALIBRATION_LUT,
}

#: run operand_fmt -> (pair_modes token, compose --stim, ops per PE per cycle, perf_model --act/--wei).
#: The stim and products are what the workspace's pair_modes.spec(tok, tok, lut) gives for that format, LUT on
#: exactly for the LUT formats (config.scheme.is_codebook); tests/selftest_ppa.py holds this table equal to it.
#: fp8n / fp8qn are the NaN-safe E4M3 kernels (operands capped at 256; mxgen reads E4M3 272..448 as NaN).
FORMATS = {
    "fp8_e4m3":      ("e4m3",  "fp8n",     1, "fp8"),
    "fp8_e4m3_quad": ("e4m3q", "fp8qn",    4, "fp8"),      # perf: fp8 + --lut is the quad arm
    "fp8_e5m2":      ("e5m2",  "fp8e5m2",  4, "fp8e5m2"),
    "fp6_e3m2":      ("e3m2",  "fp6",      4, "fp6"),
    "fp6_e2m3":      ("e2m3",  "fp6e2m3q", 4, "fp6e2m3"),
    "fp4_e2m1":      ("e2m1",  "fp4",      4, "fp4"),
}


def format_tokens(dtype: str) -> tuple[str, str, int, str]:
    """A run recipe's operand format -> (pair_modes token, --stim, products, perf token). See FORMATS."""
    try:
        return FORMATS[dtype]
    except KeyError:
        raise PpaError(f"operand format {dtype!r} has no PPA model tokens; known: {sorted(FORMATS)}") from None


def uses_lut(dtype: str) -> bool:
    """Does this format reach the mesh through LUTs? The kernels compiled for it carry them. What hardware is
    priced is the recipe's (MACHINES), not this."""
    from config.scheme import is_codebook
    return is_codebook(dtype)


def operand_family(dtype: str) -> str:
    """A run recipe's operand format -> the models' format family token (fp8 | fp6 | fp4)."""
    fam = dtype.split("_", 1)[0]
    if fam not in ("fp8", "fp6", "fp4"):
        raise PpaError(f"operand format {dtype!r} has no fp8/fp6/fp4 family")
    return fam


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


def ppa_args(recipe, dtype: str = "fp8_e4m3") -> list[str]:
    """Map a hardware recipe and a run's operand format onto compose_gemmini's CLI.

    The model speaks (expWidth, sigWidth); the recipe's properties speak
    (e, m = sig-1), so sig = m + 1 here. The acc ladder is run-length encoded
    per the model's --rows grammar; baseline must come out exactly as the
    model's documented tapeout invocation (tests/selftest_ppa.py pins this).
    The machine priced is the recipe's LUT unit's (MACHINES); the format sets the stimulus, and a LUT format
    the pair_modes products, matching the workspace's README quad line.
    """
    _, stim, products, _ = format_tokens(dtype)
    proj = recipe.lut.projection if recipe.lut is not None else None
    if proj not in MACHINES:
        raise PpaError(f"{recipe.name}: MxGemmini-workspace has no measurement of a machine with "
                       f"{'no LUT unit' if proj is None else f'a {proj} LUT unit'} on the current RTL; it has "
                       f"{', '.join(MACHINES)} (blocks/blocks.yaml)")
    args = ["--rows", acc_rows(recipe),
            "--prod", f"{recipe.prod_e},{recipe.prod_m + 1}",
            "--cols", str(recipe.dim),
            "--stim", stim,
            *MACHINES[proj],
            "--util", str(recipe.utilization), "--clock-ns", str(recipe.clock_ns)]
    if uses_lut(dtype):
        args += ["--products", str(products)]   # without a LUT the model's own default is the same number
    return args


def acc_rows(recipe) -> str:
    """The recipe's accumulator ladder in the models' --rows / --acc-rows spelling."""
    return _run_length([(e, m + 1) for e, m in zip(recipe.acc_e, recipe.acc_m)])


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


def run_ppa(recipe, dtype: str = "fp8_e4m3") -> dict:
    """Area/power/energy for the machine this hardware recipe describes, fed ``dtype`` operands.
    Raises PpaError."""
    root = ppa_root()
    args = ppa_args(recipe, dtype)
    cmd = [sys.executable, str(root / "compose_gemmini.py"), *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=root)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PpaError(f"compose_gemmini failed to run: {exc}") from exc
    if r.returncode != 0:
        if "missing leaf point" in r.stderr:            # the workspace measured this machine, not with this format
            raise PpaError(f"{recipe.name} running {dtype}: MxGemmini-workspace has no measurement of this "
                           f"mesh with that stimulus ({r.stderr.strip().splitlines()[-1]})")
        raise PpaError(f"compose_gemmini exited {r.returncode}:\n{r.stderr[-1500:]}")
    out = _parse(r.stdout)
    out["model"] = {
        "source": "MxGemmini-workspace/ppa/compose_gemmini.py",
        "args": " ".join(args),
        "tech": TECH, "corner": CORNER,
        "clock_ns": recipe.clock_ns, "util": recipe.utilization,
        "post_synthesis": True,
        "calibrated_dim": CALIBRATED_DIM,
        "calibrated": recipe.dim == CALIBRATED_DIM,
        "lut": uses_lut(dtype),
        "lut_unit": recipe.lut.projection,
        "enable_lut": recipe.enable_lut,
        "workspace_head": _workspace_head(root),
    }
    out["pe"] = pe_spec(dtype)
    out["memory"] = memory_inventory(recipe)
    return out


#: The SRAM compiler tables memory_model.py reads, relative to the workspace root. PDK data: not in the repo.
QRT_TABLE = Path("tech/sram_qrt/qrt_table.csv")

#: The workspace's memory preset for the chip our hardware recipes describe: "rocket" is the Gemmini-only Rocket
#: SoC, MxGemminiRocketConfig = gemmini-mx-cleanup standaloneMxFPConfig, the machine config/hardware/baseline.json
#: states. The recipe gives the scratchpad; accumulator, scale memory, L2 and L1 come from the preset.
MEMORY_SYSTEM = "rocket"
GEMMINI_MEMORIES = ("smem", "acc", "scale")


def _workspace_module(name: str):
    """Import one of the workspace's pure-Python modules (pair_modes, memory_model) from ppa_root()."""
    import importlib
    root = str(ppa_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(name)


def pe_spec(dtype: str) -> dict:
    """The PE mode this format runs in, as the workspace's pair_modes.spec states it (activation = weight = dtype)."""
    tok = format_tokens(dtype)[0]
    try:
        sp = _workspace_module("pair_modes").spec(tok, tok, uses_lut(dtype))
    except Exception as exc:                                   # the workspace moved: report, do not fail ppa
        return {"available": False, "why": f"pair_modes.spec: {exc}"}
    return {"mode": sp["mode"], "ops_per_pe_cycle": sp["products"], "rtl_tested": sp["rtl_ok"] is not None,
            "rtl_ok": sp["rtl_ok"], "note": sp["note"] or None}


def memory_inventory(recipe) -> dict:
    """The recipe machine's on-chip memories, from the workspace's memory_model.py. Never raises."""
    try:
        root = ppa_root()
        if not (root / QRT_TABLE).exists():
            return {"available": False,
                    "why": f"no SRAM compiler tables at {root / QRT_TABLE} (PDK data, not in the workspace repo)"}
        import argparse
        mm = _workspace_module("memory_model")
        ap = argparse.ArgumentParser()
        mm.add_args(ap, preset=MEMORY_SYSTEM)
        a = mm.apply_system(ap.parse_args([]), preset=MEMORY_SYSTEM)
        a.smem_kb = recipe.banks * recipe.rows * recipe.dim / 1024      # banks x rows of dim bytes
        a.smem_banks, a.smem_words, a.smem_wordbytes = recipe.banks, 1, recipe.dim
        a.acc_cols = recipe.dim
        mems = mm.resolve(mm.inventory(a), mm.load_qrt(), a.vdd, a.temp, a.vt)
    except Exception as exc:
        return {"available": False, "why": f"memory_model: {exc}"}
    per = {name: {"macro": m["macro"], "count": m["count"], "exact": m["exact"],
                  "area_um2": round(m["area_total_um2"], 1), "leak_mw": round(m["leak_total_uW"] / 1e3, 4),
                  "e_read_pj": round(m["e_read_pJ"], 3), "e_write_pj": round(m["e_write_pJ"], 3),
                  "taa_ns": m["taa_ns"], "tcyc_ns": m["tcyc_ns"], "meets_clock": m["tcyc_ns"] <= recipe.clock_ns}
           for name, m in mems.items()}
    gem = [n for n in per if n in GEMMINI_MEMORIES]
    return {"available": True, "system": MEMORY_SYSTEM, "vdd": a.vdd, "temp": a.temp, "vt": a.vt,
            "from_recipe": ["smem"], "memories": per,
            "gemmini_area_um2": round(sum(per[n]["area_um2"] for n in gem), 1),
            "gemmini_leak_mw": round(sum(per[n]["leak_mw"] for n in gem), 4),
            "slower_than_clock": [n for n, m in per.items() if not m["meets_clock"]],
            "note": "separate from area_um2 / power_mw: the Gemmini total already has the measured Scratchpad block"}


def line(ppa: dict) -> str:
    """The PPA line run_kernel.py prints: the recipe machine's silicon cost."""
    return (f"PPA      {ppa['area_um2']/1e3:.1f}k um2   {ppa['power_mw']:.1f} mW   "
            f"{ppa['pj_per_op']:.2f} pJ/op   "
            f"(post-syn {ppa['model']['tech']} model, calibrated={ppa['model']['calibrated']})")


def main() -> int:
    import argparse
    import json as _json
    from config.recipe import RecipeError, check, load_hardware, load_run
    ap = argparse.ArgumentParser(description="Silicon cost of a hardware recipe (PPA model)")
    ap.add_argument("--hw", "--config", dest="hw", default="baseline", help="hardware recipe name or .json path")
    ap.add_argument("--run", default="default", help="run recipe name or .json path (the operand format)")
    ap.add_argument("--json", action="store_true", help="machine-readable output only")
    a = ap.parse_args()
    try:
        hw, run = load_hardware(a.hw), load_run(a.run)
        check(hw, run, "perplexity")        # the format, reducer and LUT unit agree; not the kernel path's limits
        res = run_ppa(hw, run.operand_fmt)
    except (PpaError, RecipeError) as exc:
        print(f"ppa: {exc}", file=sys.stderr)
        return 2
    if a.json:
        print(_json.dumps(res, indent=1))
    else:
        print(f"{a.hw}: {res['area_um2']/1e3:.1f}k um2  {res['power_mw']:.1f} mW  "
              f"{res['pj_per_op']:.2f} pJ/op  ({res['throughput_gops']:.1f} Gop/s; "
              f"post-syn {TECH} model, calibrated={res['model']['calibrated']})")
        for name, b in res["blocks"].items():
            print(f"  {name:44s} {b['area_um2']/1e3:9.1f}k  {b['power_mw']:7.1f} mW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
