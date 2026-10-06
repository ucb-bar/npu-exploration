"""Self-test of the recipe -> PPA-model adapter (models/ppa/ppa.py).

Three layers, mirroring the other selftests:
  1. the MAPPING, pinned to the model's own documented tapeout invocation --
     baseline must produce exactly the README's --rows/--prod/--stim spelling
     (fp8n, the NaN-safe E4M3 kernel) -- and every operand format priced with the
     tokens the workspace's own pair_modes.spec gives it, LUT formats on the LUT PE;
  2. a LIVE run against the real model (skipped cleanly when the workspace is
     absent), checked for internal consistency rather than frozen totals, so an
     upstream recalibration does not break this repo;
  3. the FAIL-SOFT path: a missing workspace raises PpaError and nothing else.

No hardware, no toolchain, no spike.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.ppa.ppa import CALIBRATION, CALIBRATION_LUT, FORMATS, PpaError, ppa_args, ppa_root, run_ppa, uses_lut  # noqa: E402
from config.recipe import load_hardware as load, parse_hardware  # noqa: E402

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append(ok)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


def main() -> int:
    r = load("baseline")

    print("mapping: baseline -> the model's documented tapeout invocation")
    args = ppa_args(r)
    got = dict(zip(args[::2], args[1::2]))
    check("--rows run-length encoding", got["--rows"] == "8x4,5 2x4,6 5x4,7 1x8,8",
          got["--rows"])
    check("--prod (e, m+1)", got["--prod"] == "4,4", got["--prod"])
    check("--cols = dim", got["--cols"] == "16", got["--cols"])
    check("--stim = the NaN-safe E4M3 kernel (README tapeout line)", got["--stim"] == "fp8n", got["--stim"])
    check("fp8_e4m3 on the tapeout PE, no LUT", got["--fmtset"] == "mxgemmini" and "--lut" not in got)
    check("--util = implementation.utilization", got["--util"] == "0.965", got["--util"])
    check("--clock-ns = implementation.clock_ns", got["--clock-ns"] == "2.0", got["--clock-ns"])
    check("an fp4 run stimulates fp4", dict(zip(*[iter(ppa_args(r, "fp4_e2m1"))] * 2))["--stim"] == "fp4")
    print("the recipe's LUT unit picks the machine priced; the run's format only the stimulus")
    lut8 = load("lut_fp8e4m3")
    a6 = ppa_args(r, "fp6_e3m2")
    check("baseline (LutFP6E3M2) running fp6_e3m2: the tapeout machine, 4 products",
          " ".join(CALIBRATION) in " ".join(a6) and dict(zip(a6[::2], a6[1::2]))["--products"] == "4", " ".join(a6))
    a8 = ppa_args(lut8, "fp8_e4m3")
    check("lut_fp8e4m3 (LutFP8E4M3) running direct fp8_e4m3: still the MxAll machine",
          " ".join(CALIBRATION_LUT) in " ".join(a8) and "--products" not in a8, " ".join(a8))
    import copy
    raw0 = copy.deepcopy(r.raw)
    raw0["mx"]["lut"] = None
    for hw, unit in ((load("lut_fp8e5m2"), "LutFP8E5M2"), (load("lut_fp6e2m3"), "LutFP6E2M3"),
                     (parse_hardware(raw0), "no LUT unit")):
        try:
            ppa_args(hw, "fp8_e4m3")
            check(f"{unit}: not measured by the workspace -> PpaError (callers skip)", False, "accepted")
        except PpaError as exc:
            check(f"{unit}: not measured by the workspace -> PpaError (callers skip)", unit in str(exc), str(exc)[:100])
    quad = dict(zip(*[iter(ppa_args(lut8, "fp8_e4m3_quad"))] * 2))
    check("quad: fp8qn on the LUT PE, 4 products",
          (quad["--stim"], quad["--fmtset"], quad["--calib"], quad["--blocks-variant"], quad["--lut"], quad["--products"])
          == ("fp8qn", "mxgemmini-all", "all", "all", "fp8", "4"), str(quad))
    check("LUT formats are exactly config.scheme's codebook formats",
          sorted(f for f in FORMATS if uses_lut(f)) == ["fp6_e2m3", "fp6_e3m2", "fp8_e4m3_quad", "fp8_e5m2"])
    raw = copy.deepcopy(r.raw)
    raw["implementation"].update(clock_ns=1.25, utilization=0.8)
    fast = dict(zip(*[iter(ppa_args(parse_hardware(raw)))] * 2))
    check("a recipe's own clock and utilization reach the model",
          (fast["--clock-ns"], fast["--util"]) == ("1.25", "0.8"), f"{fast['--clock-ns']} {fast['--util']}")

    w = load("wide_acc")
    wrows = dict(zip(ppa_args(w)[::2], ppa_args(w)[1::2]))["--rows"]
    check("wide_acc collapses to one group", wrows == "16x8,8", wrows)

    print("live model (skipped if MxGemmini-workspace is absent)")
    try:
        root = ppa_root()
    except PpaError as exc:
        print(f"  skip  {exc}")
        root = None
    if root is not None:
        import importlib
        sys.path.insert(0, str(root))
        pm, drv = importlib.import_module("pair_modes"), importlib.import_module("ppa_driver")
        for fmt, (tok, stim, prods, _) in FORMATS.items():
            sp = pm.spec(tok, tok, uses_lut(fmt))
            check(f"{fmt}: stim and products are pair_modes.spec's ({sp['stim']}, {sp['products']})",
                  (sp["stim"], sp["products"]) == (stim, prods), f"ours {stim}, {prods}")
            check(f"{fmt}: {stim} is a stimulus the model parses", drv.STIM_RE.match(stim) is not None)
        rq = run_ppa(lut8, "fp8_e4m3_quad")
        check("quad priced on the bigger LUT PE (area > baseline fp8)", rq["area_um2"] > run_ppa(r)["area_um2"] * 1.2,
              f"{rq['area_um2']/1e3:.1f}k")
        check("one build, one area: lut_fp8e4m3 running fp8_e4m3 has the quad run's area",
              run_ppa(lut8, "fp8_e4m3")["area_um2"] == rq["area_um2"])
        check("one build, one area: baseline running fp6_e3m2 has baseline fp8's area",
              run_ppa(r, "fp6_e3m2")["area_um2"] == run_ppa(r)["area_um2"])
        try:
            run_ppa(lut8, "fp6_e2m3")
            check("a format the workspace never measured on that mesh -> PpaError (callers skip)", False, "accepted")
        except PpaError as exc:
            check("a format the workspace never measured on that mesh -> PpaError (callers skip)",
                  "no measurement" in str(exc), str(exc)[:100])
        res = run_ppa(r)
        blk = sum(b["area_um2"] for b in res["blocks"].values())
        check("total area == sum of blocks (rounding)", abs(res["area_um2"] - blk) < 500,
              f"{res['area_um2']:.0f} vs {blk:.0f}")
        pblk = sum(b["power_mw"] for b in res["blocks"].values())
        check("total power == sum of blocks (rounding)", abs(res["power_mw"] - pblk) < 1.0,
              f"{res['power_mw']:.1f} vs {pblk:.1f}")
        check("area in a plausible band", 5e5 < res["area_um2"] < 2e6,
              f"{res['area_um2']/1e3:.1f}k um2")
        check("mesh block present", "MeshWithDelays" in res["blocks"])
        check("pJ/op parsed", 0.5 < res["pj_per_op"] < 50, str(res["pj_per_op"]))
        check("metadata: calibrated at dim 16", res["model"]["calibrated"] is True)
        wa = run_ppa(w)
        na = run_ppa(load("narrow_prod"))
        check("wide_acc mesh > baseline mesh (area)",
              wa["blocks"]["MeshWithDelays"]["area_um2"]
              > res["blocks"]["MeshWithDelays"]["area_um2"],
              f"{wa['blocks']['MeshWithDelays']['area_um2']/1e3:.1f}k vs "
              f"{res['blocks']['MeshWithDelays']['area_um2']/1e3:.1f}k")
        check("narrow_prod mesh < baseline mesh (area)",
              na["blocks"]["MeshWithDelays"]["area_um2"]
              < res["blocks"]["MeshWithDelays"]["area_um2"],
              f"{na['blocks']['MeshWithDelays']['area_um2']/1e3:.1f}k")

    if root is not None:
        print("B: PE facts and the memory inventory (beside the totals, never in them)")
        from models.ppa.ppa import pe_spec
        for fmt, (tok, _, prods, _) in FORMATS.items():
            p = pe_spec(fmt)
            check(f"{fmt}: pe = pair_modes.spec (mode {p.get('mode')}, {p.get('ops_per_pe_cycle')} ops/PE/cycle, "
                  f"rtl_ok {p.get('rtl_ok')})", p.get("ops_per_pe_cycle") == prods and p["mode"] == pm.spec(tok, tok, uses_lut(fmt))["mode"])
        real = run_ppa(r)
        check("real workspace: memory unavailable, and says why", real["memory"]["available"] is False
              and "SRAM compiler tables" in real["memory"]["why"], real["memory"].get("why", "")[:90])
        import shutil
        sys.path.insert(0, str(REPO / "tests"))
        from ppa_memfixture import synthetic_workspace
        fx = synthetic_workspace(root)
        old_root = os.environ.get("MX_PPA_ROOT")
        os.environ["MX_PPA_ROOT"] = str(fx)
        try:
            syn = run_ppa(r)
        finally:
            if old_root is None:
                os.environ.pop("MX_PPA_ROOT", None)
            else:
                os.environ["MX_PPA_ROOT"] = old_root
            shutil.rmtree(fx.parent, ignore_errors=True)
        mem = syn["memory"]
        check("synthetic SRAM table: the inventory runs (plumbing only; numbers invented)", mem.get("available") is True,
              str(mem.get("why", ""))[:120])
        if mem.get("available"):
            check("inventory has scratchpad, accumulator, scale memory", {"smem", "acc", "scale"} <= set(mem["memories"]))
            check("scratchpad is the recipe's: 4 banks x 4096 rows x 16 B",
                  mem["memories"]["smem"]["count"] >= 4 and mem["from_recipe"] == ["smem"])
            check("gemmini memory area = sum of smem + acc + scale",
                  abs(mem["gemmini_area_um2"] - sum(mem["memories"][n]["area_um2"] for n in ("smem", "acc", "scale"))) < 1)
            check("timing checked against the recipe clock (synthetic tcyc 0.8 ns < 2 ns)", mem["slower_than_clock"] == [])
            check("the totals do not move with the memory model",
                  (syn["area_um2"], syn["power_mw"], syn["pj_per_op"]) == (real["area_um2"], real["power_mw"], real["pj_per_op"]))

    print("fail-soft: bad MX_PPA_ROOT raises PpaError, nothing else")
    old = os.environ.get("MX_PPA_ROOT")
    os.environ["MX_PPA_ROOT"] = "/nonexistent-ppa"
    try:
        try:
            ppa_root()
            check("PpaError raised", False)
        except PpaError:
            check("PpaError raised", True)
    finally:
        if old is None:
            os.environ.pop("MX_PPA_ROOT", None)
        else:
            os.environ["MX_PPA_ROOT"] = old

    print()
    if all(CHECKS):
        print(f"ALL {len(CHECKS)} CHECKS PASSED -- the recipe->PPA adapter is wired correctly.")
        return 0
    print(f"{CHECKS.count(False)} of {len(CHECKS)} checks FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
