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

from models.ppa.ppa import CALIBRATION_LUT, FORMATS, PpaError, ppa_args, ppa_root, run_ppa, uses_lut  # noqa: E402
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
    quad = dict(zip(*[iter(ppa_args(r, "fp8_e4m3_quad"))] * 2))
    check("quad: fp8qn on the LUT PE, 4 products",
          (quad["--stim"], quad["--fmtset"], quad["--calib"], quad["--blocks-variant"], quad["--lut"], quad["--products"])
          == ("fp8qn", "mxgemmini-all", "all", "all", "fp8", "4"), str(quad))
    check("LUT formats are exactly config.scheme's codebook formats",
          sorted(f for f in FORMATS if uses_lut(f)) == ["fp6_e2m3", "fp6_e3m2", "fp8_e4m3_quad", "fp8_e5m2"])
    import copy
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
        rq = run_ppa(r, "fp8_e4m3_quad")
        check("quad priced on the bigger LUT PE (area > baseline fp8)", rq["area_um2"] > run_ppa(r)["area_um2"] * 1.2,
              f"{rq['area_um2']/1e3:.1f}k")
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
