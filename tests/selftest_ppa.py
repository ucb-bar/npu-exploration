"""Self-test of the recipe -> PPA-model adapter (models/ppa/ppa.py).

Three layers, mirroring the other selftests:
  1. the MAPPING, pinned to the model's own documented tapeout invocation --
     baseline must produce exactly the README's --rows/--prod/--stim spelling;
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

from models.ppa.ppa import PpaError, ppa_args, ppa_root, run_ppa  # noqa: E402
from config.recipe import load  # noqa: E402

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
    check("--stim = operand_fmt", got["--stim"] == "fp8", got["--stim"])
    check("--util tapeout default", got["--util"] == "0.965", got["--util"])

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
