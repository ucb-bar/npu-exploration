"""Self-test of the recipe -> performance-model adapter (config/perf.py).

Same three layers as tests/selftest_ppa.py:
  1. the MAPPING: recipe + GEMM shape -> exactly the model's CLI spelling;
  2. a LIVE run against the real model (skipped cleanly when the workspace is
     absent), checked for internal consistency and physics ordering rather
     than frozen totals, so an upstream recalibration does not break this repo;
  3. the FAIL-SOFT path: a missing workspace raises PerfError and nothing else.

No hardware, no toolchain, no spike.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config.perf import PerfError, perf_args, perf_model_path, run_perf  # noqa: E402
from config.recipe import load  # noqa: E402

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append(ok)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


def main() -> int:
    r = load("baseline")

    print("mapping: baseline + 64x64x64 -> the model's CLI")
    args = perf_args(r, 64, 64, 64, "f8E4M3FN")
    got = dict(zip(args[::2], args[1::2]))
    check("--M/--N/--K", (got["--M"], got["--N"], got["--K"]) == ("64", "64", "64"))
    check("--rows/--cols = dim", (got["--rows"], got["--cols"]) == ("16", "16"))
    check("--act/--wei = operand family", (got["--act"], got["--wei"]) == ("fp8", "fp8"))
    check("--out-fmt f8E4M3FN -> fp8", got["--out-fmt"] == "fp8", got["--out-fmt"])
    check("--as-measured on by default", "--as-measured" in args)
    args_bf = perf_args(r, 64, 64, 64, "bf16")
    check("--out-fmt bf16 passes through",
          dict(zip(args_bf[::2], args_bf[1::2]))["--out-fmt"] == "bf16")

    print("live model (skipped if MxGemmini-workspace or perf/ is absent)")
    try:
        perf_model_path()
        live = True
    except PerfError as exc:
        print(f"  skip  {exc}")
        live = False
    if live:
        stage = {"stage": 0, "m": 64, "k": 64, "n": 64, "out_dtype": "f8E4M3FN"}
        res = run_perf(r, [stage])
        s = res["stages"][0]
        check("total == sum of phases",
              s["cycles_predicted"] == sum(s["phases"].values()),
              f"{s['cycles_predicted']} vs {sum(s['phases'].values())}")
        check("utilization in (0, 100]", 0 < s["utilization_pct"] <= 100,
              str(s["utilization_pct"]))
        check("compute phase nonzero", s["phases"]["compute"] > 0)
        check("fp8 without LUT loads nothing", s["phases"]["lut"] == 0)
        check("energy parsed (uJ > 0)",
              "energy" in res and res["energy"]["uj_kernel"] > 0,
              str(res.get("energy")))
        check("kernel totals = stage sums",
              res["total_cycles_predicted"] == s["cycles_predicted"])

        big = run_perf(r, [{"stage": 0, "m": 1024, "k": 1024, "n": 1024,
                            "out_dtype": "bf16"}], energy=False)
        check("1024^3 utilization > 64^3 (amortized setup)",
              big["stages"][0]["utilization_pct"] > s["utilization_pct"],
              f"{big['stages'][0]['utilization_pct']}% vs {s['utilization_pct']}%")
        check("1024^3 approaches peak (>90%)",
              big["stages"][0]["utilization_pct"] > 90,
              f"{big['stages'][0]['utilization_pct']}%")

        two = run_perf(r, [stage, {"stage": 1, "m": 64, "k": 64, "n": 64,
                                   "out_dtype": "bf16"}], energy=False)
        check("two stages -> two entries, summed total",
              len(two["stages"]) == 2 and two["total_cycles_predicted"]
              == sum(p["cycles_predicted"] for p in two["stages"]))

    print("fail-soft: bad MX_PPA_ROOT raises PerfError, nothing else")
    old = os.environ.get("MX_PPA_ROOT")
    os.environ["MX_PPA_ROOT"] = "/nonexistent-ppa"
    try:
        try:
            perf_model_path()
            check("PerfError raised", False)
        except PerfError:
            check("PerfError raised", True)
    finally:
        if old is None:
            os.environ.pop("MX_PPA_ROOT", None)
        else:
            os.environ["MX_PPA_ROOT"] = old

    print()
    if all(CHECKS):
        print(f"ALL {len(CHECKS)} CHECKS PASSED -- the recipe->perf adapter is wired correctly.")
        return 0
    print(f"{CHECKS.count(False)} of {len(CHECKS)} checks FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
