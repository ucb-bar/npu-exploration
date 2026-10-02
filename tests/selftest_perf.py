"""Self-test of the recipe -> performance-model adapter (models/perf/perf.py).

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

from models.perf.perf import PerfError, perf_args, perf_model_path, run_perf, spad_kb  # noqa: E402
from config.recipe import load_hardware as load, load_run, parse_hardware  # noqa: E402

#: G for a LUT format: its run recipe's lut.group (config/run/<fmt>.json), never a constant here.
def G(fmt: str) -> int:
    return load_run(fmt).lut.group

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append(ok)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


def main() -> int:
    r = load("baseline")

    print("mapping: baseline + 64x64x64 -> the model's CLI")
    args = perf_args(r, "fp8_e4m3", 64, 64, 64, "f8E4M3FN")
    got = dict(zip(args[::2], args[1::2]))
    check("--M/--N/--K", (got["--M"], got["--N"], got["--K"]) == ("64", "64", "64"))
    check("--rows/--cols = dim", (got["--rows"], got["--cols"]) == ("16", "16"))
    check("--act/--wei = the format's token", (got["--act"], got["--wei"]) == ("fp8", "fp8"))
    check("a direct format runs without the LUT", "--lut" not in args)
    for fmt, tok in (("fp8_e5m2", "fp8e5m2"), ("fp6_e2m3", "fp6e2m3"), ("fp6_e3m2", "fp6"), ("fp8_e4m3_quad", "fp8")):
        a6 = perf_args(r, fmt, 128, 64, 256, "bf16", lut_group=G(fmt))
        g6 = dict(zip(a6[::2], a6[1::2]))
        i = a6.index("--lut-a")
        check(f"{fmt}: --act {tok}, --lut, one LUT per 2 rows of A / cols of W / rows of C, only needed tables",
              g6["--act"] == tok and "--lut" in a6 and a6[i:i + 9] == ["--lut-a", "2", "256", "--lut-w", "256", "2",
                                                                       "--lut-c", "2", "64"] and "--lut-full-set" not in a6,
              " ".join(a6[i:i + 9]))
    g2 = perf_args(r, "fp6_e3m2", 128, 64, 256, "bf16", lut_group=2)
    check("lut_group 2 -> LUTs of 4 rows", g2[g2.index("--lut-a") + 1] == "4")
    try:
        perf_args(r, "fp6_e3m2", 128, 64, 256, "bf16")
        check("a LUT format without lut_group is refused (no default G)", False, "accepted")
    except PerfError as exc:
        check("a LUT format without lut_group is refused (no default G)", "lut_group" in str(exc))
    en = perf_args(load("wide_acc"), "fp8_e4m3", 64, 64, 64, "bf16", energy=True)
    check("--energy carries the recipe's ladder", en[en.index("--acc-rows") + 1] == "16x8,8")
    try:
        perf_args(r, "fp8_e4m3", 64, 64, 64, "bf16", tiles=(64, 64, 64))
        check("production-GEMM options refused as measured", False)
    except PerfError:
        check("production-GEMM options refused as measured", True)
    ideal = perf_args(r, "fp8_e4m3", 64, 64, 64, "bf16", as_measured=False, tiles=(32, 32, 64), dma_bw=16, spad_kb=spad_kb(r))
    check("--ideal takes tiles, dma-bw and the recipe's scratchpad (4 x 4096 x 16 B = 256 KB)",
          " ".join(ideal).endswith("--tm 32 --tn 32 --tk 64 --dma-bw 16 --spad-kb 256.0"), " ".join(ideal[-10:]))
    check("--out-fmt f8E4M3FN -> fp8", got["--out-fmt"] == "fp8", got["--out-fmt"])
    check("--as-measured on by default", "--as-measured" in args)
    check("--clock-ns = implementation.clock_ns", got["--clock-ns"] == "2.0", got["--clock-ns"])
    fp4 = dict(zip(*[iter(perf_args(r, "fp4_e2m1", 64, 64, 64, "bf16"))] * 2))
    check("an fp4 run gives --act/--wei fp4", (fp4["--act"], fp4["--wei"]) == ("fp4", "fp4"))
    import copy
    raw = copy.deepcopy(r.raw)
    raw["implementation"]["clock_ns"] = 1.25
    check("a recipe's own clock reaches the model",
          dict(zip(*[iter(perf_args(parse_hardware(raw), "fp8_e4m3", 64, 64, 64, "bf16"))] * 2))["--clock-ns"] == "1.25")
    args_bf = perf_args(r, "fp8_e4m3", 64, 64, 64, "bf16")
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
        res = run_perf(r, "fp8_e4m3", [stage])
        s = res["stages"][0]
        check("total == sum of phases",
              s["cycles_predicted"] == sum(s["phases"].values()),
              f"{s['cycles_predicted']} vs {sum(s['phases'].values())}")
        check("utilization in (0, 100]", 0 < s["utilization_pct"] <= 100,
              str(s["utilization_pct"]))
        check("compute phase nonzero", s["phases"]["compute"] > 0)
        check("fp8 without LUT loads nothing", s["phases"]["lut"] == 0)
        import numpy as np
        from compiler.operands import quantize_operand
        M, K, N = 64, 128, 32
        lut = run_perf(r, "fp6_e3m2", [{"stage": 0, "m": M, "k": K, "n": N, "out_dtype": "bf16"}], energy=False,
                       lut_group=G("fp6_e3m2"))
        rng = np.random.default_rng(0)
        from tests.fixtures import luts
        st6 = luts.settings("fp6_e3m2")
        _, _, a_books = quantize_operand(rng.standard_normal((M, K)).astype(np.float32), side="a", dtype="fp6_e3m2", lut=st6)
        _, _, b_books = quantize_operand(rng.standard_normal((K, N)).astype(np.float32), side="b", dtype="fp6_e3m2", lut=st6)
        want = a_books.shape[0] + b_books.shape[0]
        got_t = lut["stages"][0].get("lut_tables")
        check(f"fp6 {M}x{K}x{N}: the model's LUT tables == the compiler's LUT groups ({want})", got_t == want,
              f"model {got_t}")
        check("fp6 with the LUT spends cycles loading LUTs", lut["stages"][0]["phases"]["lut"] > 0)
        check("energy parsed (uJ > 0)",
              "energy" in res and res["energy"]["uj_kernel"] > 0,
              str(res.get("energy")))
        check("kernel totals = stage sums",
              res["total_cycles_predicted"] == s["cycles_predicted"])

        big = run_perf(r, "fp8_e4m3", [{"stage": 0, "m": 1024, "k": 1024, "n": 1024,
                            "out_dtype": "bf16"}], energy=False)
        check("1024^3 utilization > 64^3 (amortized setup)",
              big["stages"][0]["utilization_pct"] > s["utilization_pct"],
              f"{big['stages'][0]['utilization_pct']}% vs {s['utilization_pct']}%")
        check("1024^3 approaches peak (>90%)",
              big["stages"][0]["utilization_pct"] > 90,
              f"{big['stages'][0]['utilization_pct']}%")

        two = run_perf(r, "fp8_e4m3", [stage, {"stage": 1, "m": 64, "k": 64, "n": 64,
                                   "out_dtype": "bf16"}], energy=False)
        check("two stages -> two entries, summed total",
              len(two["stages"]) == 2 and two["total_cycles_predicted"]
              == sum(p["cycles_predicted"] for p in two["stages"]))

    if live:
        print("B: memory energy beside the timeline (never in it)")
        from models.ppa.ppa import ppa_root
        stage = {"stage": 0, "m": 128, "k": 128, "n": 128, "out_dtype": "bf16"}
        real = run_perf(r, "fp6_e3m2", [stage], lut_group=G("fp6_e3m2"))
        check("real workspace: memory unavailable, and says why",
              real["memory"]["available"] is False and "SRAM compiler tables" in real["memory"]["why"])
        check("the stage states its PE mode and ops/PE/cycle (fp6 LUT: mode 4, 4)",
              (real["stages"][0].get("pe_mode"), real["stages"][0].get("ops_per_pe_cycle")) == (4, 4))
        import shutil
        sys.path.insert(0, str(REPO / "tests"))
        from ppa_memfixture import synthetic_workspace
        fx = synthetic_workspace(ppa_root())
        old_root = os.environ.get("MX_PPA_ROOT")
        os.environ["MX_PPA_ROOT"] = str(fx)
        try:
            syn = run_perf(r, "fp6_e3m2", [stage], lut_group=G("fp6_e3m2"))
        finally:
            if old_root is None:
                os.environ.pop("MX_PPA_ROOT", None)
            else:
                os.environ["MX_PPA_ROOT"] = old_root
            shutil.rmtree(fx.parent, ignore_errors=True)
        sm = syn["stages"][0].get("memory", {})
        check("synthetic SRAM table: --mem parsed per stage (plumbing only; numbers invented)",
              syn["memory"].get("available") is True and {"smem", "acc", "scale"} <= set(sm.get("memories", {})),
              str(syn["memory"])[:120])
        if sm:
            check("memory energy parsed (uJ macros > 0, reads > 0)",
                  sm["uj_macros"] > 0 and sm["memories"]["smem"]["reads"] > 0)
            check("kernel memory total = stage sum", syn["memory"]["uj_macros"] == round(sm["uj_macros"], 3))
        check("cycles and energy do not move with --mem",
              (syn["total_cycles_predicted"], syn.get("energy")) == (real["total_cycles_predicted"], real.get("energy")))

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
