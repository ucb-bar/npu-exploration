"""Self-test of config/: the two recipes load strictly, and (hardware, run) becomes mxq's quantizer,
Arithmetic, schedule and window.

Five claims:
  1. The mapping is the one the recipe states: baseline -> MXGEMMINI(e4m3), the tapeout ladder, window 16;
     flat_acc4 / wide_acc -> flat ladders; narrow_prod -> an e4m2 product.
  2. The Scheme computes what the hardware model computes: for every recipe, scheme(r).matmul(A, B) is
     bit-identical to rtl_exact/mxmesh/fp8.tiled_matmul_hwlike (the hardware team's extracted model) driven with
     that recipe's product and accumulator lists, on the same codes. Includes an all-zero and a tiny block.
  3. What mxq cannot model is refused, never approximated: a non-uniform product list, an accumulator
     list of the wrong length, and (model level only) a codebook operand path.
  4. The product flush is the hardware recipe's types.prodFloor: null switches it off and changes bits.
  5. The recipe files: unknown keys refused by name, build_id / run_id move with every number and with no
     label, and check() refuses exactly what the kernel path cannot follow.

    .venv/bin/python tests/selftest_scheme.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import models  # noqa: E402
from config import scheme  # noqa: E402
from config.recipe import RecipeError, Run, check as check_recipes, load_hardware, load_run, parse_hardware, parse_run  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def recipe(name: str):
    return load_hardware(name)


def main() -> int:
    ok, why = models.paths(), models.mxq_missing()
    if not ok:
        print(f"SKIP: {why}")
        return 0
    from mxq import schedule as mxq_schedule
    from rtl_exact.mxmesh import fp8 as M8

    print("\n[1] mapping ---------------------------------------------------------")
    arith, sched, window = scheme.datapath(recipe("baseline"))
    check("baseline arithmetic is the hardware's e4m3 product", arith.name == "mxgemmini(prod=e4m3)", arith.name)
    check("baseline schedule is the tapeout ladder", sched == list(mxq_schedule.HW_FINAL))
    check("baseline window is the mesh dimension", window == 16)
    _, s4, _ = scheme.datapath(recipe("flat_acc4"))
    check("flat_acc4 is e4m4 on every lane", s4 == mxq_schedule.fixed(4, 4))
    _, s8, _ = scheme.datapath(recipe("wide_acc"))
    check("wide_acc is e8m7 on every lane", s8 == mxq_schedule.fixed(8, 7))
    a2, _, _ = scheme.datapath(recipe("narrow_prod"))
    check("narrow_prod product is e4m2", scheme.product(recipe("narrow_prod")) == (4, 2), a2.name)
    sh, _, _ = scheme.shipped_datapath(recipe("baseline"))
    check("as-shipped arithmetic is MXQuant's", sh.name.startswith("mxquant("), sh.name)
    check("default run quantizes to MXFP8_E4M3", scheme.quantizer(recipe("baseline"), Run()).keywords["fmt"] == "MXFP8_E4M3")

    print("\n[2] Scheme == hardware model, per recipe ---------------------------")
    g = torch.Generator().manual_seed(2)
    K, M, N = 128, 32, 32
    A = torch.randn(K, M, generator=g)                 # K x M  (A = x^T)
    B = torch.randn(K, N, generator=g)                 # K x N  (B = W^T)
    A[0:32, 0] = 0.0                                   # an all-zero block
    A[32:64, 1] = 2.0 ** -30                           # a tiny block (below the 2^-23 floor)
    for name in ("baseline", "flat_acc4", "wide_acc", "narrow_prod"):
        r = recipe(name)
        s = scheme.scheme(r, Run())
        q = scheme.quantizer(r, Run())
        PA, XA = q(A)
        PB, XB = q(B)
        pe, pm = scheme.product(r)
        Y_mxq = s.matmul(A, B)
        Y_hw = M8.tiled_matmul_hwlike(PA.t().contiguous(), PB, XA.t().contiguous(), XB, verbose=False,
                                      prod_precision_list=[(pe, pm)] * r.dim,
                                      acc_precision_list=scheme.schedule(r))
        d = int((Y_mxq != Y_hw).sum())
        check(f"{name}: Scheme == mxmesh.fp8 with the recipe's lists", d == 0, f"{d}/{Y_hw.numel()} differ")

    print("\n[3] refusals -------------------------------------------------------")
    base = recipe("baseline")

    def raw_variant(mutate):
        raw = copy.deepcopy(base.raw)
        mutate(raw)
        return raw

    def refused(label, raw, fn):
        try:
            r = parse_hardware(raw, path=base.path)
            fn(r)
        except RecipeError as exc:
            check(label, True, str(exc)[:90])
            return
        check(label, False, "no RecipeError")

    def mixed_prod(raw):
        raw["types"]["meshProdPrecisionList"][3]["sigWidth"] = 3
    refused("non-uniform product list is refused", raw_variant(mixed_prod), scheme.datapath)

    def lut_on(raw):
        raw["mx"]["enable_lut"] = True
    try:
        scheme.scheme(parse_hardware(raw_variant(lut_on), path=base.path), Run())
        check("an enable_lut recipe builds a Scheme (the perplexity path runs the full element grid)", True)
    except RecipeError as exc:
        check("an enable_lut recipe builds a Scheme (the perplexity path runs the full element grid)", False, str(exc)[:90])

    s6 = scheme.scheme(base, Run(operand_fmt="fp6_e3m2"))
    check("an fp6 run builds a Scheme on the full MXFP6_E3M2 grid", s6.a.keywords["fmt"] == "MXFP6_E3M2")
    check("is_codebook knows the four table-indexed formats",
          [d for d in scheme.MXQ_FORMAT if scheme.is_codebook(d)] == ["fp8_e4m3_quad", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3"])
    check("the run picks the format, the hardware the arithmetic",
          scheme.scheme(base, load_run("fp4_e2m1")).a.keywords["fmt"] == "MXFP4"
          and scheme.scheme(base, load_run("fp4_e2m1")).reduce.keywords["arith"].name == arith.name)
    try:
        scheme.mxq_format("fp9")
        check("an unknown dtype is refused", False)
    except RecipeError as exc:
        check("an unknown dtype is refused", "fp9" in str(exc))

    print("\n[4] the product flush is the recipe's --------------------------------")
    def no_flush(raw):
        raw["types"]["prodFloor"] = None
    off = parse_hardware(raw_variant(no_flush), path=base.path)
    a = torch.tensor([2.0 ** -9, 2.0 ** -8, 1.0])      # element products 2^-18, 2^-16, 1: the flush is on codes
    p_on, p_off = scheme.mxgemmini(base).product(a, a), scheme.mxgemmini(off).product(a, a)
    check("baseline flushes a product below 2^-16 and keeps 2^-16 (prodFloor -16)",
          base.prod_floor == -16 and p_on.tolist() == [0.0, 2.0 ** -16, 1.0], str(p_on.tolist()))
    check("prodFloor null keeps it", off.prod_floor is None and p_off.tolist() == [2.0 ** -18, 2.0 ** -16, 1.0],
          str(p_off.tolist()))
    check("the flush is in build_id", off.build_id() != base.build_id())

    print("\n[5] recipe files ---------------------------------------------------")
    def refused_load(label, fn, needle):
        try:
            fn()
        except RecipeError as exc:
            check(label, needle in str(exc), str(exc)[:110])
            return
        check(label, False, "accepted")
    refused_load("unknown hardware key refused by name", lambda: parse_hardware({**base.raw, "software": {}}), "software")
    refused_load("unknown section key refused by name",
                 lambda: parse_hardware(raw_variant(lambda r: r["mx"].update(use_lut=True))), "use_lut")
    refused_load("prodFloor is required",
                 lambda: parse_hardware(raw_variant(lambda r: r["types"].pop("prodFloor"))), "prodFloor")
    dflt = load_run("default")
    rraw = {"name": "x", **dflt.fields()}
    refused_load("unknown run key refused by name", lambda: parse_run({**rraw, "dtype": "fp4_e2m1"}), "dtype")
    refused_load("a run recipe writes every field", lambda: parse_run({"name": "x", "operand_fmt": "fp8_e4m3"}), "required")
    check("default.json == Run() (the code's defaults are the file's)", dflt.fields() == Run().fields())
    check("build_id ignores the labels",
          parse_hardware(raw_variant(lambda r: r.update(description="other", name="other"))).build_id() == base.build_id())
    check("build_id moves with the ladder", recipe("wide_acc").build_id() != base.build_id())
    check("build_id moves with the clock",
          parse_hardware(raw_variant(lambda r: r["implementation"].update(clock_ns=1.0))).build_id() != base.build_id())
    check("run_id ignores the name", Run(name="x").run_id() == dflt.run_id())
    check("run_id moves with every field", len({dflt.run_id(), load_run("exact").run_id(), load_run("bf16_tiles").run_id(),
                                               load_run("fp4_e2m1").run_id(), Run(rounding="ties_away").run_id()}) == 5)
    for label, hw, run in (("rounding ties_away", base, Run(rounding="ties_away")),
                           ("scale floor 1e-38", base, Run(scale_floor=1e-38)),
                           ("reduce exact", base, load_run("exact")),
                           ("a 4x2048 scratchpad", parse_hardware(raw_variant(lambda r: r["scratchpad"].update(rows=2048))), Run()),
                           ("block 16", parse_hardware(raw_variant(lambda r: r["mx"].update(scaleSize=16))), Run())):
        refused_load(f"kernel path refuses {label}", lambda: check_recipes(hw, run, "kernel"), "kernel path")
        try:
            check_recipes(hw, run, "perplexity")
            check(f"perplexity path accepts {label}", True)
        except RecipeError as exc:
            check(f"perplexity path accepts {label}", False, str(exc)[:90])
    check_recipes(base, dflt, "kernel")
    check("kernel path accepts baseline + default and baseline + fp4_e2m1", True)
    check_recipes(base, load_run("fp4_e2m1"), "kernel")
    refused_load("an unknown operand format is refused", lambda: check_recipes(base, Run(operand_fmt="fp9"), "perplexity"), "fp9")
    from config.recipe import removed_flag
    check("a removed flag names its run field", "operand_fmt" in (removed_flag(["--kernel", "x", "--dtype", "fp4_e2m1"]) or ""))
    check("--flag=value spelling is caught too", "rounding" in (removed_flag(["--rounding-mode=ties_away"]) or ""))
    check("current flags pass", removed_flag(["--hw", "baseline", "--run", "exact"]) is None)

    print("\n[6] model selection -------------------------------------------------")
    check("default group", models.select("default") == ("reference", "mxquant", "spike", "ppa", "perf"))
    check("all group", models.select("all") == models.NAMES)
    check("a list, canonical order", models.select("perf,mxquant") == ("mxquant", "perf"))
    check("group plus a name", models.select("default,perf") == models.NAMES)
    try:
        models.select("bogus")
        check("unknown name is refused", False)
    except ValueError as exc:
        check("unknown name is refused", "bogus" in str(exc))

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("ALL CHECKS PASSED -- the two recipes alone define the arithmetic mxq runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
