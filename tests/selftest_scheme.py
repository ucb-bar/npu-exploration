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

    def no_lut(raw):
        raw["mx"]["lut"] = None
    try:
        scheme.scheme(parse_hardware(raw_variant(no_lut), path=base.path), Run())
        check("a build without a LUT unit (mx.lut null) builds a Scheme for a direct format", True)
    except RecipeError as exc:
        check("a build without a LUT unit (mx.lut null) builds a Scheme for a direct format", False, str(exc)[:90])

    s6 = scheme.scheme(base, load_run("fp6_e3m2"))
    check("an fp6 run builds a Scheme on the full MXFP6_E3M2 grid", s6.a.keywords["fmt"] == "MXFP6_E3M2")
    from config.recipe import check as recipe_check
    s6d = scheme.scheme(base, load_run("fp6_e3m2_direct"))
    check("an fp6 run without a lut block quantizes straight to the MXFP6_E3M2 grid (LUT off)",
          s6d.a.keywords["fmt"] == "MXFP6_E3M2" and s6d.a.func is scheme.scheme(base, load_run("fp4_e2m1")).a.func
          and s6d.a.func is not s6.a.func and scheme.lut_record(load_run("fp6_e3m2_direct")) is None
          and not scheme.uses_lut(load_run("fp6_e3m2_direct")) and scheme.uses_lut(load_run("fp6_e3m2")))
    try:
        recipe_check(base, load_run("fp6_e3m2_direct"), "perplexity")
        check("LUT off fp6 passes the perplexity path's check", True)
    except RecipeError as exc:
        check("LUT off fp6 passes the perplexity path's check", False, str(exc)[:90])
    try:
        recipe_check(base, load_run("fp6_e3m2_direct"), "kernel")
        check("LUT off fp6 is refused on the kernel path (the chip's requantizer needs the LUT)", False, "accepted")
    except RecipeError as exc:
        check("LUT off fp6 is refused on the kernel path (the chip's requantizer needs the LUT)", "lut block" in str(exc))
    from dataclasses import replace
    ocp_run = replace(load_run("fp4_e2m1"), scale="ocp")
    s_ocp = scheme.scheme(base, ocp_run)
    check("scale ocp quantizes through mxq.block.ocp (block max at the format max), rne -> even",
          s_ocp.a.func.__module__ == "mxq.block.ocp" and s_ocp.a.keywords["rounding_mode"] == "even"
          and s_ocp.a.keywords["fmt"] == "MXFP4")
    check("scale ocp changes run_id; scale mxgemmini (the default) keeps it",
          ocp_run.run_id() != load_run("fp4_e2m1").run_id()
          and replace(load_run("fp4_e2m1"), scale="mxgemmini").run_id() == load_run("fp4_e2m1").run_id())
    try:
        parse_run({**{k: v for k, v in vars(load_run("fp4_e2m1")).items() if k in ("name", "operand_fmt", "rounding", "scale_floor", "reduce", "allow_lossy_chain", "fp32_tol")}, "scale": "imx"})
        check("an unknown scale is refused", False, "accepted")
    except RecipeError as exc:
        check("an unknown scale is refused", "scale" in str(exc))
    try:
        recipe_check(base, ocp_run, "kernel")
        check("scale ocp is refused on the kernel path", False, "accepted")
    except RecipeError as exc:
        check("scale ocp is refused on the kernel path", "scale ocp" in str(exc))
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

    print("\n[5b] the LUT, in both recipes ------------------------------------------")
    from config.recipe import LUT_SERVES, Fit, Lut, emitter_params, lut_settings
    from compiler.codebook import Settings
    check("recorded run_ids unchanged (a recipe without a lut block hashes as before)",
          {r: load_run(r).run_id() for r in ("default", "exact", "bf16_tiles", "fp4_e2m1")}
          == {"default": "611f047101d540f2", "exact": "f4c02e1652fe39ec",
              "bf16_tiles": "e28620394769c9a0", "fp4_e2m1": "19527a31314dcfa5"})
    check("no lut block: Run.lut is None and fields() has no lut key", dflt.lut is None and "lut" not in dflt.fields())
    lrun = load_run("fp6_e3m2")
    check("config/run/fp6_e3m2.json writes today's compiler rule",
          lrun.lut == Lut(group=1, weights="data", activations="data", outputs="estimate", pick="host",
                          fit=Fit(method="kmeans", init="quantile", max_iters=50)), lrun.describe())
    lraw = {"name": "x", **lrun.fields()}
    good = lraw["lut"]
    check("fields() round-trips through parse_run", parse_run(lraw) == parse_run({**lraw}) and parse_run(lraw).lut == lrun.lut)
    check("run_id moves with the lut block",
          lrun.run_id() != parse_run({**lraw, "lut": {**good, "fit": {**good["fit"], "max_iters": 49}}}).run_id())
    refused_load("unknown lut key refused by name", lambda: parse_run({**lraw, "lut": {**good, "size": 16}}), "size")
    refused_load("every lut key is written", lambda: parse_run({**lraw, "lut": {"group": 1}}), "required")
    refused_load("every lut.fit key is written",
                 lambda: parse_run({**lraw, "lut": {**good, "fit": {"method": "kmeans"}}}), "required")
    for key, val in (("weights", "tables.json"), ("activations", "top16"), ("outputs", "calibrated"), ("pick", "finder")):
        refused_load(f"lut.{key} {val!r} is not implemented yet, refused by name",
                     lambda: parse_run({**lraw, "lut": {**good, key: val}}), "implemented today")
    refused_load("lut.fit.method other than kmeans refused",
                 lambda: parse_run({**lraw, "lut": {**good, "fit": {**good["fit"], "method": "lloyd"}}}), "implemented today")
    refused_load("lut.group is a non-negative integer", lambda: parse_run({**lraw, "lut": {**good, "group": -1}}), "group")
    refused_load("a lut block on a direct format is refused",
                 lambda: check_recipes(base, parse_run({**rraw, "lut": good}), "perplexity"), "not a LUT format")
    nolut = parse_run({k: v for k, v in lraw.items() if k != "lut"})
    refused_load("a LUT format without a lut block is refused on the kernel path (the requantizer needs the LUT)",
                 lambda: check_recipes(base, nolut, "kernel"), "lut block")
    try:
        check_recipes(base, nolut, "perplexity")
        check("a LUT format without a lut block runs LUT off on the perplexity path", True)
    except RecipeError as exc:
        check("a LUT format without a lut block runs LUT off on the perplexity path", False, str(exc)[:90])

    print("\n[5c] the hardware recipe's mx.lut -----------------------------------")
    check("baseline's LUT unit is the stock one (LutFP6E3M2, 6-bit, 64 per table, 16-bit G)",
          (base.lut.projection, base.lut.entry_bits, base.lut.index_bits, base.lut.tables, base.lut.group_bits)
          == ("LutFP6E3M2", 6, 4, (64, 64, 64), 16), str(base.lut))
    refused_load("mx.lut is required", lambda: parse_hardware(raw_variant(lambda r: r["mx"].pop("lut"))), "mx.lut")
    refused_load("every mx.lut key is written",
                 lambda: parse_hardware(raw_variant(lambda r: r["mx"]["lut"].pop("projFormat"))), "required")
    refused_load("an unknown projFormat is refused",
                 lambda: parse_hardware(raw_variant(lambda r: r["mx"]["lut"].update(projFormat="LutFP4"))), "projFormat")
    refused_load("an FP8 projection needs 8-bit entries",
                 lambda: parse_hardware(raw_variant(lambda r: r["mx"]["lut"].update(projFormat="LutFP8E4M3"))), "rdataWidth 8")
    refused_load("numBits is 16 entries x rdataWidth",
                 lambda: parse_hardware(raw_variant(lambda r: r["mx"]["lut"].update(numBits=[128, 128, 128]))), "numBits")
    unit = parse_hardware(raw_variant(no_lut))
    for path in ("kernel", "perplexity"):
        refused_load(f"a LUT format on a build without a LUT unit is refused ({path})",
                     lambda: check_recipes(unit, lrun, path), "without a LUT unit")
        refused_load(f"a LUT format the projection does not serve is refused ({path})",
                     lambda: check_recipes(base, load_run("fp8_e4m3_quad"), path), "serves")
    refused_load("raddrWidth other than the 4-bit index is refused",
                 lambda: check_recipes(parse_hardware(raw_variant(lambda r: r["mx"]["lut"].update(
                     raddrWidth=5, numBits=[192, 192, 192]))), lrun, "kernel"), "raddrWidth")
    refused_load("an asymmetric LUT build is refused",
                 lambda: check_recipes(parse_hardware(raw_variant(lambda r: r["mx"]["lut"].update(actCodeWidth=6))),
                                       lrun, "kernel"), "asymmetric")
    refused_load("G beyond the G register is refused",
                 lambda: check_recipes(parse_hardware(raw_variant(lambda r: r["mx"]["lut"].update(lutUpdateRegularityWidth=1))),
                                       parse_run({**lraw, "lut": {**good, "group": 2}}), "kernel"), "G register")
    check_recipes(base, parse_run({**lraw, "lut": {**good, "group": 2}}), "kernel")
    check("check accepts G = 2 (the emitter holds each table to its capacity)", True)
    fp8 = load_hardware("lut_fp8e4m3")
    refused_load("fp6 on an 8-bit-entry build is refused on the kernel path (wider entries not modelled)",
                 lambda: check_recipes(fp8, lrun, "kernel"), "not modelled")
    refused_load("...and on the perplexity path (mxq.lut holds the format's own entries)",
                 lambda: check_recipes(fp8, lrun, "perplexity"), "not modelled")
    for fmt in ("fp8_e4m3_quad", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3"):
        builds = [h for h in ("baseline", "lut_fp8e4m3", "lut_fp8e5m2", "lut_fp6e2m3")
                  if fmt in LUT_SERVES[load_hardware(h).lut.projection]]
        ok = []
        for h in builds:
            try:
                check_recipes(load_hardware(h), load_run(fmt), "kernel")
                ok.append(h)
            except RecipeError:
                pass
        check(f"{fmt} has a build in config/hardware/ that runs it on the kernel path", bool(ok), ", ".join(ok))
    check("lut_settings: G and the fit's passes, from the recipes",
          lut_settings(base, lrun) == Settings(group=1, max_iters=50) and lut_settings(base, dflt) is None)
    check("emitter_params: G and each table's capacity for a LUT run, the geometry alone otherwise",
          emitter_params(base, lrun) == {**base.geometry(), "lut_group": 1, "lut_tables": [64, 64, 64]}
          and emitter_params(base, dflt) == base.geometry())
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
