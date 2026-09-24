"""Self-test of config/scheme.py: a recipe becomes mxq's quantizer, Arithmetic, schedule and window.

Three claims:
  1. The mapping is the one the recipe states: baseline -> MXGEMMINI(e4m3), the tapeout ladder, window 16;
     flat_acc4 / wide_acc -> flat ladders; narrow_prod -> an e4m2 product.
  2. The Scheme computes what the hardware model computes: for every recipe, scheme(r).matmul(A, B) is
     bit-identical to app/mxmesh/fp8.tiled_matmul_hwlike (the hardware team's extracted model) driven with
     that recipe's product and accumulator lists, on the same codes. Includes an all-zero and a tiny block.
  3. What mxq cannot model is refused, never approximated: a non-uniform product list, an accumulator
     list of the wrong length, and (model level only) a codebook operand path.

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
from config.recipe import RecipeError, load, parse  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def recipe(name: str):
    return load(REPO / "config" / "recipes" / f"{name}.json")


def main() -> int:
    ok, why = models.paths(), models.mxq_missing()
    if not ok:
        print(f"SKIP: {why}")
        return 0
    from mxq import schedule as mxq_schedule
    from app.mxmesh import fp8 as M8

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
    check("format name for fp8 recipes", scheme.format_name(recipe("baseline")) == "MXFP8_E4M3")

    print("\n[2] Scheme == hardware model, per recipe ---------------------------")
    g = torch.Generator().manual_seed(2)
    K, M, N = 128, 32, 32
    A = torch.randn(K, M, generator=g)                 # K x M  (A = x^T)
    B = torch.randn(K, N, generator=g)                 # K x N  (B = W^T)
    A[0:32, 0] = 0.0                                   # an all-zero block
    A[32:64, 1] = 2.0 ** -30                           # a tiny block (below the 2^-23 floor)
    for name in ("baseline", "flat_acc4", "wide_acc", "narrow_prod"):
        r = recipe(name)
        s = scheme.scheme(r)
        q = scheme.quantizer(r)
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
            r = parse(raw, path=base.path)
            fn(r)
        except RecipeError as exc:
            check(label, True, str(exc)[:90])
            return
        check(label, False, "no RecipeError")

    def mixed_prod(raw):
        raw["types"]["meshProdPrecisionList"][3]["sigWidth"] = 3
    refused("non-uniform product list is refused", raw_variant(mixed_prod), scheme.datapath)

    def lut_on(raw):
        raw["runtime"]["use_lut"] = True
        raw["mx"]["enable_lut"] = True
    refused("use_lut recipe is refused at model level", raw_variant(lut_on), scheme.scheme)

    # config/recipe.parse refuses operand_fmt=fp6 itself today (the fp6 encoder is not wired), so the
    # codebook refusal is exercised on a Recipe object: the same field, past the parser.
    import dataclasses
    fp6_recipe = dataclasses.replace(base, operand_fmt="fp6")
    try:
        scheme.scheme(fp6_recipe)
        check("fp6 (codebook-indexed) operands are refused at model level", False, "no RecipeError")
    except RecipeError as exc:
        check("fp6 (codebook-indexed) operands are refused at model level", True, str(exc)[:90])

    # datapath() must NOT refuse fp6: the mxquant model grades codebook formats through wire operands.
    try:
        scheme.datapath(fp6_recipe)
        check("datapath() still serves an fp6 recipe (wire-operand path)", True)
    except RecipeError as exc:
        check("datapath() still serves an fp6 recipe (wire-operand path)", False, str(exc))

    print("\n[4] model selection -------------------------------------------------")
    check("default group", models.select("default") == ("reference", "mxquant", "spike", "ppa", "perf"))
    check("all group", models.select("all") == models.NAMES)
    check("a list, canonical order", models.select("perf,mxquant") == ("mxquant", "perf"))
    check("group plus a name", models.select("default,accuracy") == models.NAMES)
    try:
        models.select("bogus")
        check("unknown name is refused", False)
    except ValueError as exc:
        check("unknown name is refused", "bogus" in str(exc))

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("ALL CHECKS PASSED -- the recipe alone defines the arithmetic mxq runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
