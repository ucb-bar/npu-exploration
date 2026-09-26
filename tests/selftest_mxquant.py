"""models/mxquant (the reference on mxq) must reproduce the legacy tier, follow the recipe, and degrade honestly.

  1. Legacy equivalence. For every registry kernel shape (linear, mlp2, mlp3, attention) and every operand
     format, with the edges the pipeline's own lowering builds (fused chain with codebooks, graph),
     models.mxquant.run(...).y and every intermediate equal grade/mxquant_ref.simulate(rtl_exact=True)
     element for element. Skipped, and said so, when the MXQuant clone is absent.
  2. Recipe-aware. For each of the four recipes, the model equals the extracted hardware model
     (app/mxmesh/fp8.tiled_matmul_hwlike) driven with that recipe's product and accumulator lists on the
     same wire operands. The legacy tier could not do this: it hardcoded the tapeout ladder.
  3. Degrade. An edge the requantizer cannot reproduce raises Unavailable, so the pipeline grades on the
     fp32 tier instead of inventing a reference.
  4. As-shipped self-consistency: the informational number is mxq's MXQuant mode.

    .venv/bin/python tests/selftest_mxquant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

FAILURES: list[str] = []
FORMATS = ("fp8_e4m3", "fp8_e4m3_quad", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1")
KERNELS = ("linear", "mlp2", "mlp3", "attention")


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def same(a, b) -> bool:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return a.shape == b.shape and bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


def edges_for(pipeline, spec, dtype):
    """Exactly what grade/pipeline.run hands the model, built by the same lowering."""
    from compiler.lower import lower
    return lower(spec, dtype, allow_lossy_chain=True).edges


def main() -> int:
    import models
    if not models.paths():
        print(f"SKIP: {models.mxq_missing()}")
        return 0
    from grade import pipeline
    pipeline._wire_paths(REPO)
    from config.recipe import load
    from kernels.registry import build
    from models import mxquant
    from app.mxmesh import fp8 as M8
    from app.mxq_golden import quantize_operand, wire_to_px
    from config import scheme

    recipes = {n: load(REPO / "config" / "recipes" / f"{n}.json") for n in
               ("baseline", "flat_acc4", "wide_acc", "narrow_prod")}
    base = recipes["baseline"]

    print("\n[1] legacy equivalence: models.mxquant == grade.mxquant_ref, every kernel x format ---")
    try:
        from grade import mxquant_ref as legacy
        ok, why = legacy.available()
    except Exception as exc:
        ok, why = False, str(exc)
    if not ok:
        print(f"  skip  legacy tier unavailable ({why}); needs the MXQuant clone (--with-mxquant)")
    else:
        n_pairs = 0
        for kname in KERNELS:
            for dtype in FORMATS:
                spec = build(kname)
                try:
                    edges = edges_for(pipeline, spec, dtype)
                except Exception as exc:
                    print(f"  skip  {kname}/{dtype}: lowering refused ({type(exc).__name__}: {str(exc)[:70]})")
                    continue
                try:
                    old = legacy.simulate(spec, rtl_exact=True, dtype=dtype, edges=edges)
                except legacy.MxQuantUnavailable as exc:
                    old = None
                try:
                    new = mxquant.run(spec, base, dtype=dtype, edges=edges, shipped=False)
                except mxquant.Unavailable as exc:
                    new = None
                if old is None or new is None:
                    check(f"{kname}/{dtype}: both refuse or both model", (old is None) == (new is None))
                    continue
                n_pairs += 1
                y_ok = same(old.y, new["y"])
                st_ok = all(same(old.stages[k], new["stages"][k]) for k in old.stages)
                check(f"{kname}/{dtype}: y and every stage identical", y_ok and st_ok,
                      "" if y_ok and st_ok else f"y {'ok' if y_ok else 'DIFF'} stages {'ok' if st_ok else 'DIFF'}")
        print(f"  ({n_pairs} kernel x format pairs compared)")

    print("\n[2] recipe-aware: model == mxmesh.fp8 with the recipe's lists, on wire operands ---")
    spec = build("linear")
    x = spec.x.numpy().astype(np.float32)
    x[:, 0:32] = 0.0                                       # an all-zero block in the A operand
    x[:, 32:64] = 2.0 ** -30                                # a tiny block under the 2^-23 floor
    spec.x = __import__("torch").from_numpy(x)
    W = spec.stages[0].weight.numpy().astype(np.float32)
    ac, asc, al = quantize_operand(x, side="a", dtype="fp8_e4m3")
    bc, bsc, bl = quantize_operand(W, side="b", dtype="fp8_e4m3")
    PA, XA = wire_to_px(ac, asc, side="a", dtype="fp8_e4m3", books=al)
    PB, XB = wire_to_px(bc, bsc, side="b", dtype="fp8_e4m3", books=bl)
    for name, r in recipes.items():
        pe, pm = scheme.product(r)
        y_hw = M8.tiled_matmul_hwlike(PA.t().contiguous(), PB, XA.t().contiguous(), XB, verbose=False,
                                      prod_precision_list=[(pe, pm)] * r.dim,
                                      acc_precision_list=scheme.schedule(r)).numpy()
        y = mxquant.run(spec, r, dtype="fp8_e4m3", edges=edges_for(pipeline, spec, "fp8_e4m3"), shipped=False)["y"]
        check(f"{name}: model == hardware model with the recipe's ladder", same(y, y_hw),
              f"{int((np.asarray(y) != y_hw).sum())}/{y_hw.size} differ")
    a_flat, _, _ = scheme.datapath(recipes["wide_acc"])
    y_base = mxquant.run(spec, base, dtype="fp8_e4m3", shipped=False)["y"]
    y_wide = mxquant.run(spec, recipes["wide_acc"], dtype="fp8_e4m3", shipped=False)["y"]
    check("baseline and wide_acc give different bits (the recipe is honoured)", not same(y_base, y_wide))

    print("\n[3] degrade: an edge the requantizer cannot model raises Unavailable ------------")
    spec48 = build("mlp2", h=48)      # N=48 is not a multiple of the 32-block; the direct e4m3 requantizer refuses it
    edges = {st.name: {"via": "requant", "books": None} for st in spec48.stages}
    try:
        mxquant.run(spec48, base, dtype="fp8_e4m3", edges=edges, shipped=False)
        check("chained fp8_e4m3 with N % 32 != 0 raises Unavailable", False, "no exception")
    except mxquant.Unavailable as exc:
        check("chained fp8_e4m3 with N % 32 != 0 raises Unavailable", True, str(exc)[:80])
    ok, why = mxquant.available()
    check("available() reports mxq and its commit", ok and why.startswith("mxq "), why)

    print("\n[4] as-shipped: mxq MXQuant mode ---------------------------------------------")
    res = mxquant.run(spec, base, dtype="fp8_e4m3", edges=edges_for(pipeline, spec, "fp8_e4m3"))
    import torch
    from mxq import block, matmul
    s_arith, sched, window = scheme.shipped_datapath(base)
    PA2, XA2 = block.mxquant.quantize(torch.from_numpy(np.ascontiguousarray(x.T)), "MXFP8_E4M3", axis=0)
    PB2, XB2 = block.mxquant.quantize(torch.from_numpy(W), "MXFP8_E4M3", axis=0)
    y_ship = matmul.systolic(PA2, XA2, PB2, XB2, s_arith, sched, window=window).numpy()
    check("shipped_y is block.mxquant + MXQUANT on the recipe's ladder", same(res["shipped_y"], y_ship))
    check("shipped differs from the hardware reference (the gap is real)", not same(res["shipped_y"], res["y"]))
    check("model record names mxq, the arithmetic and the build_id",
          res["model"]["source"] == "mxq" and res["model"]["arith"].startswith("mxgemmini")
          and res["model"]["build_id"] == base.build_id())

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("ALL CHECKS PASSED -- models.mxquant is the legacy reference, made recipe-aware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
