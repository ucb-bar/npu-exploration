"""models/mxquant (the reference on mxq) must reproduce the legacy tier, follow the recipe, and degrade honestly.

  1. Legacy equivalence. For every registry kernel shape (linear, mlp2, mlp3, attention) and every operand
     format, with the edges the pipeline's own lowering builds (fused chain with codebooks, graph),
     models.mxquant.run(...).y and every intermediate equal grade/mxquant_ref.simulate(rtl_exact=True)
     element for element. Skipped, and said so, when the MXQuant clone is absent.
  2. Recipe-aware. For each of the four recipes, the model equals the extracted hardware model
     (rtl_exact/mxmesh/fp8.tiled_matmul_hwlike) driven with that recipe's product and accumulator lists on the
     same wire operands. The legacy tier could not do this: it hardcoded the tapeout ladder.
  3. Degrade. An edge the requantizer cannot reproduce raises Unavailable, so the pipeline grades on the
     fp32 tier instead of inventing a reference.
  4. As-shipped self-consistency: the informational number is mxq's MXQuant mode.
  5. Recorded bits. Every kernel x format x recipe (120 records, y and every stage) equals
     tests/oracle/mxquant_bits.json, captured from the model BEFORE its operand path moved from
     compiler/operands's wire round trip to mxq's block quantizer (2026-09-28). This is what proves the
     direct formats run on mxq alone without a single bit moving.
  6. The requant rule. For an fp8_e4m3 chain, mxq's quantizer on the bf16 accumulator IS the device
     requantizer transcribed in compiler/operands.py: 0 differing on 10^6 values incl. ties, subnormals,
     zero and sub-2^-23 blocks. For fp4_e2m1 it is NOT (two-step rounding), which is why that chain
     edge stays on the device model -- measured here so the exception is a number, not a belief.

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
    from config.recipe import load_hardware
    from kernels.registry import build
    from models import mxquant
    from rtl_exact.mxmesh import fp8 as M8
    from compiler.operands import quantize_operand, wire_to_px
    from config import scheme

    recipes = {n: load_hardware(n) for n in
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

    print("\n[5] recorded bits: every kernel x format x recipe == tests/oracle/mxquant_bits.json ----")
    import hashlib, json
    from compiler.lower import lower
    rec = json.loads((REPO / "tests" / "oracle" / "mxquant_bits.json").read_text())

    def h(a):
        return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()).hexdigest()[:24]
    n_ok, n_all, bad = 0, 0, []
    for key, want in rec["records"].items():
        kname, dtype, rname = key.split("/")
        sp = build(kname)
        out = mxquant.run(sp, recipes[rname], dtype=dtype, edges=lower(sp, dtype, allow_lossy_chain=True).edges,
                          shipped=False)
        n_all += 1
        if h(out["y"]) == want["y"] and all(h(out["stages"][s]) == hh for s, hh in want["stages"].items()):
            n_ok += 1
        else:
            bad.append(key)
    check(f"{n_ok}/{n_all} records identical (y and every stage)", not bad, ", ".join(bad[:6]))

    print("\n[6] the requant rule: mxq quantize(bf16(C)) vs the device requantizer (compiler/operands) ----")
    from compiler.operands import requantize_chained
    from models.mxquant import kernel as K
    rng = np.random.default_rng(0)

    def cases():
        M, N = 64, 64
        yield "normal", rng.standard_normal((M, N)).astype(np.float32) * 3
        yield "wide range", (rng.standard_normal((M, N)) * np.exp2(rng.integers(-20, 20, (M, N)))).astype(np.float32)
        z = rng.standard_normal((M, N)).astype(np.float32); z[:, :32] = 0; yield "zero blocks", z
        t = rng.standard_normal((M, N)).astype(np.float32); t[:, :32] *= 2.0 ** -30; yield "sub-2^-23 blocks", t
        sn = np.ones((M, N), np.float32); sn[:, 1:] = 2.0 ** -12; yield "subnormal after scaling", sn
        ties = np.full((M, N), 1.0625, np.float32); ties[:, 0] = 1.0; ties[:, 1] = -1.0625; yield "exact e4m3 ties", ties
        yield "1e6 random", (rng.standard_normal((1024, 1024)) * np.exp2(rng.integers(-10, 10, (1024, 1024)))).astype(np.float32)

    def diff(dtype, C):
        P0, X0 = requantize_chained(C, dtype=dtype, books=None)
        P1, X1 = K._requant(C, dtype, None)
        return int((P0.numpy() != P1.numpy()).sum()), int((X0.numpy() != X1.numpy()).sum()), P0.numel()
    for name, C in cases():
        dp, dx, n = diff("fp8_e4m3", C)
        check(f"fp8_e4m3 {name}: mxq == device requantizer", dp == 0 and dx == 0, f"P {dp}/{n} X {dx}")
    # fp4: the device rounds bf16 -> E3M1 -> E2M1; mxq's via=(3, 1) is that. Its scale floor is E8M0's 2^-126,
    # so the sub-2^-23 case is a real check here and the zero-block case is excluded (the device's scale
    # underflows to 0 there, mxq keeps 2^-126; the codes are 0 either way).
    from rtl_exact.mxmesh import fp4 as M4
    for name, C in cases():
        if name == "zero blocks":
            continue
        C = torch.from_numpy(C).to(torch.bfloat16).float().numpy()
        Pd, Xd = M4.matrix_mx_requantize(torch.from_numpy(C.copy()), "fp4:e2m1")
        P1, X1 = K._requant(C, "fp4_e2m1", None)
        dp, dx = int((Pd.t() != P1).sum()), int((Xd.t() != X1).sum())
        check(f"fp4_e2m1 {name}: mxq via E3M1 == device requantizer", dp == 0 and dx == 0, f"P {dp}/{P1.numel()} X {dx}")
    one = torch.from_numpy(rng.standard_normal((64, 64)).astype(np.float32) * 3).to(torch.bfloat16).float().numpy()
    Pd, _ = M4.matrix_mx_requantize(torch.from_numpy(one.copy()), "fp4:e2m1")
    P1, _ = K._quantize(one, "MXFP4", axis=1)
    print(f"  info  fp4_e2m1 with a single rounding would differ on {int((Pd != P1).sum())}/{P1.numel()}: the via step is load-bearing")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("ALL CHECKS PASSED -- models.mxquant is the legacy reference, made recipe-aware.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
