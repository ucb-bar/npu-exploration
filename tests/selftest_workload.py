"""The mxquant model's perplexity path (models/mxquant/workload.py) without a GPU run: the rule lists choose the
right layers, the two paths of the model produce the same bits on a linear layer, the cache key is what it
should be, availability is honest, the line formats. With --gpu it also measures one sample on GPU 0 and
checks the cache.

    .venv/bin/python tests/selftest_workload.py
    .venv/bin/python tests/selftest_workload.py --gpu       # + one real sample (~2 min: model load + torch.compile)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def main() -> int:
    import models
    if not models.paths():
        print(f"SKIP: {models.mxq_missing()}")
        return 0
    import torch
    from torch import nn
    from config import scheme
    from config.recipe import RecipeError, load
    from models.mxquant import kernel as K
    from models.mxquant import rules, workload as W, workloads
    from mxq.nn import MXLinear, patch

    base = load("baseline")
    wide = load("wide_acc")

    print("\n[1] rule lists on a toy model -------------------------------------------")

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(64, 64) for _ in range(4))

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj, self.down_proj = nn.Linear(64, 128), nn.Linear(128, 64)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.mlp = Attn(), Mlp()

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])
            self.lm_head = nn.Linear(64, 1000)

    toy = Toy()
    s = scheme.scheme(base)                                   # uncompiled: patch's smoke matmul runs on the CPU
    h = patch(toy, rules.build("mxquant_layers", s), dry_run=True)
    got = {name: sch for name, _, _, _, sch in h.table}
    attn = [n for n in got if ".self_attn." in n]
    rest = [n for n in got if ".self_attn." not in n]
    check("mxquant_layers leaves every attention projection alone", attn and all(got[n] is None for n in attn),
          f"{len(attn)} projections")
    check("mxquant_layers puts the Scheme on every other Linear incl. lm_head",
          rest and all(got[n] == s.name for n in rest) and "lm_head" in got, f"{len(rest)} layers -> {s.name}")
    h2 = patch(toy, rules.build("all_linear", s), dry_run=True)
    check("all_linear puts the Scheme on every Linear", all(sch == s.name for *_, sch in h2.table),
          f"{len(h2.table)} layers")
    h3 = patch(toy, rules.build("linears_no_head", s), dry_run=True)
    got3 = {name: sch for name, _, _, _, sch in h3.table}
    check("linears_no_head leaves lm_head alone and takes every other Linear, attention included",
          got3.get("lm_head", "missing") is None and all(v == s.name for n, v in got3.items() if n != "lm_head"),
          f"{sum(1 for v in got3.values() if v)} of {len(got3)}")
    check("dry_run changed nothing", all(isinstance(m, nn.Linear) for m in toy.modules() if hasattr(m, "weight")))
    try:
        rules.build("everything", s)
        check("unknown rule list is refused", False)
    except ValueError as exc:
        check("unknown rule list is refused", "everything" in str(exc))

    print("\n[2] one Scheme, two paths: the bit path and the patched layer give the same bits ---")
    A, B = torch.randn(64, 32), torch.randn(64, 48)
    y_scheme = s.matmul(A, B)
    arith, sched, window = scheme.datapath(base)
    from mxq import matmul
    PA, XA = scheme.quantizer(base)(A)
    PB, XB = scheme.quantizer(base)(B)
    y_direct = matmul.systolic(PA, XA, PB, XB, arith, sched, window=window, block_size=base.block)
    check("Scheme.matmul == quantizer + datapath from the same recipe", torch.equal(y_scheme, y_direct))
    check("wide_acc gives different bits from baseline", not torch.equal(y_scheme, scheme.scheme(wide).matmul(A, B)))

    print("\n[2b] the reducer choice: same quantizers, a different multiply ------------------")
    from mxq import block
    ex = scheme.scheme(base, reduce="exact")
    Aq = block.dequantize(PA, XA, axis=0, block_size=base.block)
    Bq = block.dequantize(PB, XB, axis=0, block_size=base.block)
    check("exact == the float64 product of the dequantized operands",
          torch.equal(ex.matmul(A, B), (Aq.double().t() @ Bq.double()).float()))
    check("exact shares the quantizers with hardware", ex.a is not None and ex.a.keywords == s.a.keywords)
    check("exact differs from the hardware array", not torch.equal(ex.matmul(A, B), y_scheme))
    bt = scheme.scheme(base, reduce="bf16_tiles")
    A1, B1 = torch.randn(base.block, 32), torch.randn(base.block, 48)         # one block: no cross-block step
    PA1, XA1 = scheme.quantizer(base)(A1)
    PB1, XB1 = scheme.quantizer(base)(B1)
    S = torch.zeros(32, 48)
    for k in range(base.block):                                              # fp32 products, fp32 adds, in order
        S = S + PA1[k].unsqueeze(1) * PB1[k].unsqueeze(0)
    want = (S * (XA1[0].unsqueeze(1) * XB1[0].unsqueeze(0))).to(torch.bfloat16).float()
    check("bf16_tiles on one block == the fp32 block sum rounded to bf16 once", torch.equal(bt.matmul(A1, B1), want))
    y_bt = bt.matmul(A, B)
    check("bf16_tiles on two blocks differs from both exact and hardware",
          not torch.equal(y_bt, ex.matmul(A, B)) and not torch.equal(y_bt, y_scheme))
    check("Scheme names say which reducer ran", (s.name, ex.name, bt.name) == ("baseline", "baseline/exact", "baseline/bf16_tiles"))
    try:
        scheme.scheme(base, reduce="fp32")
        check("an unknown reducer is refused", False)
    except RecipeError as exc:
        check("an unknown reducer is refused", "fp32" in str(exc))

    from compiler.lower import lower
    from kernels.registry import build

    def mxlinear(weight_kn: torch.Tensor, sch) -> MXLinear:            # the kernel's W[K][N] as an nn.Linear
        lin = nn.Linear(weight_kn.shape[0], weight_kn.shape[1], bias=False)
        lin.weight.data = weight_kn.t().contiguous().float()
        return MXLinear(lin, sch)
    for rname, r in (("baseline", base), ("wide_acc", wide)):
        sch = scheme.scheme(r)
        spec = build("linear")
        y_bits = K.run(spec, r, dtype="fp8_e4m3", edges=lower(spec, "fp8_e4m3").edges, shipped=False)["y"]
        y_layer = mxlinear(spec.stages[0].weight, sch)(spec.x.float())
        check(f"{rname}: linear kernel bits == MXLinear on the same tensors",
              torch.equal(torch.from_numpy(y_bits), y_layer), f"{int((torch.from_numpy(y_bits) != y_layer).sum())}/4096 differ")
        spec2 = build("mlp2")
        y_bits = K.run(spec2, r, dtype="fp8_e4m3", edges=lower(spec2, "fp8_e4m3").edges, shipped=False)["y"]
        l1, l2 = (mxlinear(st.weight, sch) for st in spec2.stages)
        y_layer = l2(l1(spec2.x.float()).to(torch.bfloat16).float())      # the device requantizer reads bf16
        check(f"{rname}: mlp2 chain bits == two MXLinear with the bf16 accumulator between",
              torch.equal(torch.from_numpy(y_bits), y_layer), f"{int((torch.from_numpy(y_bits) != y_layer).sum())}/4096 differ")

    print("\n[3] workloads and the cache key -----------------------------------------------")
    w = workloads.build("tinyllama")
    check("tinyllama is registered as MXQuant measured it", (w.nsamples, w.seqlen, w.seed, w.rules) == (16, 2048, 0, "mxquant_layers"))
    check("a field can be overridden", workloads.build("tinyllama", nsamples=4).nsamples == 4)
    try:
        workloads.build("tinyllama", batch=3)
        check("an unknown field is refused", False)
    except ValueError as exc:
        check("an unknown field is refused", "batch" in str(exc))
    try:
        workloads.build("nope")
        check("an unknown workload is refused", False)
    except ValueError as exc:
        check("an unknown workload is refused", "nope" in str(exc))
    k0 = W.key("tinyllama", base)
    check("key is stable", k0 == W.key("tinyllama", base) and len(k0) == 16, k0)
    check("key changes with the recipe", k0 != W.key("tinyllama", wide))
    check("key changes with the operand format", k0 != W.key("tinyllama", base, dtype="fp4_e2m1"))
    check("key is the same for the recipe's own format spelled out", k0 == W.key("tinyllama", base, dtype="fp8_e4m3"))
    check("key changes with the operand rounding", k0 != W.key("tinyllama", base, rounding_mode="ties_away"))
    check("key changes with the scale floor", k0 != W.key("tinyllama", base, scale_floor=1e-38))
    check("key changes with nsamples", k0 != W.key("tinyllama", base, nsamples=4))
    check("key changes for the whole split and for samples in order",
          len({k0, W.key("tinyllama", base, nsamples=0), W.key("tinyllama", base, seed=None), W.key("tinyllama", base, nsamples=0, seed=None)}) == 4)
    check("nsamples=0 and seed=None are accepted overrides",
          workloads.build("tinyllama", nsamples=0, seed=None).seed is None and workloads.build("tinyllama", nsamples=0).nsamples == 0)
    check("key changes with the reducer", k0 != W.key("tinyllama", base, reduce="exact") != W.key("tinyllama", base, reduce="bf16_tiles"))
    check("the default reducer leaves every existing key as it was", k0 == W.key("tinyllama", base, reduce="hardware")
          and "reduce" not in W._settings(w, base, dtype="fp8_e4m3", rounding_mode="rne", scale_floor=2.0 ** -23))
    check("bf16 key does not depend on the recipe", W.key("tinyllama", None) == W.key("tinyllama", None, rounding_mode="ties_away"))
    st = W._settings(w, base, dtype="fp8_e4m3", rounding_mode="rne", scale_floor=2.0 ** -23)
    check("key includes the mxq commit", models.mxq_commit() and st["mxq_commit"] == models.mxq_commit())
    check("a direct format carries no codebook note", "codebook" not in st)
    st6 = W._settings(w, base, dtype="fp6_e3m2", rounding_mode="rne", scale_floor=2.0 ** -23)
    check("a codebook format is run on the full grid and says so", st6.get("codebook") == "not modelled" and st6["format"] == "MXFP6_E3M2")

    print("\n[4] refusals and availability ---------------------------------------------")
    try:
        W.evaluate("tinyllama", base, dtype="fp9", results_dir=Path(tempfile.mkdtemp()))
        check("an unknown operand format is refused before any GPU work", False, "no RecipeError")
    except RecipeError as exc:
        check("an unknown operand format is refused before any GPU work", True, str(exc)[:70])
    check("a codebook format builds a Scheme (full grid)", scheme.scheme(base, dtype="fp6_e3m2").a.keywords["fmt"] == "MXFP6_E3M2")
    ok, why = W.available()
    check("available() answers with a reason", isinstance(ok, bool) and isinstance(why, str), why)

    print("\n[5] the PPL line -------------------------------------------------------------")
    fake = {"perplexity": 7.343833, "bf16_perplexity": 7.188465, "delta": 7.343833 - 7.188465, "workload": "tinyllama",
            "dtype": "fp8_e4m3", "model_id": w.model_id, "nsamples": 16, "seqlen": 2048, "rules": "mxquant_layers",
            "seconds": 970.4, "cached": True, "codebook": None}
    ln = W.line(fake)
    check("line names the number, the baseline, the delta and [cached]",
          ln.startswith("PPL") and "7.3438" in ln and "7.1885" in ln and "+0.1554" in ln and "[cached]" in ln, ln)
    check("line flags a codebook format", "codebook not modelled" in W.line({**fake, "codebook": "not modelled"}))
    check("line names a non-default reducer", "reduce exact" in W.line({**fake, "reduce": "exact"}) and "reduce" not in ln)
    check("line says 'all' and 'in order' for those sample choices",
          "allx2048 in order" in W.line({**fake, "nsamples": 0, "seed": None}) and "in order" not in ln)

    print("\n[6] the standing numbers (tests/oracle/accuracy_baseline.json) ------------------")
    import json
    oracle = json.loads((REPO / "tests" / "oracle" / "accuracy_baseline.json").read_text())
    env = W.environment()
    check("oracle names the recipe build_id it was taken on", oracle["build_id"] == base.build_id(), base.build_id())
    n_checked = 0
    for e in oracle["entries"]:
        if (e["torch"], e["transformers"]) != (env["torch"], env["transformers"]):
            continue
        r = None if e["recipe"] is None else base
        k = W.key("tinyllama", r, nsamples=oracle["nsamples"], seqlen=oracle["seqlen"], seed=oracle["seed"],
                  rules=oracle["rules"], rounding_mode=e["rounding_mode"] or scheme.ROUNDING, scale_floor=e["scale_floor"],
                  reduce=e.get("reduce", "hardware"))
        path = W.RESULTS / f"{k}.json"
        if not path.exists():
            print(f"  skip  {e['name']}: not measured in this environment / mxq commit yet ({path.name})")
            continue
        n_checked += 1
        got = json.loads(path.read_text())["perplexity"]
        check(f"{e['name']} == {e['perplexity']!r}", got == e["perplexity"], f"measured {got!r}")
    if n_checked == 0:
        print(f"  (no cached measurement for torch {env['torch']} / transformers {env['transformers']} / mxq "
              f"{models.mxq_commit()}; python -m models.mxquant --workload tinyllama --config baseline --gpus 0,1,2,3 produces one)")

    if "--gpu" in sys.argv:
        print("\n[7] one real sample on GPU 0 ---------------------------------------------------")
        if not ok:
            check("GPU run", False, why)
        else:
            tmp = Path(tempfile.mkdtemp(prefix="workload_selftest_"))
            m = W.evaluate("tinyllama", base, nsamples=1, gpus="0", results_dir=tmp)
            check("one sample measured", m["perplexity"] > 1 and m["bf16_perplexity"] > 1 and not m["cached"], W.line(m))
            check("cache files written", Path(m["path"]).exists() and Path(m["bf16_path"]).exists())
            check("layers counted", m["layers_quantized"] > 0 and m["layers_quantized"] < m["layers_total"],
                  f"{m['layers_quantized']}/{m['layers_total']}")
            m2 = W.evaluate("tinyllama", base, nsamples=1, gpus="0", results_dir=tmp)
            check("second call is served from the cache with the same number",
                  m2["cached"] and m2["perplexity"] == m["perplexity"], W.line(m2))
    else:
        print("\n[7] skipped: pass --gpu to measure one real sample")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("ALL CHECKS PASSED -- the perplexity path runs the recipe's Scheme, and the same bits as the bit path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
