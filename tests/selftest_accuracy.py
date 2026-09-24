"""models/accuracy without a GPU run: the rule lists choose the right layers, the cache key is what it should be,
availability is honest, the line formats. With --gpu it also measures one sample on GPU 0 and checks the cache.

    .venv/bin/python tests/selftest_accuracy.py
    .venv/bin/python tests/selftest_accuracy.py --gpu       # + one real sample (~2 min: model load + torch.compile)
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
    from models.accuracy import accuracy, rules
    from mxq.nn import patch

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
    check("dry_run changed nothing", all(isinstance(m, nn.Linear) for m in toy.modules() if hasattr(m, "weight")))
    try:
        rules.build("everything", s)
        check("unknown rule list is refused", False)
    except ValueError as exc:
        check("unknown rule list is refused", "everything" in str(exc))

    print("\n[2] the Scheme the model runs is the recipe's ------------------------------")
    A, B = torch.randn(64, 32), torch.randn(64, 48)
    y_scheme = s.matmul(A, B)
    arith, sched, window = scheme.datapath(base)
    from mxq import matmul
    PA, XA = scheme.quantizer(base)(A)
    PB, XB = scheme.quantizer(base)(B)
    y_direct = matmul.systolic(PA, XA, PB, XB, arith, sched, window=window, block_size=base.block)
    check("Scheme.matmul == quantizer + datapath from the same recipe", torch.equal(y_scheme, y_direct))
    check("wide_acc gives different bits from baseline", not torch.equal(y_scheme, scheme.scheme(wide).matmul(A, B)))

    print("\n[3] cache key --------------------------------------------------------------")
    k0 = accuracy.key(base)
    check("key is stable", k0 == accuracy.key(base) and len(k0) == 16, k0)
    check("key changes with the recipe", k0 != accuracy.key(wide))
    check("key changes with the operand rounding", k0 != accuracy.key(base, rounding_mode="ties_away"))
    check("key changes with the scale floor", k0 != accuracy.key(base, scale_floor=1e-38))
    check("key changes with nsamples", k0 != accuracy.key(base, nsamples=4))
    check("bf16 key does not depend on the recipe", accuracy.key(None) == accuracy.key(None, rounding_mode="ties_away"))
    check("key includes the mxq commit", models.mxq_commit() and accuracy._settings(
        base, model_id=accuracy.MODEL_ID, nsamples=16, seqlen=2048, seed=0, rules="mxquant_layers",
        rounding_mode="rne", scale_floor=2.0 ** -23)["mxq_commit"] == models.mxq_commit())

    print("\n[4] refusals and availability ---------------------------------------------")
    import dataclasses
    fp6 = dataclasses.replace(base, operand_fmt="fp6")
    try:
        accuracy.run(fp6, results_dir=Path(tempfile.mkdtemp()))
        check("codebook recipe is refused before any GPU work", False, "no RecipeError")
    except RecipeError as exc:
        check("codebook recipe is refused before any GPU work", True, str(exc)[:70])
    ok, why = accuracy.available()
    check("available() answers with a reason", isinstance(ok, bool) and isinstance(why, str), why)

    print("\n[5] the PPL line -------------------------------------------------------------")
    fake = {"perplexity": 7.343833, "bf16_perplexity": 7.188465, "delta": 7.343833 - 7.188465,
            "model_id": accuracy.MODEL_ID, "nsamples": 16, "seqlen": 2048, "rules": "mxquant_layers",
            "seconds": 970.4, "cached": True}
    ln = accuracy.line(fake)
    check("line names the number, the baseline, the delta and [cached]",
          ln.startswith("PPL") and "7.3438" in ln and "7.1885" in ln and "+0.1554" in ln and "[cached]" in ln, ln)

    print("\n[6] the standing numbers (tests/oracle/accuracy_baseline.json) ------------------")
    import json
    oracle = json.loads((REPO / "tests" / "oracle" / "accuracy_baseline.json").read_text())
    env = accuracy.environment()
    check("oracle names the recipe build_id it was taken on", oracle["build_id"] == base.build_id(), base.build_id())
    n_checked = 0
    for e in oracle["entries"]:
        if (e["torch"], e["transformers"]) != (env["torch"], env["transformers"]):
            continue
        r = None if e["recipe"] is None else base
        k = accuracy.key(r, nsamples=oracle["nsamples"], seqlen=oracle["seqlen"], seed=oracle["seed"],
                         rules=oracle["rules"], rounding_mode=e["rounding_mode"] or scheme.ROUNDING,
                         scale_floor=e["scale_floor"])
        path = accuracy.RESULTS / f"{k}.json"
        if not path.exists():
            print(f"  skip  {e['name']}: not measured in this environment yet ({path.name})")
            continue
        n_checked += 1
        got = json.loads(path.read_text())["perplexity"]
        check(f"{e['name']} == {e['perplexity']!r}", got == e["perplexity"], f"measured {got!r}")
    if n_checked == 0:
        print(f"  (no cached measurement for torch {env['torch']} / transformers {env['transformers']}; "
              f"python -m models.accuracy --config baseline --gpus 0,1,2,3 produces one)")

    if "--gpu" in sys.argv:
        print("\n[7] one real sample on GPU 0 ---------------------------------------------------")
        if not ok:
            check("GPU run", False, why)
        else:
            tmp = Path(tempfile.mkdtemp(prefix="accuracy_selftest_"))
            m = accuracy.run(base, nsamples=1, gpus="0", results_dir=tmp)
            check("one sample measured", m["perplexity"] > 1 and m["bf16_perplexity"] > 1 and not m["cached"],
                  accuracy.line(m))
            check("cache files written", Path(m["path"]).exists() and Path(m["bf16_path"]).exists())
            check("layers counted", m["layers_quantized"] > 0 and m["layers_quantized"] < m["layers_total"],
                  f"{m['layers_quantized']}/{m['layers_total']}")
            m2 = accuracy.run(base, nsamples=1, gpus="0", results_dir=tmp)
            check("second call is served from the cache with the same number",
                  m2["cached"] and m2["perplexity"] == m["perplexity"], accuracy.line(m2))
    else:
        print("\n[7] skipped: pass --gpu to measure one real sample")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("ALL CHECKS PASSED -- the accuracy model patches the right layers with the recipe's Scheme.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
