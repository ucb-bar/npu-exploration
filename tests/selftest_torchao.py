"""Self-test of mxq.nn.torchao: mxq's arithmetic behind torchao.quantize_, the same as mxq.nn.patch.

Six claims:
  1. mxq imports without torchao: the adapter is optional, and nothing else in mxq pulls it in.
  2. MXQConfig refuses what it cannot build (unknown format or reducer, a ladder that is not one format per lane,
     a malformed product), and survives torchao's config_to_dict and MXQConfig.from_dict unchanged.
  3. mxq_config(hw, run).scheme() is the Scheme config.scheme.scheme(hw, run) is: bit-identical matmul for every
     hardware recipe x run recipe (bf16_tiles, npu-only, refused) x all six formats, with rounding and floor
     variants, on inputs that include an all-zero block and a tiny one.
  4. A model through quantize_(model, cfg) computes what patch(model, [(nn.Linear, scheme)]) does, bit for bit,
     eager on the CPU (a small random Llama).
  5. The same with the compiled Arithmetic on a GPU, and both equal eager (skipped with no GPU).
  6. Hugging Face: TinyLlama loaded with TorchAoConfig(cfg) converts every Linear but lm_head, in place, and its
     logits equal patch's (skipped if the model is not in the local cache).

    PYTHONPATH=<dir with torchao> .venv/bin/python tests/selftest_torchao.py
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import models  # noqa: E402
from config import scheme  # noqa: E402
from config.recipe import RecipeError, list_hardware, list_runs, load_hardware, load_run  # noqa: E402
from config.recipe import check as check_recipes  # noqa: E402

FAILURES: list[str] = []
TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def raises(fn, exc=ValueError) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def operands(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A: K×M, B: K×N, K = 96 (three blocks). Block 0 of A's column 0 all zero, block 1 of B's column 0 tiny."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(96, 8, generator=g) * torch.logspace(-3, 3, 8)
    B = torch.randn(96, 12, generator=g)
    A[0:32, 0] = 0.0
    B[32:64, 0] *= 2.0 ** -30
    return A, B


def small_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).eval()


def main() -> int:
    ok, why = models.paths(), models.mxq_missing()
    if not ok:
        print(f"SKIP: {why}")
        return 0

    print("\n[1] optional ------------------------------------------------------------")
    env = dict(os.environ, PYTHONPATH="")
    code = ("import sys; sys.path.insert(0, %r); import models, mxq, mxq.nn; "
            "import importlib.util as u; print(u.find_spec('torchao') is None, 'torchao' in sys.modules)" % str(REPO))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=REPO)
    check("import mxq, mxq.nn without torchao on the path", out.returncode == 0 and out.stdout.split()[-2:] == ["True", "False"],
          (out.stdout + out.stderr).strip().splitlines()[-1] if (out.stdout + out.stderr).strip() else "")

    try:
        from torchao.core.config import config_to_dict
        from torchao.quantization import quantize_
        from mxq.nn.torchao import MXQConfig
    except ImportError as exc:
        print(f"SKIP [2]-[6]: torchao is not importable ({exc})")
        return 1 if FAILURES else 0
    from mxq.nn import MXLinear, patch

    print("\n[2] refusals and serialization -------------------------------------------")
    check("unknown format", raises(lambda: MXQConfig(fmt="MXFP5")))
    check("FP32 is no quantization", raises(lambda: MXQConfig(fmt="FP32")))
    check("unknown reducer", raises(lambda: MXQConfig(reduce="bf16_tiles")))
    check("ladder shorter than the window", raises(lambda: MXQConfig(ladder=[[4, 4]] * 8)))
    check("ladder entry not [e, m]", raises(lambda: MXQConfig(ladder=[[4, 4]] * 15 + [[4]])))
    check("product not [e, m]", raises(lambda: MXQConfig(prod=[4, 0])))
    check("prod_floor not an integer", raises(lambda: MXQConfig(prod_floor=-16.5)))
    check("block not a multiple of the window", raises(lambda: MXQConfig(block_size=24)))
    c = MXQConfig(prod=(4, 3), via=(3, 1), fmt="MXFP4")
    check("tuples become lists (torchao's encoder refuses tuples)", c.prod == [4, 3] and c.via == [3, 1])
    d = config_to_dict(c)
    back = MXQConfig.from_dict(d)
    check("config_to_dict -> from_dict is the same config", back == c and d["_type"] == "MXQConfig", str(d["_data"])[:80])
    check("from_dict refuses another config's dict", raises(lambda: MXQConfig.from_dict({"_type": "Float8Config", "_data": {}})))

    print("\n[3] mxq_config(hw, run).scheme() == scheme(hw, run) -----------------------")
    runs = [load_run(r) for r in list_runs()]
    runs = [r for r in runs if r.lut is None]   # each base run, crossed with every format below
    variants = []
    for r in runs:
        if r.reduce == "bf16_tiles":
            try:
                scheme.mxq_config(load_hardware("baseline"), r)
                check("bf16_tiles is refused", False)
            except RecipeError:
                check("bf16_tiles is refused", True)
            continue
        for fmt in scheme.MXQ_FORMAT:
            # A LUT format carries its run recipe's lut block (config/run/<fmt>.json), never one made here.
            lut = load_run(fmt).lut if scheme.is_codebook(fmt) else None
            variants.append(dataclasses.replace(r, operand_fmt=fmt, lut=lut))
        variants.append(dataclasses.replace(r, rounding="ties_away", scale_floor=1e-38))
    n = bad = 0
    for hname in list_hardware():
        hw = load_hardware(hname)
        for run in variants:
            try:
                check_recipes(hw, run, "perplexity")
            except RecipeError:
                continue                            # a LUT format this build's LUT unit does not serve
            want_s, got_s = scheme.scheme(hw, run), scheme.mxq_config(hw, run).scheme()
            for seed in (0, 1):
                A, B = operands(seed)
                n += 1
                if not torch.equal(want_s.matmul(A, B), got_s.matmul(A, B)):
                    bad += 1
                    print(f"    differs: {hname} {run.name} {run.operand_fmt} {run.rounding} seed {seed}")
    check(f"bit-identical on {n} (hardware, run, format, seed) cases", bad == 0, f"{bad} differ")

    print("\n[4] quantize_ == patch, eager, CPU ----------------------------------------")
    hw, run = load_hardware("baseline"), load_run("default")
    ids = torch.randint(0, 256, (2, 40), generator=torch.Generator().manual_seed(0))
    for fmt in ("fp8_e4m3", "fp4_e2m1"):
        r = dataclasses.replace(run, operand_fmt=fmt)
        with torch.no_grad():
            m_p = small_llama()
            patch(m_p, [(torch.nn.Linear, scheme.scheme(hw, r))])
            y_p = m_p(ids).logits
            m_q = small_llama()
            linears = [mod for mod in m_q.modules() if isinstance(mod, torch.nn.Linear)]
            quantize_(m_q, scheme.mxq_config(hw, r))
            y_q = m_q(ids).logits
        same_objects = all(isinstance(mod, torch.nn.Linear) and hasattr(mod, "_mxq") for mod in linears)
        check(f"{fmt}: every Linear converted in place ({len(linears)})", same_objects)
        check(f"{fmt}: no duplicate parameters", len(list(m_q.parameters())) == len(list(small_llama().parameters())))
        check(f"{fmt}: logits equal patch's", torch.equal(y_p, y_q), f"max diff {float((y_p - y_q).abs().max()):.3g}")

    print("\n[5] compiled, GPU ---------------------------------------------------------")
    if not torch.cuda.is_available():
        print("  SKIP  no GPU")
    else:
        import torch._inductor.config as inductor_config
        inductor_config.cpp.simdlen = 1      # patch's CPU smoke test of a compiled Scheme, as in models/mxquant/_worker.py
        dev = "cuda:0"
        with torch.no_grad():
            m_e = small_llama().to(dev)
            quantize_(m_e, scheme.mxq_config(hw, run))
            y_e = m_e(ids.to(dev)).logits
            m_c = small_llama().to(dev)
            quantize_(m_c, scheme.mxq_config(hw, run, compiled=True))
            y_c = m_c(ids.to(dev)).logits
            m_pc = small_llama().to(dev)
            patch(m_pc, [(torch.nn.Linear, scheme.scheme(hw, run, compiled=True))])
            y_pc = m_pc(ids.to(dev)).logits
        check("compiled quantize_ == eager quantize_", torch.equal(y_c, y_e))
        check("compiled quantize_ == compiled patch", torch.equal(y_c, y_pc))

    print("\n[6] Hugging Face TorchAoConfig ---------------------------------------------")
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = isinstance(try_to_load_from_cache(TINYLLAMA, "config.json"), str)
    except Exception:
        cached = False
    if not cached or not torch.cuda.is_available():
        print(f"  SKIP  needs {TINYLLAMA} in the local cache and a GPU")
    else:
        from transformers import AutoModelForCausalLM, TorchAoConfig
        dev = "cuda:0"
        kw = dict(dtype=torch.bfloat16, device_map=dev)
        tok = torch.randint(100, 30000, (1, 32), generator=torch.Generator().manual_seed(0)).to(dev)
        cfg = scheme.mxq_config(hw, run, compiled=True)
        with torch.no_grad():
            ref = AutoModelForCausalLM.from_pretrained(TINYLLAMA, **kw).eval()
            eligible = sum(isinstance(m, torch.nn.Linear) for n_, m in ref.named_modules() if n_ != "lm_head")
            patch(ref, [("lm_head", None), (torch.nn.Linear, cfg.scheme())])
            y_ref = ref(tok).logits
            del ref
            torch.cuda.empty_cache()
            hf = AutoModelForCausalLM.from_pretrained(TINYLLAMA, quantization_config=TorchAoConfig(cfg), **kw).eval()
            converted = sum(hasattr(m, "_mxq") for m in hf.modules())
            y_hf = hf(tok).logits
        check(f"HF converts every Linear but lm_head in place ({converted}/{eligible})", converted == eligible)
        check("HF logits equal patch's", torch.equal(y_hf, y_ref), f"max diff {float((y_hf.float() - y_ref.float()).abs().max()):.3g}")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("ALL CHECKS PASSED -- quantize_(model, mxq_config(hw, run)) is patch(model, scheme(hw, run)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
