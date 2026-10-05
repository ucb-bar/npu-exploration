"""The perplexity path's LUT operands == the kernel path's, bit for bit (LUT_integration.md, link L5).

The kernel path quantizes an operand to the wire (``compiler.operands.quantize_operand``: block codes, a table
per 2**G rows of A / columns of B, 4-bit indices) and the mxquant model reads it back (``wire_to_px``); spike
equals that model on every LUT kernel. The perplexity path quantizes the same operand with
``config.scheme.quantizer`` (``mxq.block.lut``). If the two give the same (P, X) for A and for B, a layer's
output is the same Scheme reducer on the same codes, so the perplexity number is computed on what the chip
would multiply.

For each LUT format on its build, G = 0, 1, 2, A (activations, 2**G tokens per table) and B (weights, 2**G
output channels per table): P and X equal (torch.equal), on TinyLlama q/gate/down weights and token embeddings
when the checkpoint is cached, else on random operands with outliers. Then Scheme.matmul equals the reducer on
the wire operands. Also: MXLinear with the recipe's Scheme gives the same output for every chunk size.

    .venv/bin/python tests/selftest_lut_layer.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models  # noqa: F401,E402
from config import scheme as S  # noqa: E402
from config.recipe import check, lut_settings  # noqa: E402
from compiler.operands import quantize_operand, wire_to_px  # noqa: E402
from mxq.nn import MXLinear  # noqa: E402
from tests.fixtures import luts  # noqa: E402

FAILS: list[str] = []


def ok(cond: bool, what: str) -> None:
    if not cond:
        FAILS.append(what)
        print("  FAIL", what)


def tensors():
    """(name, A M×K activations, W N×K weight): real TinyLlama slices if cached, else random with outliers."""
    try:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float32,
                                                 local_files_only=True)
        g = torch.Generator().manual_seed(0)
        tok = torch.randint(0, m.config.vocab_size, (64,), generator=g)
        A = (m.model.embed_tokens.weight[tok] * 40).detach()[:, :256].contiguous()      # 64 tokens × K 256
        L = m.model.layers[5]
        Ws = [("q_proj", L.self_attn.q_proj.weight), ("gate_proj", L.mlp.gate_proj.weight),
              ("down_proj", L.mlp.down_proj.weight)]
        return [(n, A, W[:64, :256].detach().contiguous()) for n, W in Ws]
    except Exception as e:                                    # noqa: BLE001
        print(f"  (TinyLlama not cached: {type(e).__name__}; random operands)")
        g = torch.Generator().manual_seed(0)
        A = torch.randn(64, 256, generator=g) * torch.where(torch.rand(64, 256, generator=g) < 0.01, 30.0, 1.0)
        return [("random", A, torch.randn(64, 256, generator=g) * 0.02)]


def main() -> int:
    print("selftest_lut_layer: perplexity LUT operands == kernel wire operands")
    cases = tensors()
    n = 0
    for fmt in [f for f in luts.PAIRS if S.is_codebook(f)]:
        hw, run0 = luts.recipes(fmt)
        for g in (0, 1, 2):
            run = dataclasses.replace(run0, lut=dataclasses.replace(run0.lut, group=g))
            check(hw, run, "perplexity")
            check(hw, run, "kernel")
            q = S.quantizer(hw, run)
            settings = lut_settings(hw, run)
            for name, A, W in cases:
                tag = f"{fmt} G={g} {name}"
                B = W.t().contiguous()                                            # K×N
                # A side: the wire takes A as M×K; the Scheme takes Aᵀ (K×M)
                ca, sa, ba = quantize_operand(A.numpy(), side="a", dtype=fmt, lut=settings)
                PA_w, XA_w = wire_to_px(ca, sa, side="a", dtype=fmt, books=ba, shape=tuple(A.shape), lut=settings)
                PA, XA = q(A.t().contiguous())
                ok(torch.equal(PA, PA_w) and torch.equal(XA, XA_w), f"A operand {tag}")
                cb, sb, bb = quantize_operand(B.numpy(), side="b", dtype=fmt, lut=settings)
                PB_w, XB_w = wire_to_px(cb, sb, side="b", dtype=fmt, books=bb, shape=tuple(B.shape), lut=settings)
                PB, XB = q(B)
                ok(torch.equal(PB, PB_w) and torch.equal(XB, XB_w), f"B operand {tag}")
                sch = S.scheme(hw, run)
                ok(torch.equal(sch.matmul(A.t().contiguous(), B), sch.reduce(PA_w, XA_w, PB_w, XB_w)),
                   f"matmul {tag}")
                n += 1
            # MXLinear: chunking cannot move a bit (tables never straddle a chunk)
            name, A, W = cases[0]
            lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=False)
            lin.weight.data = W.clone()
            sch = S.scheme(hw, run)
            whole = MXLinear(lin, sch, chunk=A.shape[0])(A)
            for chunk in sorted({1 << g, 8, 16, 32}):
                ok(torch.equal(MXLinear(lin, sch, chunk=chunk)(A), whole), f"MXLinear chunk {chunk} {fmt} G={g}")
    print(f"  {n} (format, G, layer) cases, A and B and matmul each")
    print("FAIL" if FAILS else "PASS", f"({len(FAILS)} failures)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
