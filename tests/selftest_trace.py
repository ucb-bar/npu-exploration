"""Self-test of the torch.fx -> KernelSpec frontend (kernels/trace.py).

Four layers:
  1. EQUIVALENCE with the hand-written registry: tracing the same torch module
     (same seed) must yield the same stages, weights and fp32 reference as
     `linear` and the mlp2-style chain -- the frontend may not change semantics.
  2. FIDELITY: for every traced module, spec.reference() == module(x) in fp32.
     This is the property that makes tracing trustworthy at all.
  3. GRAPH shapes: an attention-style module (matmul / scale / causal mask /
     softmax / matmul) lowers to the right stage kinds, all emittable, and a
     swiglu MLP folds silu(g)*u into one host stage.
  4. FAIL-CLOSED: bias, activations, standalone transpose/silu, bare multiply
     all raise TraceError -- nothing is silently approximated.

No hardware, no toolchain, no spike.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kernels.registry import build            # noqa: E402
from kernels.spec import HostStage, Stage     # noqa: E402
from kernels.trace import TRANSLATORS, RoPE, TraceError, rmsnorm_translator, trace   # noqa: E402

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append(bool(ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


def raises(label, fn):
    try:
        fn()
        check(label, False, "no TraceError raised")
    except TraceError as exc:
        check(label, True, str(exc)[:60])


def close(a, b):
    return np.allclose(np.asarray(a, np.float32), np.asarray(b, np.float32),
                       rtol=1e-5, atol=1e-6)


def main() -> int:
    print("equivalence: traced module == hand-written registry kernel")
    torch.manual_seed(0)
    layer = nn.Linear(64, 64, bias=False)
    x = torch.randn(64, 64)
    t = trace(layer, x, name="linear")
    r = build("linear", m=64, k=64, n=64, seed=0)
    check("linear: one mesh stage", len(t.stages) == 1 and isinstance(t.stages[0], Stage))
    check("linear: weight identical to registry",
          torch.equal(t.stages[0].weight, r.stages[0].weight))
    check("linear: x identical to registry", torch.equal(t.x, r.x))
    check("linear: fp32 reference identical", close(t.reference(), r.reference()))

    torch.manual_seed(1)
    seq = nn.Sequential(nn.Linear(64, 64, bias=False), nn.Linear(64, 64, bias=False))
    xs = torch.randn(64, 64)
    ts = trace(seq, xs, name="mlp2")
    check("chain: is_chain (fused lowering)", ts.is_chain)
    check("chain: fp32 reference == module(x)",
          close(ts.reference(), seq(xs).detach().numpy()))

    print("graph: attention-style module")

    class Attn(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d, bias=False)
            self.v = nn.Linear(d, d, bias=False)
            self.register_buffer(
                "mask", torch.triu(torch.ones(d, d, dtype=torch.bool), diagonal=1))
            self.scale = 1.0 / math.sqrt(d)

        def forward(self, x):
            q, k, v = self.q(x), self.k(x), self.v(x)
            s = torch.matmul(q, k.t()) * self.scale
            p = F.softmax(s.masked_fill(self.mask, float("-inf")), dim=-1)
            return torch.matmul(p, v)

    torch.manual_seed(2)
    attn = Attn()
    xa = torch.randn(64, 64)
    ta = trace(attn, xa, name="attn")
    kinds = [(type(s).__name__, getattr(s, "op", None)) for s in ta.stages]
    check("attn: 5 mesh + 1 host softmax",
          sum(1 for s in ta.stages if isinstance(s, Stage)) == 5
          and [k for k in kinds if k[0] == "HostStage"] == [("HostStage", "softmax")],
          str(kinds))
    sm = next(s for s in ta.stages if isinstance(s, HostStage))
    check("attn: scale folded", abs(sm.params["scale"] - 1 / 8) < 1e-9,
          str(sm.params["scale"]))
    check("attn: causal mask recognized", sm.params["causal"] is True)
    check("attn: k.t() folded into rhs '.T'",
          any(isinstance(s, Stage) and s.rhs and s.rhs.endswith(".T") for s in ta.stages))
    check("attn: not a chain, all stages emittable",
          not ta.is_chain and all(getattr(s, "emittable", True) for s in ta.stages))
    check("attn: fp32 reference == module(x)",
          close(ta.reference(), attn(xa).detach().numpy()))

    print("graph: swiglu MLP")

    class SwiGLU(nn.Module):
        def __init__(self, d=64, h=64):
            super().__init__()
            self.g = nn.Linear(d, h, bias=False)
            self.u = nn.Linear(d, h, bias=False)
            self.d = nn.Linear(h, d, bias=False)

        def forward(self, x):
            return self.d(F.silu(self.g(x)) * self.u(x))

    torch.manual_seed(3)
    swi = SwiGLU()
    xw = torch.randn(64, 64)
    tw = trace(swi, xw, name="swiglu_mlp")
    check("swiglu: silu(g)*u folds to one host stage",
          [getattr(s, "op", None) for s in tw.stages if isinstance(s, HostStage)] == ["swiglu"])
    check("swiglu: fp32 reference == module(x)",
          close(tw.reference(), swi(xw).detach().numpy()))

    print("translators: leaves, RMSNorm, RoPE, validate")

    class MyNorm(nn.Module):                      # a custom module: inlined unless it is a leaf
        def __init__(self, d=64, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d) * 1.5)
            self.eps = eps

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    TRANSLATORS[MyNorm] = rmsnorm_translator
    torch.manual_seed(4)
    normed = nn.Sequential(MyNorm(), nn.Linear(64, 64, bias=False))
    xn = torch.randn(64, 64)
    tn = trace(normed, xn, name="normed")
    check("custom module with a translator is ONE rmsnorm stage, not inlined",
          [getattr(s, "op", None) for s in tn.stages] == ["rmsnorm", None])
    check("custom rmsnorm: fp32 reference == module(x)",
          close(tn.reference(), normed(xn).detach().numpy()))
    del TRANSLATORS[MyNorm]
    if hasattr(nn, "RMSNorm"):
        torch.manual_seed(5)
        tnn = nn.Sequential(nn.RMSNorm(64, eps=1e-5), nn.Linear(64, 64, bias=False))
        with torch.no_grad():
            tnn[0].weight.mul_(0.5).add_(0.75)
        xr = torch.randn(64, 64)
        trn = trace(tnn, xr, name="rmsnorm")
        check("nn.RMSNorm -> rmsnorm stage, fp32 reference == module(x)",
              getattr(trn.stages[0], "op", None) == "rmsnorm"
              and close(trn.reference(), tnn(xr).detach().numpy()))
    torch.manual_seed(6)
    pos = torch.arange(64).float()[:, None]
    freq = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
    ang = torch.cat([pos * freq, pos * freq], dim=-1)          # [64][64], duplicated halves as llama
    roped = nn.Sequential(nn.Linear(64, 64, bias=False), RoPE(ang.cos(), ang.sin()),
                          nn.Linear(64, 64, bias=False))
    xp = torch.randn(64, 64)
    tp = trace(roped, xp, name="roped")
    check("RoPE module -> rope host stage between two mesh stages",
          [getattr(s, "op", None) for s in tp.stages] == [None, "rope", None])
    check("rope: fp32 reference == module(x)", close(tp.reference(), roped(xp).detach().numpy()))
    from app import mxgraph
    g = mxgraph.from_spec(tp)
    rope_step = next(s for s in g.steps if getattr(s, "op", None) == "rope")
    check("rope: graph bakes cos/sin as consts", set(rope_step.const_names) == {"cos", "sin"}
          and all(n in g.consts for n in rope_step.const_names.values()))
    raises("validate=True refuses a shape the mesh cannot take (60 rows)",
           lambda: trace(nn.Linear(64, 64, bias=False), torch.randn(60, 64)))

    print("fail-closed")
    raises("bias refused",
           lambda: trace(nn.Linear(64, 64, bias=True), torch.randn(64, 64)))
    raises("activation module refused",
           lambda: trace(nn.Sequential(nn.Linear(64, 64, bias=False), nn.ReLU()),
                         torch.randn(64, 64)))

    class BareT(nn.Module):
        def forward(self, x):
            return x.t()
    raises("standalone transpose refused", lambda: trace(BareT(), torch.randn(64, 64)))

    class BareSilu(nn.Module):
        def forward(self, x):
            return F.silu(x)
    raises("standalone silu refused", lambda: trace(BareSilu(), torch.randn(64, 64)))

    class BareMul(nn.Module):
        def forward(self, x):
            return x * x
    raises("bare tensor multiply refused", lambda: trace(BareMul(), torch.randn(64, 64)))

    print()
    if all(CHECKS):
        print(f"ALL {len(CHECKS)} CHECKS PASSED -- torch -> KernelSpec tracing is faithful.")
        return 0
    print(f"{CHECKS.count(False)} of {len(CHECKS)} checks FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
