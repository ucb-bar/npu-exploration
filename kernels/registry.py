"""Named kernels. A kernel is DATA — a list of stages — not a module per type.

Adding one is a few lines here. A new *file* is only needed for something that is
not a chain of matmuls (attention's softmax, convolution); those need new lowering
in the backend, not a new builder.

    list_kernels()                 -> names
    build("mlp2", m=64, k=64, ...) -> KernelSpec
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from .spec import KernelSpec, from_module

#: name -> (builder, human description). Builders take keyword shape args only.
_KERNELS: dict[str, tuple[Callable[..., KernelSpec], str]] = {}


def register(name: str, description: str):
    def deco(fn: Callable[..., KernelSpec]) -> Callable[..., KernelSpec]:
        _KERNELS[name] = (fn, description)
        return fn
    return deco


def list_kernels() -> dict[str, str]:
    return {n: d for n, (_, d) in sorted(_KERNELS.items())}


def build(name: str, **kwargs: Any) -> KernelSpec:
    if name not in _KERNELS:
        raise KeyError(f"unknown kernel {name!r}; known: {sorted(_KERNELS)}")
    fn, _ = _KERNELS[name]
    return fn(**kwargs)


# --- the kernels ------------------------------------------------------------------------------

@register("linear", "one nn.Linear (K->N): a single MX matmul")
def _linear(*, m: int = 64, k: int = 64, n: int = 64, seed: int = 0, **_) -> KernelSpec:
    # Seed order matches app/torch_linear/run_linear.py exactly, so a graded run is
    # numerically identical to that baseline.
    torch.manual_seed(seed)
    layer = nn.Linear(k, n, bias=False)
    x = torch.randn(m, k)
    return from_module(layer, x, name="linear")


@register("mlp2", "two chained nn.Linear (K->H->N): two MX matmuls, host-carried intermediate")
def _mlp2(*, m: int = 64, k: int = 64, h: int = 64, n: int = 64, seed: int = 0, **_) -> KernelSpec:
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(k, h, bias=False), nn.Linear(h, n, bias=False))
    x = torch.randn(m, k)
    return from_module(net, x, name="mlp2")


@register("mlp3", "three chained nn.Linear (K->H->H->N)")
def _mlp3(*, m: int = 64, k: int = 64, h: int = 64, n: int = 64, seed: int = 0, **_) -> KernelSpec:
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(k, h, bias=False),
                        nn.Linear(h, h, bias=False),
                        nn.Linear(h, n, bias=False))
    x = torch.randn(m, k)
    return from_module(net, x, name="mlp3")


@register("attention", "single-head attention: 6 mesh matmuls + host softmax (QK^T, PV)")
def _attention(*, m: int = 64, k: int = 64, h: int = 64, seed: int = 0, **_) -> KernelSpec:
    """Q/K/V projections, `S = Q@K^T`, host softmax, `O = P@V`, output projection.

    Two things a straight chain cannot express and this needs: operands that reference earlier
    stages (`S` contracts Q with K^T; `O` contracts P with V), and a host stage — softmax is a row
    max, a row sum and a divide, and the mesh has no reduction hardware.

    `m` is the sequence length, `k` d_model, `h` d_head.
    """
    import numpy as np

    from .spec import HostStage, Stage

    torch.manual_seed(seed)
    wq, wk, wv = (nn.Linear(k, h, bias=False) for _ in range(3))
    wo = nn.Linear(h, k, bias=False)
    x = torch.randn(m, k)
    scale = 1.0 / float(np.sqrt(h))

    def softmax_scaled(s: "np.ndarray") -> "np.ndarray":
        z = s * scale
        z = z - z.max(axis=-1, keepdims=True)          # max-subtracted for stability
        e = np.exp(z)
        return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

    def t(layer: nn.Linear) -> torch.Tensor:
        return layer.weight.detach().T.contiguous().float()

    return KernelSpec(name="attention", x=x.detach().float(), stages=[
        Stage("Q", weight=t(wq), lhs="x"),
        Stage("K", weight=t(wk), lhs="x"),
        Stage("V", weight=t(wv), lhs="x"),
        Stage("S", lhs="Q", rhs="K.T"),
        HostStage("P", fn=softmax_scaled, src="S", note="softmax(S/sqrt(d)) — no reduction on mesh"),
        Stage("O", lhs="P", rhs="V"),
        Stage("Y", weight=t(wo), lhs="O"),
    ])
