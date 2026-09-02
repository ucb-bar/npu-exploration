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
