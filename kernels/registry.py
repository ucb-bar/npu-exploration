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


def _mlp(depth: int, *, m: int, k: int, h: int, n: int, seed: int, name: str) -> KernelSpec:
    """A depth-``depth`` MLP chain ``K -> H -> ... -> H -> N``. No bias, no activation.

    ONE builder for every depth, so the RNG draw order is identical across them — layers in order,
    then ``x``. That is what keeps mlp2/mlp3 numerically identical to every run recorded before the
    deeper chains existed, and what makes a depth sweep a controlled comparison rather than a set of
    unrelated models.

    Every depth is a straight chain, so the pipeline fuses it: one command buffer, one ELF, one spike
    run, with each intermediate staying on device as the next stage's A operand.
    """
    if depth < 2:
        raise ValueError(f"{name}: depth {depth} is not a chain; use the 'linear' kernel")
    torch.manual_seed(seed)
    widths = [k] + [h] * (depth - 1) + [n]
    net = nn.Sequential(*(nn.Linear(widths[i], widths[i + 1], bias=False) for i in range(depth)))
    x = torch.randn(m, k)
    return from_module(net, x, name=name)


@register("mlp2", "two chained nn.Linear (K->H->N): 2 MX matmuls fused into one ELF")
def _mlp2(*, m: int = 64, k: int = 64, h: int = 64, n: int = 64, seed: int = 0, **_) -> KernelSpec:
    return _mlp(2, m=m, k=k, h=h, n=n, seed=seed, name="mlp2")


@register("mlp3", "three chained nn.Linear (K->H->H->N)")
def _mlp3(*, m: int = 64, k: int = 64, h: int = 64, n: int = 64, seed: int = 0, **_) -> KernelSpec:
    return _mlp(3, m=m, k=k, h=h, n=n, seed=seed, name="mlp3")


# Deeper chains: the fused path's real subject. Each added stage is another requantizer -> mesh hop,
# so these are what show where the `weight` seam's accuracy actually degrades and whether the MX
# shared-memory allocator holds up (each stage needs a DISJOINT region: M*N/16 rows of 16384).
@register("mlp4", "four chained nn.Linear: 4 MX matmuls, 3 seams, one ELF")
def _mlp4(*, m: int = 64, k: int = 64, h: int = 64, n: int = 64, seed: int = 0, **_) -> KernelSpec:
    return _mlp(4, m=m, k=k, h=h, n=n, seed=seed, name="mlp4")


@register("mlp6", "six chained nn.Linear: 6 MX matmuls, 5 seams, one ELF")
def _mlp6(*, m: int = 64, k: int = 64, h: int = 64, n: int = 64, seed: int = 0, **_) -> KernelSpec:
    return _mlp(6, m=m, k=k, h=h, n=n, seed=seed, name="mlp6")


@register("mlp8", "eight chained nn.Linear: 8 MX matmuls, 7 seams, one ELF")
def _mlp8(*, m: int = 64, k: int = 64, h: int = 64, n: int = 64, seed: int = 0, **_) -> KernelSpec:
    return _mlp(8, m=m, k=k, h=h, n=n, seed=seed, name="mlp8")


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

    def t(layer: nn.Linear) -> torch.Tensor:
        return layer.weight.detach().T.contiguous().float()

    return KernelSpec(name="attention", x=x.detach().float(), stages=[
        Stage("Q", weight=t(wq), lhs="x"),
        Stage("K", weight=t(wk), lhs="x"),
        Stage("V", weight=t(wv), lhs="x"),
        Stage("S", lhs="Q", rhs="K.T"),
        # Declared, not closed over: `op` names both the Python twin and the mx_host.h function,
        # which is what lets this stage be emitted into the fused ELF instead of run between ELFs.
        HostStage("P", op="softmax", params={"scale": scale}, src="S",
                  note="softmax(S/sqrt(d)) — no reduction on mesh"),
        Stage("O", lhs="P", rhs="V"),
        Stage("Y", weight=t(wo), lhs="O"),
    ])


# --- real data: one TinyLlama decoder layer ------------------------------------------------------
#
# Captured by `app/capture_llama_layer.py` from a real forward pass: the residual stream, both
# RMSNorm weights, and the layer's actual projection weights, all index-consistent.
#
# `d_model = 2048` is kept FULL, so RMSNorm and every projection INPUT is exact; the slice is on the
# output side (NF of 5632 FFN neurons, 1 of 32 heads). So gate/up are exact real llama values and
# down is an honest partial sum over those neurons -- and the fp32 reference is truncated the same
# way, so it grades what the device actually computes. See planning/llama_layer_hw_plan.md D1.

def _capture(path=None):
    """Load the captured layer, or explain how to make one."""
    import numpy as np
    from pathlib import Path

    d = Path(__file__).resolve().parent.parent / "out" / "layer_capture"
    fs = sorted(d.glob("*.npz")) if d.is_dir() else []
    if path is None and not fs:
        raise FileNotFoundError(
            f"no captured llama layer in {d}. Make one with:\n"
            "    .venv/bin/python -m app.capture_llama_layer")
    return np.load(Path(path) if path else fs[-1], allow_pickle=False)


@register("llama_mlp", "a real TinyLlama MLP: RMSNorm -> gate/up -> SwiGLU -> down (one ELF)")
def _llama_mlp(*, capture=None, **_) -> KernelSpec:
    import numpy as np

    from .spec import HostStage, KernelSpec, Stage

    z = _capture(capture)
    eps = float(z["meta_rms_eps"])
    h_mid = torch.from_numpy(z["h_mid"].astype(np.float32))       # [M][D], the residual stream
    stages = [
        HostStage("Xn", op="rmsnorm", src="x",
                  params={"weight": z["w_post_ln"].astype(np.float32), "eps": eps},
                  note="post-attention RMSNorm, over the FULL d_model"),
        Stage("G", weight=torch.from_numpy(z["Wg"].astype(np.float32)), lhs="Xn"),
        Stage("U", weight=torch.from_numpy(z["Wu"].astype(np.float32)), lhs="Xn"),
        HostStage("H", op="swiglu", src=("G", "U"), note="silu(gate) * up -- no hardware for it"),
        Stage("Y", weight=torch.from_numpy(z["Wd"].astype(np.float32)), lhs="H"),
    ]
    return KernelSpec(name="llama_mlp", x=h_mid, stages=stages)


@register("llama_attention",
          "a real TinyLlama attention head: RMSNorm -> Q/K/V -> RoPE -> S -> softmax -> O -> Wo")
def _llama_attention(*, capture=None, **_) -> KernelSpec:
    import numpy as np

    from .spec import HostStage, KernelSpec, Stage

    z = _capture(capture)
    eps = float(z["meta_rms_eps"])
    head_dim = int(z["meta_head_dim"])
    cos, sin = z["rope_cos"].astype(np.float32), z["rope_sin"].astype(np.float32)
    h_pre = torch.from_numpy(z["h_pre"].astype(np.float32))       # the residual stream in

    def W(k):
        return torch.from_numpy(z[k].astype(np.float32))

    stages = [
        HostStage("Xn", op="rmsnorm", src="x",
                  params={"weight": z["w_in_ln"].astype(np.float32), "eps": eps},
                  note="input RMSNorm, over the FULL d_model"),
        Stage("Q", weight=W("Wq"), lhs="Xn"),
        Stage("K", weight=W("Wk"), lhs="Xn"),
        Stage("V", weight=W("Wv"), lhs="Xn"),
        HostStage("Qr", op="rope", src="Q", params={"cos": cos, "sin": sin}),
        HostStage("Kr", op="rope", src="K", params={"cos": cos, "sin": sin}),
        # Kr.T is a real byte transpose on the scalar core: the MX loop path ignores B_transpose.
        Stage("S", lhs="Qr", rhs="Kr.T"),
        HostStage("P", op="softmax", src="S",
                  params={"scale": 1.0 / float(np.sqrt(head_dim)), "causal": True},
                  note="causal mask + softmax -- no reduction hardware"),
        Stage("O", lhs="P", rhs="V"),
        Stage("Y", weight=W("Wo"), lhs="O"),
    ]
    return KernelSpec(name="llama_attention", x=h_pre, stages=stages)
