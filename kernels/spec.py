"""What we hand the pipeline: a kernel, or a chain of them.

The unit is a ``Stage`` — one weight-stationary MX matmul, which is exactly what
the backend lowers (``mxgemm_emit._plan`` accepts one RES_PACK + one matmul + one
COMMIT and raises otherwise). A model is therefore an ORDERED LIST of stages,
each its own command buffer, with the intermediate carried between them. That is
the same shape ``app/chain_2gemm/run_chain.py`` uses; we are generalizing it, not
inventing a second scheme.

Deliberately declarative rather than a graph tracer: ``from_module`` reads an
``nn.Linear`` / ``nn.Sequential`` and produces stages. Anything it cannot express
raises, instead of silently lowering something different from what PyTorch would
compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class Stage:
    """One MX matmul: ``[M][K] @ weight[K][N]``."""

    name: str
    weight: torch.Tensor          # [K][N], fp32 — the B operand as the device wants it

    @property
    def k(self) -> int:
        return int(self.weight.shape[0])

    @property
    def n(self) -> int:
        return int(self.weight.shape[1])


@dataclass
class KernelSpec:
    """A kernel (one stage) or a chain of kernels (several), plus its FP32 reference."""

    name: str
    x: torch.Tensor               # [M][K] fp32 input
    stages: list[Stage] = field(default_factory=list)

    @property
    def m(self) -> int:
        return int(self.x.shape[0])

    def reference(self) -> torch.Tensor:
        """The FP32 answer for the WHOLE chain — no quantization anywhere."""
        y = self.x
        for st in self.stages:
            y = y @ st.weight
        return y

    def describe(self) -> str:
        dims = " -> ".join([str(self.stages[0].k)] + [str(s.n) for s in self.stages])
        return f"{self.name}: [{self.m}][{self.stages[0].k}] through {dims}"

    def validate(self, dim: int = 16, block: int = 32) -> list[str]:
        """Shape legality, checked BEFORE any build so failures are cheap and named.

        Mirrors ``mxgemm_emit._validate``; duplicated here only to fail early with a
        message naming the stage, not to replace it (the backend still enforces).
        """
        errs = []
        if self.m % dim:
            errs.append(f"M={self.m} is not a multiple of the PE tile ({dim})")
        prev_n = None
        for i, st in enumerate(self.stages):
            if prev_n is not None and st.k != prev_n:
                errs.append(f"stage {i} ({st.name}): K={st.k} does not match "
                            f"previous stage's N={prev_n}")
            if st.k % block:
                errs.append(f"stage {i} ({st.name}): K={st.k} is not a multiple of the "
                            f"block-scale group ({block})")
            for label, v in (("K", st.k), ("N", st.n)):
                if v % dim:
                    errs.append(f"stage {i} ({st.name}): {label}={v} is not a multiple "
                                f"of the PE tile ({dim})")
            # Non-final stages commit through the REQUANTIZER (MX output), which emits one
            # E8M0 code per 32 output columns -- so their N has a stricter constraint than
            # the final stage's. mxgemm_emit._validate enforces the same rule.
            if i < len(self.stages) - 1 and st.n % block:
                errs.append(f"stage {i} ({st.name}): N={st.n} must be a multiple of "
                            f"{block} because it feeds a following stage through the "
                            "requantizer (one E8M0 code per 32 output columns)")
            prev_n = st.n
        return errs


# --- Building a spec from PyTorch -------------------------------------------------------------

def from_module(module: nn.Module, x: torch.Tensor, *, name: str = "module") -> KernelSpec:
    """Turn an ``nn.Linear`` or a ``nn.Sequential`` of them into a KernelSpec.

    ``nn.Linear`` stores weight as [out_features][in_features] = [N][K] and computes
    ``x @ Wᵀ``, so each stage's B operand is ``W.T`` -> [K][N].

    Raises on anything not expressible on this datapath today rather than quietly
    approximating it:

    * ``bias`` — would need a COMMIT epilogue, which the backend explicitly refuses
      (*"the E8M0 requant IS this datapath's scaling"*).
    * activations (ReLU, GELU, ...) — same reason. Chained stages already round-trip
      through the host, so a host-side activation is a plausible next step, but it
      would change what the device is being credited with, so it is not silently done.
    """
    layers = list(module) if isinstance(module, nn.Sequential) else [module]
    stages: list[Stage] = []
    for i, layer in enumerate(layers):
        if not isinstance(layer, nn.Linear):
            raise ValueError(
                f"{name}: layer {i} is {type(layer).__name__}; only nn.Linear is lowerable "
                "today (activations and bias need a COMMIT epilogue, which the backend refuses)")
        if layer.bias is not None:
            raise ValueError(f"{name}: layer {i} has a bias; build it with bias=False "
                             "(bias needs a COMMIT epilogue, unsupported on this datapath)")
        stages.append(Stage(name=f"L{i}", weight=layer.weight.detach().T.contiguous().float()))
    return KernelSpec(name=name, x=x.detach().float(), stages=stages)
