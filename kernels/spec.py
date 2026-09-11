"""What we hand the pipeline: a kernel — a small dataflow graph of stages.

The mesh unit is a ``Stage``: one weight-stationary MX matmul, which is exactly what the backend
lowers (``mxgemm_emit._plan`` accepts one RES_PACK + one matmul + one COMMIT and raises otherwise).
A kernel is an ORDERED LIST of stages, each its own command buffer, with values carried between
them.

Two things beyond a straight chain, both needed by attention:

* **operands can reference earlier stages**, not just resident weights — ``S = Q @ K^T`` and
  ``O = P @ V`` both contract two computed values. An operand is a name: ``"x"`` (the model input),
  a stage name, or either with a ``".T"`` suffix.
* **:class:`HostStage` runs on the Rocket host, not the mesh** — softmax is a row max, a row sum and
  a divide, and there is no reduction hardware. The target contract declares one compute unit,
  ``mx_systolic_mesh``, ``ops: [matmul]``; `merlin_iface` has no softmax op either. Marking these
  explicitly is what lets the report say which cycles were the accelerator's and which were not.

Deliberately declarative rather than a graph tracer: ``from_module`` reads an ``nn.Linear`` /
``nn.Sequential`` and produces stages. Anything it cannot express raises, instead of silently
lowering something different from what PyTorch would compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np
import torch
from torch import nn

#: The model input's operand name.
INPUT = "x"


def _split_ref(ref: str) -> tuple[str, bool]:
    """``"K.T"`` -> ``("K", True)``."""
    return (ref[:-2], True) if ref.endswith(".T") else (ref, False)


@dataclass
class Stage:
    """One MX matmul on the mesh: ``lhs[M][K] @ rhs[K][N]``.

    Exactly one of ``weight`` / ``rhs`` is set. ``weight`` is the common case (a resident weight,
    [K][N] fp32); ``rhs`` names an earlier stage when the B operand is itself computed.
    ``lhs`` defaults to the previous stage's output, so a straight chain needs neither.
    """

    name: str
    weight: torch.Tensor | None = None
    lhs: str | None = None
    rhs: str | None = None

    def __post_init__(self) -> None:
        if (self.weight is None) == (self.rhs is None):
            raise ValueError(f"stage {self.name!r}: set exactly one of weight= or rhs=")

    @property
    def on_mesh(self) -> bool:
        return True


@dataclass
class HostStage:
    """A stage the mesh cannot do, run on the scalar core in fp32.

    DECLARATIVE, not a closure: ``op`` names an entry in :data:`app.mxhost.OPS`, which carries both
    a Python twin (for the reference) and the ``mx_host.h`` function to call (for the device). A
    closure can be run but not compiled, so a kernel built from closures can only execute on the
    host BETWEEN ELFs -- exactly what merlin_glue_port_plan.md D4 abolishes.

    ``fn`` is still accepted, and still works, for a host op that has no C twin yet. Such a kernel
    is confined to the per-stage path and cannot be fused; ``emittable`` says which case a stage is.
    """

    name: str
    op: str | None = None
    params: dict = field(default_factory=dict)
    fn: Callable[[np.ndarray], np.ndarray] | None = None
    src: "str | tuple[str, ...] | None" = None
    note: str = ""

    @property
    def srcs(self) -> tuple:
        """The value names this stage consumes, always as a tuple."""
        if self.src is None:
            return ()
        return (self.src,) if isinstance(self.src, str) else tuple(self.src)

    def __post_init__(self) -> None:
        if (self.op is None) == (self.fn is None):
            raise ValueError(
                f"host stage {self.name!r}: set exactly one of op= (emittable) or fn= "
                "(python-only, forces the per-stage path)")
        if self.op is not None:
            from app import mxhost
            o = mxhost.get(self.op, **self.params)   # validates the name and the params, early
            if self.src is not None and len(self.srcs) != o.arity:
                raise ValueError(
                    f"host stage {self.name!r}: op {self.op!r} takes {o.arity} input(s), "
                    f"got {len(self.srcs)} ({self.srcs})")

    @property
    def on_mesh(self) -> bool:
        return False

    @property
    def emittable(self) -> bool:
        """Can this stage be emitted as C into the fused driver?"""
        return self.op is not None

    def run(self, *xs: np.ndarray) -> np.ndarray:
        """Evaluate on the host, for the reference and the golden."""
        if self.op is None:
            return self.fn(*xs)
        from app import mxhost
        return mxhost.get(self.op, **self.params).fn(*xs, **self.params)


AnyStage = Union[Stage, HostStage]


@dataclass
class KernelSpec:
    """A kernel: an input, a list of stages, and its FP32 reference."""

    name: str
    x: torch.Tensor                       # [M][K] fp32
    stages: list[AnyStage] = field(default_factory=list)

    @property
    def m(self) -> int:
        return int(self.x.shape[0])

    @property
    def mesh_stages(self) -> list[Stage]:
        return [s for s in self.stages if s.on_mesh]

    @property
    def is_chain(self) -> bool:
        """True when this is a straight matmul chain — every stage a matmul against a resident
        weight, consuming the previous output. Only then can intermediates stay in the
        requantizer's codes+scales form; anything else round-trips through float on the host."""
        return all(isinstance(s, Stage) and s.weight is not None and s.rhs is None
                   and (s.lhs is None or s.lhs == (INPUT if i == 0 else self.stages[i - 1].name))
                   for i, s in enumerate(self.stages))

    # --- shape resolution ----------------------------------------------------------------------
    def shapes(self) -> dict[str, tuple[int, int]]:
        """Output shape of every stage (and ``x``), by walking the graph in order."""
        out: dict[str, tuple[int, int]] = {INPUT: tuple(int(v) for v in self.x.shape)}
        prev = INPUT
        for st in self.stages:
            if isinstance(st, HostStage):
                # A multi-input host op is elementwise, so its shape is its first
                # source's (swiglu's gate and up are the same shape by construction).
                out[st.name] = out[_split_ref((st.srcs or (prev,))[0])[0]]
            else:
                lhs, lt = _split_ref(st.lhs or prev)
                a = out[lhs][::-1] if lt else out[lhs]
                if st.weight is not None:
                    b = tuple(int(v) for v in st.weight.shape)
                else:
                    rhs, rt = _split_ref(st.rhs)
                    b = out[rhs][::-1] if rt else out[rhs]
                out[st.name] = (a[0], b[1])
            prev = st.name
        return out

    def stage_mnk(self) -> dict[str, tuple[int, int, int]]:
        """``(M, K, N)`` for each MESH stage."""
        sh = self.shapes()
        mnk: dict[str, tuple[int, int, int]] = {}
        prev = INPUT
        for st in self.stages:
            if isinstance(st, Stage):
                lhs, lt = _split_ref(st.lhs or prev)
                a = sh[lhs][::-1] if lt else sh[lhs]
                b = (tuple(int(v) for v in st.weight.shape) if st.weight is not None
                     else (sh[_split_ref(st.rhs)[0]][::-1] if _split_ref(st.rhs)[1]
                           else sh[_split_ref(st.rhs)[0]]))
                mnk[st.name] = (a[0], a[1], b[1])
            prev = st.name
        return mnk

    # --- reference -----------------------------------------------------------------------------
    def reference(self) -> torch.Tensor:
        """The FP32 answer for the WHOLE graph — no quantization anywhere. Host stages run here
        exactly as they do on device, so the comparison isolates the mesh's format cost."""
        vals: dict[str, np.ndarray] = {INPUT: self.x.numpy().astype(np.float32)}
        prev = INPUT
        for st in self.stages:
            if isinstance(st, HostStage):
                vals[st.name] = st.run(*[vals[_split_ref(r)[0]] for r in (st.srcs or (prev,))])
            else:
                lhs, lt = _split_ref(st.lhs or prev)
                a = vals[lhs].T if lt else vals[lhs]
                if st.weight is not None:
                    b = st.weight.numpy().astype(np.float32)
                else:
                    rhs, rt = _split_ref(st.rhs)
                    b = vals[rhs].T if rt else vals[rhs]
                vals[st.name] = a @ b
            prev = st.name
        return torch.from_numpy(np.ascontiguousarray(vals[prev]))

    # --- reporting / legality ------------------------------------------------------------------
    def describe(self) -> str:
        mnk = self.stage_mnk()
        n_host = len(self.stages) - len(self.mesh_stages)
        chain = " -> ".join(f"{s.name}[{mnk[s.name][0]}x{mnk[s.name][2]}x{mnk[s.name][1]}]"
                            if isinstance(s, Stage) else f"{s.name}(host)"
                            for s in self.stages)
        return (f"{self.name}: [{self.m}][{self.x.shape[1]}]  {chain}"
                f"   ({len(self.mesh_stages)} mesh, {n_host} host)")

    def validate(self, dim: int = 16, block: int = 32) -> list[str]:
        """Shape legality, checked BEFORE any build so failures are cheap and named.

        Mirrors ``mxgemm_emit._validate``; duplicated here only to fail early with a message naming
        the stage, not to replace it (the backend still enforces).
        """
        errs: list[str] = []
        try:
            mnk = self.stage_mnk()
        except KeyError as exc:
            return [f"unresolved operand reference {exc}"]
        requant_chain = self.is_chain
        mesh = self.mesh_stages
        for i, st in enumerate(mesh):
            m, k, n = mnk[st.name]
            if m % dim:
                errs.append(f"stage {st.name}: M={m} is not a multiple of the PE tile ({dim})")
            if k % block:
                errs.append(f"stage {st.name}: K={k} is not a multiple of the block-scale "
                            f"group ({block})")
            if n % dim:
                errs.append(f"stage {st.name}: N={n} is not a multiple of the PE tile ({dim})")
            # A non-final stage in a REQUANT chain commits through the requantizer, which emits one
            # E8M0 code per 32 output columns. Graphs that carry values as float on the host do not
            # take that path, so the stricter rule does not apply to them.
            if requant_chain and i < len(mesh) - 1 and n % block:
                errs.append(f"stage {st.name}: N={n} must be a multiple of {block} because it "
                            "feeds a following stage through the requantizer")
        return errs


# --- Building a spec from PyTorch -------------------------------------------------------------

def from_module(module: nn.Module, x: torch.Tensor, *, name: str = "module") -> KernelSpec:
    """Turn an ``nn.Linear`` or a ``nn.Sequential`` of them into a KernelSpec.

    ``nn.Linear`` stores weight as [out_features][in_features] = [N][K] and computes ``x @ Wᵀ``,
    so each stage's B operand is ``W.T`` -> [K][N].

    Raises on anything not expressible on this datapath today rather than quietly approximating it:

    * ``bias`` — would need a COMMIT epilogue, which the backend explicitly refuses
      (*"the E8M0 requant IS this datapath's scaling"*).
    * activations (ReLU, GELU, ...) — same reason. They could be added as a
      :class:`HostStage`, but that credits the host with work, so it is not silently done.
    """
    layers = list(module) if isinstance(module, nn.Sequential) else [module]
    stages: list[AnyStage] = []
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
