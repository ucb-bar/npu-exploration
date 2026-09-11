"""A kernel graph as a flat step list, for kernels that are not a straight chain.

``merlin_iface`` expresses a CHAIN: N repetitions of pack/matmul/commit/evict where stage *i+1*'s
lhs is stage *i*'s output. Attention is not that, in three independent ways:

* ``Q``, ``K`` and ``V`` all take ``X`` as lhs — three live values, not one running value;
* ``S = Q @ Kᵀ`` and ``O = P @ V`` contract two COMPUTED values, neither a resident weight;
* a softmax sits in the middle, and the grammar has no op for it (v0.1 has five ops, and
  ``commit``'s epilogue is limited to ``bias_add | requant | acc_scale | relu``).

So the graph rides the command buffer as a side channel, next to ``mx_operands`` — the same shape of
decision, for the same reason: the generic contract carries what it can express, and what it cannot
travels alongside rather than being faked. Labelled as ours, not mistaken for merlin convention.

**Every edge here goes through host fp32 memory.** A mesh output is drained as bf16, converted, and
re-quantized before its next use. That is not a shortcut: for attention it is what the hardware
requires, because every seam except ``P@V -> O@Wo`` has a host op in it (RoPE, softmax) and those
values have to reach the scalar core anyway. ``llama_attention.c`` is built exactly this way, and
uses the resident seam only where no host op intervenes.

What this buys is D4: **one ELF, one run, and numpy never between two stages.** It does not buy
speed — ``llama_layer_hw_plan.md`` §8.3 measured the scalar glue at ~99.9% of cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: How a value is consumed by a mesh matmul. A operands block along K by rows, B operands by
#: columns, so the same value used on both sides needs quantizing twice, differently.
USE_A = "a"
USE_B = "b"
USE_BT = "b.T"


@dataclass
class MeshStep:
    """One matmul on the mesh: ``lhs @ rhs -> out``, all named."""

    name: str
    m: int
    k: int
    n: int
    lhs: str                    #: value name feeding the A side
    rhs: str                    #: value name feeding the B side
    rhs_transposed: bool        #: rhs is used as ``rhs.T`` (a host byte transpose, see below)
    out: str

    @property
    def kind(self) -> str:
        return "mesh"


@dataclass
class HostStep:
    """One op on the scalar core: ``op(src) -> out``, in fp32."""

    name: str
    op: str
    params: dict
    srcs: tuple            #: value names it consumes -- SwiGLU takes two, most take one
    out: str
    m: int
    n: int
    #: param name -> the baked fp32 const holding it (RMSNorm's weight, RoPE's tables). Separate
    #: from `params`, which carries only scalars: mixing them would make the op's own parameter
    #: validation reject the emitter's bookkeeping.
    const_names: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "host"


@dataclass
class Graph:
    """A kernel as an ordered step list plus the leaf operands it bakes."""

    steps: list = field(default_factory=list)
    #: value name -> (rows, cols) for every value the graph names, leaves included.
    shapes: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: value name -> how it is consumed (a set of USE_*). Decides which quantizations to emit.
    uses: dict[str, set] = field(default_factory=dict)
    #: leaf value name -> the fp32 tensor to bake AS A QUANTIZED MESH OPERAND.
    leaves: dict[str, np.ndarray] = field(default_factory=dict)
    #: name -> an fp32 array baked VERBATIM, for host ops. Two sources: an array-valued param (
    #: RMSNorm's weight, RoPE's cos/sin) and a leaf a host op consumes (the residual's input).
    #: These are never quantized -- the scalar core reads them as floats.
    consts: dict[str, np.ndarray] = field(default_factory=dict)
    #: the graph's final output value.
    result: str = ""

    def is_leaf(self, name: str) -> bool:
        return name in self.leaves


def from_spec(spec) -> Graph:
    """Lower a :class:`kernels.spec.KernelSpec` to a :class:`Graph`.

    The spec's stage list is already topologically ordered — a stage may only reference values
    produced before it — so no scheduling is needed, only naming. Refuses a host stage that has no
    C twin, rather than emitting a driver with a hole in it.
    """
    from kernels.spec import INPUT

    g = Graph()
    mnk = spec.stage_mnk()
    g.leaves[INPUT] = spec.x.numpy().astype(np.float32)
    g.shapes[INPUT] = tuple(spec.x.shape)

    def note_use(value: str, how: str) -> None:
        g.uses.setdefault(value, set()).add(how)

    prev = INPUT
    for st in spec.stages:
        if not st.on_mesh:
            if not st.emittable:
                raise ValueError(
                    f"host stage {st.name!r} carries a python closure (fn=), which cannot be "
                    "emitted as C. Declare it with op= (see app/mxhost.OPS) or accept the "
                    "per-stage path.")
            srcs = st.srcs or (prev,)
            r, c = g.shapes[srcs[0]]
            # An array-valued param is DATA the driver must carry, so it is split out of `params`
            # (which reaches the emitter as scalars) and baked as an fp32 const.
            scalars, arrays = {}, {}
            for pk, pv in st.params.items():
                if isinstance(pv, np.ndarray) or hasattr(pv, "numpy"):
                    arr = pv.numpy() if hasattr(pv, "numpy") else pv
                    arrays[pk] = np.ascontiguousarray(arr, dtype=np.float32)
                else:
                    scalars[pk] = pv
            const_names = {}
            for pk, arr in arrays.items():
                cname = f"{st.name}_{pk}"
                g.consts[cname] = arr
                const_names[pk] = cname
            # A leaf a host op reads needs an fp32 copy: it never went through the mesh, so no
            # `_f32` buffer exists for it.
            for srcname in srcs:
                if srcname in g.leaves and srcname not in g.consts:
                    g.consts[srcname] = np.ascontiguousarray(g.leaves[srcname], dtype=np.float32)
            g.steps.append(HostStep(name=st.name, op=st.op, params=scalars,
                                    srcs=tuple(srcs), out=st.name, m=r, n=c,
                                    const_names=const_names))
            g.shapes[st.name] = (r, c)
        else:
            m, k, n = mnk[st.name]
            lhs = st.lhs or prev
            if st.weight is not None:
                rhs, transposed = f"W_{st.name}", False
                g.leaves[rhs] = st.weight.numpy().astype(np.float32)
                g.shapes[rhs] = (k, n)
            else:
                rhs, transposed = (st.rhs[:-2], True) if st.rhs.endswith(".T") else (st.rhs, False)
            note_use(lhs, USE_A)
            note_use(rhs, USE_BT if transposed else USE_B)
            g.steps.append(MeshStep(name=st.name, m=m, k=k, n=n, lhs=lhs, rhs=rhs,
                                    rhs_transposed=transposed, out=st.name))
            g.shapes[st.name] = (m, n)
        prev = st.name

    g.result = spec.stages[-1].name
    return g


def operand_bundles(g: Graph, *, dtype: str = "fp8_e4m3") -> dict:
    """Quantize every LEAF, on whichever side(s) it is used. Computed values are quantized on device.

    A leaf used as both A and B (attention's ``X`` is only an A; a weight only a B) would appear
    twice with different blocking, which is why the key carries the side.
    """
    from .mxq_golden import quantize_operand

    out: dict[str, dict] = {}
    for name, V in g.leaves.items():
        # `uses` is populated only by MESH steps, so an empty entry means no matmul reads this
        # tensor -- the model input of a kernel that starts with a host op, for instance. Baking it
        # as an operand anyway would emit tens of KB of dead C.
        for how in sorted(g.uses.get(name, ())):
            side = USE_A if how == USE_A else USE_B
            src = V.T if how == USE_BT else V
            codes, scales, luts = quantize_operand(
                np.ascontiguousarray(src, dtype=np.float32), side=side, dtype=dtype)
            out[f"{name}:{how}"] = {"codes": codes, "scales": scales, "luts": luts}
    return out
