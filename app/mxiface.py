"""Emit ``merlin_iface`` interface MLIR for an MX matmul.

This is the front end's output and the compiler's input. ``merlin_iface`` is merlin's **frozen,
versioned contract grammar** (`merlin/contract/interface_grammar.md`) — the documented handoff
format between a producer and an out-of-tree target-backend package:

    "the frozen, versioned input format the experiment ABI hands to an out-of-tree target-backend
     package. A package's job is to consume an *.interface.mlir file written in this grammar and
     produce (a) a command_buffer.json and (b) lowered LLVM/RoCC"

It is deliberately decoupled from xDSL — producers emit plain text, consumers parse plain text — so
this module is a text emitter and `merlin.targetgen.contract.interface_emit.parse_interface_mlir`
is the reader. Writing the MLIR is what makes the capsules in
`merlin/contract/capsules/mx_gemmini/` and our own front end the *same* kind of input.
"""
from __future__ import annotations

from dataclasses import dataclass

#: merlin_iface spells operand types with MLIR's builtin fp8 names, as the shipped MX capsules do
#: (`tensor<32x32xf8E4M3FN>`). Block scaling is a property of the TARGET, not of the element type,
#: so it does not appear here — the target contract declares it.
#: Our format key -> the element type spelling used in the interface module. Sourced from the one
#: format table so the two cannot drift; formats the frozen grammar has no builtin name for (the LUT
#: sub-formats) fall back to their own key, which merlin's reader passes through as an opaque token.
def _elem_types() -> dict[str, str]:
    # This module is imported BOTH as `app.mxiface` (by the pipeline) and as a top-level `mxiface`
    # (by the backend, which puts app/ on sys.path). Support both rather than forcing one, since
    # picking a side would break whichever caller is not updated.
    try:
        from . import mxformats
    except ImportError:                     # imported flat, with app/ on sys.path
        import mxformats
    return {f.name: (f.mlir or f.name) for f in mxformats.FORMATS.values()}


MX_ELEM_TYPE = _elem_types()


@dataclass(frozen=True)
class MatmulStage:
    """One matmul in a chain: ``lhs[m][k] @ weight[k][n] -> out[m][n]``.

    ``lhs`` is a NAME, and for stage *i* > 0 it is the previous stage's ``out`` — that is the whole
    of how a chain is expressed in the grammar. ``out_dtype`` is ``"bf16"`` for a stage read back to
    the host, or an MX element type (``"f8E4M3FN"``) for one whose requantized output the next stage
    consumes on device.

    A chained stage carries NO extra attribute. It used to declare
    ``mx_gemmini.chain_code_shift``, a target-namespaced commit attribute asking the seam to shift
    every code — compensation for a requantizer that filled the element format's range. The
    hardware stopped doing that (``chain_seam_hw_notes.md`` §8) and the seam itself is now empty
    (``merlin_glue_port_plan.md`` Step 3), so the attribute is gone. That is worth noting because
    it also removes this target's one extension to frozen grammar v0.1: every module emitted here
    is now plain ``merlin_iface``, with no namespaced key in it.
    """

    m: int
    k: int
    n: int
    weight: str
    out: str
    lhs: str
    out_dtype: str = "bf16"


def chain_interface_mlir(stages: list[MatmulStage], *,
                         operand_fmt: str = "fp8_e4m3",
                         acc_dtype: str = "bf16",
                         target: str = "mx_gemmini_rocket") -> str:
    """A CHAIN of weight-stationary matmuls as one merlin_iface module.

    Nothing here extends the frozen v0.1 grammar: a chain is N repetitions of the same four ops, with
    stage *i+1*'s ``matmul`` taking stage *i*'s ``commit`` result as its lhs. Only leaf tensors are
    declared — an intermediate is named by its COMMIT, exactly as `interface_grammar.md` specifies
    ("committed outputs are named by the COMMIT op"), so it never appears in the tensor table and its
    shape is derived by the consumer.

    A single-stage chain emits byte-for-byte what :func:`matmul_interface_mlir` always emitted, so
    the one-matmul path and the shipped MX capsules are unaffected.
    """
    if operand_fmt not in MX_ELEM_TYPE:
        raise ValueError(f"operand_fmt {operand_fmt!r} not in {sorted(MX_ELEM_TYPE)}")
    if not stages:
        raise ValueError("a chain needs at least one stage")
    et = MX_ELEM_TYPE[operand_fmt]

    # Leaf tensors only: every weight, then whatever lhs names are not produced by a commit. The
    # input is declared last so a 1-stage chain keeps the historical "weight, then input" order.
    produced = {st.out for st in stages}
    lines = [
        f'module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "{target}", '
        f'merlin_iface.abi_version = "0.1"}} {{',
    ]
    for st in stages:
        lines.append(f'  %{st.weight} = merlin_iface.tensor '
                     f'{{name = "{st.weight}", role = "weight"}} : tensor<{st.k}x{st.n}x{et}>')
    for st in stages:
        if st.lhs not in produced:
            lines.append(f'  %{st.lhs} = merlin_iface.tensor '
                         f'{{name = "{st.lhs}", role = "input"}} : tensor<{st.m}x{st.k}x{et}>')

    for i, st in enumerate(stages):
        # An intermediate lhs carries its commit's output type, not the operand type: the
        # requantizer's write-back IS the next stage's operand encoding.
        lhs_t = (f'tensor<{st.m}x{st.k}x{stages[i - 1].out_dtype}>' if st.lhs in produced
                 else f'tensor<{st.m}x{st.k}x{et}>')
        lines += [
            f'  %{st.weight}_res = merlin_iface.resident_pack %{st.weight} '
            f'{{layout = "packed_rhs"}} : (tensor<{st.k}x{st.n}x{et}>) -> !merlin_iface.resident',
            # The accumulator is ALWAYS bf16 (the mesh's accumulate type); the commit is what
            # requantizes, so out_dtype may be an MX format for a chained matmul.
            f'  %acc{i} = merlin_iface.matmul %{st.lhs}, %{st.weight}_res '
            f': ({lhs_t}, !merlin_iface.resident) -> !merlin_iface.acc<{acc_dtype}>',
            f'  %{st.out} = merlin_iface.commit %acc{i} {{name = "{st.out}", epilogue = [], '
            f'output_dtype = "{st.out_dtype}"'
            + f'}} : (!merlin_iface.acc<{acc_dtype}>) -> tensor<{st.m}x{st.n}x{st.out_dtype}>',
            # Evict where the weight actually dies — right after its own commit — so a chain does
            # not claim N residents are live at once.
            f'  merlin_iface.evict %{st.weight}_res : (!merlin_iface.resident) -> ()',
        ]
    return "\n".join(lines + ["}", ""])


def matmul_interface_mlir(m: int, n: int, k: int, *,
                          operand_fmt: str = "fp8_e4m3",
                          out_dtype: str = "bf16",
                          acc_dtype: str = "bf16",
                          target: str = "mx_gemmini_rocket",
                          lhs: str = "X", weight: str = "W", out: str = "Y0") -> str:
    """One weight-stationary matmul ``A[m][k] @ B[k][n]`` as a merlin_iface module.

    Byte-for-byte the shape the shipped MX capsules use, so a capsule and this front end are
    interchangeable inputs to the backend. A one-element :func:`chain_interface_mlir`.
    """
    return chain_interface_mlir(
        [MatmulStage(m=m, k=k, n=n, weight=weight, out=out, lhs=lhs, out_dtype=out_dtype)],
        operand_fmt=operand_fmt, acc_dtype=acc_dtype, target=target)


def to_command_buffer(interface_mlir: str,
                      mx_operands: dict | list[dict] | None = None) -> dict:
    """Lower interface MLIR to a command buffer using **merlin's** reference parser.

    ``mx_operands`` (raw MX codes + E8M0 block scales) is attached as a side-channel: the grammar
    names shapes and dtypes, never buffer contents, and the datapath consumes codes plus a separate
    scale stream that decoded values cannot reconstruct. Same channel merlin's own MX path uses.

    A LIST attaches one bundle per matmul, in command order. A chained stage supplies only its
    ``b_*`` keys: its A operand is the previous stage's requantizer output, which is produced on
    device and never travels through here.
    """
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    cb = parse_interface_mlir(interface_mlir)
    if mx_operands is not None:
        import numpy as np

        def _bundle(d: dict) -> dict:
            return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in d.items() if v is not None}

        cb["mx_operands"] = ([_bundle(d) for d in mx_operands]
                             if isinstance(mx_operands, list) else _bundle(mx_operands))
    return cb
