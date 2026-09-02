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

#: merlin_iface spells operand types with MLIR's builtin fp8 names, as the shipped MX capsules do
#: (`tensor<32x32xf8E4M3FN>`). Block scaling is a property of the TARGET, not of the element type,
#: so it does not appear here — the target contract declares it.
MX_ELEM_TYPE = {"fp8": "f8E4M3FN", "fp6": "f6E3M2FN", "fp4": "f4E2M1FN"}


def matmul_interface_mlir(m: int, n: int, k: int, *,
                          operand_fmt: str = "fp8",
                          out_dtype: str = "bf16",
                          acc_dtype: str = "bf16",
                          target: str = "mx_gemmini_rocket",
                          lhs: str = "X", weight: str = "W", out: str = "Y0") -> str:
    """One weight-stationary matmul ``A[m][k] @ B[k][n]`` as a merlin_iface module.

    Byte-for-byte the shape the shipped MX capsules use, so a capsule and this front end are
    interchangeable inputs to the backend.
    """
    if operand_fmt not in MX_ELEM_TYPE:
        raise ValueError(f"operand_fmt {operand_fmt!r} not in {sorted(MX_ELEM_TYPE)}")
    et = MX_ELEM_TYPE[operand_fmt]
    return "\n".join([
        f'module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "{target}", '
        f'merlin_iface.abi_version = "0.1"}} {{',
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} '
        f': tensor<{k}x{n}x{et}>',
        f'  %{lhs} = merlin_iface.tensor {{name = "{lhs}", role = "input"}} '
        f': tensor<{m}x{k}x{et}>',
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
        f': (tensor<{k}x{n}x{et}>) -> !merlin_iface.resident',
        # The accumulator is ALWAYS bf16 (the mesh's accumulate type); the commit is what
        # requantizes, so out_dtype may be an MX format for a chained matmul.
        f'  %acc0 = merlin_iface.matmul %{lhs}, %{weight}_res '
        f': (tensor<{m}x{k}x{et}>, !merlin_iface.resident) -> !merlin_iface.acc<{acc_dtype}>',
        f'  %{out} = merlin_iface.commit %acc0 {{name = "{out}", epilogue = [], '
        f'output_dtype = "{out_dtype}"}} '
        f': (!merlin_iface.acc<{acc_dtype}>) -> tensor<{m}x{n}x{out_dtype}>',
        f'  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()',
        "}",
        "",
    ])


def to_command_buffer(interface_mlir: str, mx_operands: dict | None = None) -> dict:
    """Lower interface MLIR to a command buffer using **merlin's** reference parser.

    ``mx_operands`` (raw MX codes + E8M0 block scales) is attached as a side-channel: the grammar
    names shapes and dtypes, never buffer contents, and the datapath consumes codes plus a separate
    scale stream that decoded values cannot reconstruct. Same channel merlin's own MX path uses.
    """
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    cb = parse_interface_mlir(interface_mlir)
    if mx_operands is not None:
        import numpy as np
        cb["mx_operands"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in mx_operands.items() if v is not None}
    return cb
