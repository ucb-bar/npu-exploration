"""MXQuant's block-quantizer API, computed by mxq.

``app/mxq_golden.py`` (the operand encoder both the compiler and the mxquant model use) was written
against MXQuant's ``end_to_end_linear/mx_block_quant.py``: ``quantize_mx_block32(V, fmt, axis,
round_mode) -> (P, X)`` and ``_broadcast_scales(X, shape, axis)``. This module provides the same three
names with the same shapes and numerics, so that file's call sites do not change and the MXQuant
clone is no longer needed to compile a kernel.

Numerics: ``mxq.block.mxgemmini.quantize`` with round-to-nearest-even and the block max floored at
FLT_EPSILON = 2^-23, which is what MXQuant's ``_po2`` does and what the MX-Gemmini requantizer does.
Proved bit-identical to MXQuant on every format, including exact ties, element subnormals, all-zero
and tiny blocks: ``tests/selftest_block.py``.

Shapes (MXQuant's): ``P`` like ``V``; ``X`` is ``[R][ceil(C/32)]`` for ``axis="row"`` (blocks run along
the columns of each row) and ``[ceil(R/32)][C]`` for ``axis="col"``.
"""
from __future__ import annotations

from typing import Literal, NamedTuple

import torch

import models  # noqa: F401  -- puts the mxq submodule on sys.path
from mxq import scale_factor

BLOCK = 32
#: MXQuant `_po2`: torch.finfo(float32).eps; the MX-Gemmini requantizer: max(amax, FLT_EPSILON).
SCALE_FLOOR = scale_factor.HARDWARE_FLOOR    # 2^-23
#: MXQuant's rounding vocabulary -> mxq's. "floor" has no hardware meaning and is refused.
_ROUNDING = {"even": "rne", "nearest": "ties_away"}


class Out(NamedTuple):
    P: torch.Tensor
    X: torch.Tensor


def quantize_mx_block32(V, fmt: str, axis: Literal["row", "col"] = "row", round_mode: str = "even") -> Out:
    if not models.paths():
        raise ImportError(models.mxq_missing())
    from mxq import block
    if round_mode not in _ROUNDING:
        raise ValueError(f"round_mode must be one of {sorted(_ROUNDING)}, got {round_mode!r}")
    if axis not in ("row", "col"):
        raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")
    V = torch.as_tensor(V).to(torch.float32)
    P, X = block.mxgemmini.quantize(V, fmt, axis=1 if axis == "row" else 0, block_size=BLOCK,
                                    rounding_mode=_ROUNDING[round_mode], scale_floor=SCALE_FLOOR)
    return Out(P, X)


def _broadcast_scales(X: torch.Tensor, shape: tuple[int, int], axis: str) -> torch.Tensor:
    """Each block scale repeated over its 32 elements, cut to ``shape``."""
    R, C = shape
    if axis == "row":
        return torch.repeat_interleave(X, BLOCK, dim=1)[:, :C]
    return torch.repeat_interleave(X, BLOCK, dim=0)[:R, :]
