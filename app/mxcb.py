"""Build merlin command buffers for MX matmuls.

App layer: turns operands into the target-independent buffer the compiler backend consumes. The
backend never sees a tensor, a model, or a file — only this dict.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def matmul_cb(m: int, n: int, k: int, mx_operands: dict[str, Any], *,
              operand_dtype: str = "mxfp8", out_dtype: str = "bf16") -> dict[str, Any]:
    """One RES_PACK -> MATMUL_RESIDENT -> COMMIT -> EVICT buffer for ``A[m][k] @ B[k][n]``."""
    ops = {key: (val.tolist() if isinstance(val, np.ndarray) else val)
           for key, val in mx_operands.items() if val is not None}
    return {
        "abi_version": "0.1",
        "target": "mx_gemmini_rocket",
        "backend": "spike_mx_gemmini",
        "tensors": {
            "A0": {"dtype": operand_dtype, "shape": [m, k]},
            "W": {"dtype": operand_dtype, "shape": [k, n]},
            "Y0": {"dtype": out_dtype, "shape": [m, n]},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"}, "attributes": {}},
            {"opcode": "EVICT", "operands": {"handle": "W_res"}},
        ],
        "params": {},
        "resources": {"buffers": [], "handles": ["W_res", "acc0"]},
        "metrics_requested": ["cycles"],
        # MX side-channel: raw operand codes + E8M0 block scales. The decoded tensor table above
        # cannot carry these — see mxgemm_emit._mx_operands.
        "mx_operands": ops,
    }
