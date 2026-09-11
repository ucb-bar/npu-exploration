"""The datapath's arithmetic primitives — the five `rtl_exact` needs to be bit-identical.

A thin re-export of :mod:`app.mxmesh.fp8`, which holds the extracted mesh model. Kept as its own
name because these five are a meaningful unit — the PE's truncating product, the per-lane RNE
quantizer, the exact add, and the bf16 pair used for cross-tile accumulation — and because
`rtl_exact` imports exactly them. Re-exporting rather than re-extracting keeps ONE copy of the
source in this repo.
"""
from __future__ import annotations

from .mxmesh.fp8 import (  # noqa: F401
    bf16_accum_add,
    fp_add_exact,
    fp_quantize_rne,
    mx_product_quantize_trunc,
    q_bf16_rne,
)

__all__ = ["mx_product_quantize_trunc", "fp_quantize_rne", "fp_add_exact", "q_bf16_rne",
           "bf16_accum_add"]
