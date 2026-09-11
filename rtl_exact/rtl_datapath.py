"""Make MXQuant's simulated MX matmul BIT-IDENTICAL to the MxGemmini hardware.

MXQuant already models the narrow mesh datapath -- `prodacc_bundle/eval_complete.py` quantizes every
outer product and re-quantizes the running sum per lane from an `--acc-schedule`, which is
structurally what `gemmini.cc` and the RTL do. Three details differ, and all three must be matched
before the two agree:

  0. wire operands  shipped: re-quantizes A and B itself
                    hardware: consumes exactly the codes the compiler emitted -- which for a
                    CODEBOOK format cannot be re-derived, since the codebook is compile output
  1. product        shipped: float_quantize(rounding="nearest")
                    hardware: mantissa TRUNCATION, no exponent clamp at the product stage
  2. accumulate     shipped: fp32 add, then quantize the SUM to the lane
                    hardware: quantize BOTH addends to the lane (RNE), then add exactly
  3. cross-tile     shipped: fp32 accumulation of scaled tiles
                    hardware: round the scaled tile to bf16 and accumulate in bf16

With all three: 65536/65536 elements identical, max abs diff 0.0, on a real TinyLlama MLP.
See `mxgemmini_rtl.json` for the config and its provenance, and `verify_rtl_exact.py` for the gate.

THESE DO NOT DECOMPOSE. Each change alone LOWERS the error versus fp32 and two of the three make
the divergence from hardware WORSE; only all three together are exact.

Usage
-----
    import eval_complete                      # MXQuant's prodacc bundle
    from rtl_exact import rtl_datapath
    cfg = rtl_datapath.load_config()
    rtl_datapath.install(eval_complete)       # rebinds MXLinearSim._simulate_atw
    sim = eval_complete.MXLinearSim(layer, "MXFP8_E4M3", False,
                                    *cfg.product, cfg.acc_schedule, 0, 0, window=cfg.window)

The arithmetic primitives are IMPORTED from the hardware golden model (`fp8_matmul_model` in the
gemmini tree) rather than transcribed, so there is exactly one implementation of each and it cannot
drift. Point `MXGEMMINI_ROOT` at that tree if it is not at the default relative path; this module
raises rather than falling back to a lookalike.

Known cost: the golden's `fp_quantize_rne` (exp<8) and `fp_add_exact` are exact-dyadic SCALAR Python
loops, so an RTL-exact run is far slower than the shipped path -- fine for a layer, painful for a
full perplexity sweep. Vectorizing them (and checking the vectorized form elementwise against the
scalar one) is the obvious next step if this is used at model scale.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "mxgemmini_rtl.json"


@dataclass(frozen=True)
class RtlConfig:
    """The configuration under which MXQuant reproduces the hardware exactly."""
    mx_fmt: str
    product: Tuple[int, int]          #: (exp, man) of the PE product
    acc_schedule: List[Tuple[int, int]]
    window: int
    block: int
    raw: dict

    @property
    def acc_schedule_csv(self) -> Path:
        return HERE / self.raw["acc_schedule_csv"]


def load_config(path: Path | None = None) -> RtlConfig:
    raw = json.loads((path or CONFIG_PATH).read_text())
    p = raw["formats"]["product"]
    return RtlConfig(
        mx_fmt=raw["formats"]["operand"],
        product=(int(p["exp"]), int(p["man"])),
        acc_schedule=[(int(e), int(m)) for e, m in raw["acc_schedule"]],
        window=int(raw["mesh"]["window"]),
        block=int(raw["mesh"]["block"]),
        raw=raw,
    )


def _golden(cfg: RtlConfig | None = None):
    """The datapath's arithmetic primitives.

    These now live IN THIS REPO (``app/mxarith.py``), mechanically EXTRACTED from
    ``gemmini-rocc-tests/fp8_matmul_model.py`` rather than transcribed. That removes the last
    runtime dependency the graded path had on the reference tree.

    The "one implementation, cannot drift" property that the previous cross-tree import provided is
    preserved as a TEST instead: ``tests/selftest_mxarith.py`` re-runs the extraction, diffs it
    against the upstream source, and checks the two agree elementwise. A silent divergence fails
    there rather than in a kernel's numbers.
    """
    del cfg
    repo = HERE.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from app import mxarith
    return mxarith


# --- the three hardware behaviours ---------------------------------------------------------------

def product_quantize(x: torch.Tensor, exp: int, man: int, FM) -> torch.Tensor:
    """(1) The PE product: mantissa truncation, no exponent clamp, no subnormal grid."""
    return FM.mx_product_quantize_trunc(x, exp, man)


def accumulate(acc: torch.Tensor, prod: torch.Tensor, e: int, m: int, FM) -> torch.Tensor:
    """(2) Quantize BOTH addends to the lane's precision, then add exactly."""
    return FM.fp_add_exact(FM.fp_quantize_rne(acc, e, m), FM.fp_quantize_rne(prod, e, m), e, m)


def cross_tile_accumulate(C: torch.Tensor, tile: torch.Tensor, FM) -> torch.Tensor:
    """(3) Round the scaled tile to bf16 and accumulate in bf16 (`mx_smem` is bf16)."""
    return FM.bf16_accum_add(C, FM.q_bf16_rne(tile))


# --- installation ---------------------------------------------------------------------------------

def install(eval_complete_module, cfg: RtlConfig | None = None) -> None:
    """Rebind `MXLinearSim._simulate_atw` to the RTL-exact version.

    The body below is MXQuant's own loop with exactly three lines changed (the three behaviours
    above); everything else -- the block quantization, the scale map, the lane indexing, the LUT
    hook -- is called straight out of the module being patched. If upstream restructures
    `_simulate_atw`, re-sync this copy; `verify_rtl_exact.py` is what tells you it needs doing.
    """
    cfg = cfg or load_config()
    FM = _golden(cfg)
    EC = eval_complete_module
    BLOCK = EC.BLOCK

    @torch.no_grad()
    def _simulate_atw_rtl(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        K, M = A.shape
        _, N = B.shape
        device = A.device

        # (4) WIRE OPERANDS. When the caller has already produced the exact (P, X) the device was
        # given, use them and skip re-quantizing. This is what makes the CODEBOOK formats
        # reproducible: their operands are 4-bit indices into a 16-entry table that is COMPILE
        # OUTPUT, fitted to the data. Re-deriving it here could not match -- MXQuant's own k-means
        # seeds from `torch.multinomial`, i.e. randomly -- and re-quantizing to element codes
        # skips the codebook step the hardware performs, which measured ~12-15% off.
        #
        # For the direct formats (E4M3, E2M1) this changes nothing: their P is what
        # mx_block32_quantize produces anyway, verified bit-identical either way.
        wire = getattr(self, "_wire_operands", None)
        if wire is not None:
            P_A, X_A, P_B, X_B = (t.to(device) for t in wire)
        elif self.no_input_mx or self.mx_fmt == "FP32":
            P_A, P_B = A, B
            X_A = torch.ones((math.ceil(K / BLOCK), M), dtype=A.dtype, device=device)
            X_B = torch.ones((math.ceil(K / BLOCK), N), dtype=B.dtype, device=device)
        else:
            P_A, X_A = EC.mx_block32_quantize(A, self.mx_fmt, axis="col")
            P_B, X_B = EC.mx_block32_quantize(B, self.mx_fmt, axis="col")

        if wire is None and self.lut_weight and self.mx_fmt != "FP32" and not self.no_input_mx:
            P_B = EC.apply_lut_to_mx_weight(
                P_B, granularity=self.lut_granularity, num_signposts=self.lut_signposts,
                iters=self.lut_iters)

        C = torch.zeros((M, N), dtype=torch.float32, device=device)
        window = self.window

        for g in range(0, K, BLOCK):
            g_end = min(g + BLOCK, K)
            g_block = g // BLOCK
            sA, sB = X_A[g_block, :], X_B[g_block, :]
            scale_map = sA.unsqueeze(1) * sB.unsqueeze(0)

            for k_base in range(g, g_end, window):
                k_batch_end = min(k_base + window, g_end)
                S_red = torch.zeros((M, N), dtype=torch.float32, device=device)

                for k in range(k_base, k_batch_end):
                    outer = P_A[k, :].unsqueeze(1) * P_B[k, :].unsqueeze(0)
                    outer_q = product_quantize(outer, self.prod_e, self.prod_m, FM)   # (1)
                    lane = k % window
                    if self.acc_schedule is not None:
                        e_acc, m_acc = self.acc_schedule[lane]
                    else:
                        e_acc, m_acc = self.acc_fixed_e, self.acc_fixed_m
                    S_red = accumulate(S_red, outer_q, e_acc, m_acc, FM)              # (2)

                C = cross_tile_accumulate(C, S_red * scale_map, FM)                   # (3)

        return C

    EC.MXLinearSim._simulate_atw = _simulate_atw_rtl
    EC.MXLinearSim._rtl_exact = True


def is_installed(eval_complete_module) -> bool:
    return bool(getattr(eval_complete_module.MXLinearSim, "_rtl_exact", False))
