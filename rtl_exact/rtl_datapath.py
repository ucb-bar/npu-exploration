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

import os as _os

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


#: Peak device memory the batched window loop may use for its [W, M, N] working set. Lower it if a
#: run OOMs on a small GPU; it only changes the batch size, never the result.
RTL_BATCH_BYTES = 2 << 30


def _simulate_atw_rtl_serial(self, A, B, P_A, X_A, P_B, X_B, C, window, FM):
    """The original per-k form. Kept for shapes the batched path cannot cover -- a K that is not a
    whole number of windows -- so a ragged reduction still gets the right answer, slowly."""
    import torch
    K = P_A.shape[0]
    for g in range(0, K, BLOCK):
        g_end = min(g + BLOCK, K)
        scale_map = X_A[g // BLOCK, :].unsqueeze(1) * X_B[g // BLOCK, :].unsqueeze(0)
        for k_base in range(g, g_end, window):
            S_red = torch.zeros(C.shape, dtype=torch.float32, device=C.device)
            for k in range(k_base, min(k_base + window, g_end)):
                outer = P_A[k, :].unsqueeze(1) * P_B[k, :].unsqueeze(0)
                outer_q = product_quantize(outer, self.prod_e, self.prod_m, FM)
                lane = k % window
                e_acc, m_acc = (self.acc_schedule[lane] if self.acc_schedule is not None
                                else (self.acc_fixed_e, self.acc_fixed_m))
                S_red = accumulate(S_red, outer_q, e_acc, m_acc, FM)
            C = cross_tile_accumulate(C, S_red * scale_map, FM)
    return C


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

#: MxFPMul PROD_FLOOR: a product below 2^-16 is flushed to zero (mxq MXGEMMINI prod_floor).
PROD_FLOOR = -16


def product_quantize(x: torch.Tensor, exp: int, man: int, FM) -> torch.Tensor:
    """(1) The PE product: mantissa truncation, flushed below 2^PROD_FLOOR, no subnormal grid."""
    q = FM.mx_product_quantize_trunc(x, exp, man)
    return torch.where(q.abs() < 2.0 ** PROD_FLOOR, torch.zeros_like(q), q)


def accumulate(acc: torch.Tensor, prod: torch.Tensor, e: int, m: int, FM) -> torch.Tensor:
    """(2) Quantize BOTH addends to the lane's precision, then add exactly."""
    return FM.fp_add_exact(FM.fp_quantize_rne(acc, e, m), FM.fp_quantize_rne(prod, e, m), e, m)


def cross_tile_accumulate(C: torch.Tensor, tile: torch.Tensor, FM) -> torch.Tensor:
    """(3) Round the scaled tile to bf16 and accumulate in bf16 (`mx_smem` is bf16)."""
    return FM.bf16_accum_add(C, FM.q_bf16_rne(tile))


#: Fuse the elementwise chains with torch.compile. OFF by default, and **UNVERIFIED**: it is
#: wired up but has NOT been gated by verify_rtl_exact.py, because compiling these functions
#: on CPU did not finish in 40 minutes on the box it was written on. Before trusting any
#: number produced with it ON, run `MXG_RTL_COMPILE=1 python3 rtl_exact/verify_rtl_exact.py`
#: on the target machine and require the usual 65536/65536.
#:
#: WHY IT MATTERS. MEASURED on an sm_120 GPU at seqlen 2048, the datapath runs at a flat
#: **0.31 G element-steps/s** across every projection shape (q_proj 27.9 s for 8.6 G, gate_proj
#: 76.3 s for 23.6 G, down_proj 76.4 s for 23.6 G -- the same rate), i.e. 1.8 h per window and ~29 h
#: for 16. It is not shape-bound: `_rne_vec` and `_add_exact_vec` each issue 20-40 separate CUDA
#: kernels, every one reading and writing the whole tensor, so a 50 M-element batched lane-step
#: moves ~32 GB. These are pure elementwise chains on fixed shapes -- exactly what inductor fuses.
RTL_COMPILE = bool(int(_os.environ.get("MXG_RTL_COMPILE", "0"))) if "_os" in dir() else False


def _compiled_ops(FM):
    """(product_quantize, accumulate, cross_tile) with the module closed over, so they compile.

    `e`/`m` stay plain ints, which makes them compile-time constants -- inductor specializes per
    lane precision, and there are only four distinct pairs in the schedule.
    """
    def prod(x, exp: int, man: int):
        q = FM.mx_product_quantize_trunc(x, exp, man)
        return torch.where(q.abs() < 2.0 ** PROD_FLOOR, torch.zeros_like(q), q)

    def acc(a, prod_t, e: int, m: int):
        return FM.fp_add_exact(FM.fp_quantize_rne(a, e, m), FM.fp_quantize_rne(prod_t, e, m), e, m)

    def cross(C, tile):
        return FM.bf16_accum_add(C, FM.q_bf16_rne(tile))

    c = dict(dynamic=False, fullgraph=False)
    return (torch.compile(prod, **c), torch.compile(acc, **c), torch.compile(cross, **c))


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
    _prod_op, _acc_op, _cross_op = (_compiled_ops(FM) if RTL_COMPILE else (None, None, None))

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

        # WINDOWS ARE INDEPENDENT, so they are computed in BATCHES. `S_red` is zeroed per window
        # and only folded into C at the end, so nothing couples one window's reduction to another's
        # -- the sequential part is the 16 LANES inside a window, whose accumulator precision
        # varies. Batching turns K sequential steps into `window` of them on tensors `W` times
        # larger, which is what makes this tractable on a GPU: the per-k form issues ~80 small
        # kernels over an [M,N] tile and is dominated by launch overhead.
        #
        # THE FOLD INTO C STAYS SEQUENTIAL AND IN ORDER. `cross_tile_accumulate` is bf16 and bf16
        # addition is not associative, so reordering it would change the result.
        if K % window or (BLOCK % window and K > BLOCK):
            return _simulate_atw_rtl_serial(self, A, B, P_A, X_A, P_B, X_B, C, window, FM)

        n_win = K // window
        # Peak live memory is ~10 tensors of [W, M, N] float32 inside the quantizers.
        budget = getattr(self, "_rtl_batch_bytes", RTL_BATCH_BYTES)
        W = max(1, min(n_win, int(budget // max(1, 10 * 4 * M * N))))
        lanes = ([self.acc_schedule[l] for l in range(window)] if self.acc_schedule is not None
                 else [(self.acc_fixed_e, self.acc_fixed_m)] * window)

        for w0 in range(0, n_win, W):
            w1 = min(w0 + W, n_win)
            widx = torch.arange(w0, w1, device=device)
            S_red = torch.zeros((w1 - w0, M, N), dtype=torch.float32, device=device)
            for l in range(window):
                ks = widx * window + l                                   # [W]
                a = P_A.index_select(0, ks)                              # [W, M]
                b = P_B.index_select(0, ks)                              # [W, N]
                outer = a.unsqueeze(2) * b.unsqueeze(1)                  # [W, M, N]
                e_acc, m_acc = lanes[l]
                if _prod_op is not None:
                    outer_q = _prod_op(outer, self.prod_e, self.prod_m)           # (1)
                    S_red = _acc_op(S_red, outer_q, e_acc, m_acc)                 # (2)
                else:
                    outer_q = product_quantize(outer, self.prod_e, self.prod_m, FM)
                    S_red = accumulate(S_red, outer_q, e_acc, m_acc, FM)
            for i in range(w1 - w0):
                g_block = ((w0 + i) * window) // BLOCK
                scale_map = X_A[g_block, :].unsqueeze(1) * X_B[g_block, :].unsqueeze(0)
                tile = S_red[i] * scale_map
                C = _cross_op(C, tile) if _cross_op is not None else \
                    cross_tile_accumulate(C, tile, FM)                            # (3)

        return C

    EC.MXLinearSim._simulate_atw = _simulate_atw_rtl
    EC.MXLinearSim._rtl_exact = True


def is_installed(eval_complete_module) -> bool:
    return bool(getattr(eval_complete_module.MXLinearSim, "_rtl_exact", False))
