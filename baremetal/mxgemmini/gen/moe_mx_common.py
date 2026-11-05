"""Shared by gen_moe_mx.py (data) and golden_moe_mx.py (bit-exact golden) for src/moe_layer_mx.c: the MoE block of one
gpt-oss-20b decoder layer on MxGemmini, for T tokens (a decode step, or a speculative-decoding verify of T - 1 drafts).

  shape    gpt-oss-20b: hidden 2880, 32 experts of FFN 2880 (gate / up / down, with biases), top-4 routing, router bias,
           RMSNorm eps 1e-5, SwiGLU variant: clamp(g, max 7), clamp(u, -7, 7), g * sigmoid(1.702 g) * (u + 1)
  formats  fp8  experts E4M3 (activations E4M3)
           fp4  experts E2M1 (the checkpoint's native MXFP4; activations E2M1 too, on the quad mesh)
           the router is E4M3 in both (gpt-oss keeps it in BF16)
  meshes   fp8 on 8x8, 16x16, 32x32; fp4 on 4x4, 8x8, 16x16 (the same three peak rates)

The tokens are padded to MT rows (a multiple of 16 and of DIM): every touched expert runs all MT rows, the rows of tokens
not routed to it weighted 0 in the combine. For T <= PQ * DIM this is the same mesh work as gathering each expert's
tokens (its group would be padded to PQ * DIM rows anyway); it is the decode / verify regime this kernel is for.

The plan functions below decide every chunking that changes a NUMBER (RMSNorm's partial sums). The golden writes them
into the per-mesh expect header and the kernel compiles against those, so the two cannot disagree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
KDIR = HERE.parent
DATA = KDIR / "data"
NPU = HERE.parents[2]
RTL_EXACT = NPU / "rtl_exact"


@dataclass(frozen=True)
class Moe:
    name: str
    D: int = 2880
    F: int = 2880
    E: int = 32          # experts
    K: int = 4           # experts per token
    eps: float = 1e-5
    alpha: float = 1.702
    limit: float = 7.0


GPTOSS = Moe("gpt-oss-20b")


@dataclass(frozen=True)
class Cfg:
    """One input: T tokens with a target routing of locality rho (each token keeps each of the previous token's
    experts with probability rho; 0 = independent uniform draws, the pessimistic case for speculative verify)."""
    T: int
    rho: float = 0.0

    @property
    def key(self) -> str:
        return f"gptoss20b_t{self.T}" + (f"_r{round(self.rho * 100)}" if self.rho else "")


CFGS = {c.key: c for c in [Cfg(t) for t in (1, 2, 4, 8, 16)] + [Cfg(t, 0.5) for t in (4, 8, 16)]}
FORMATS = {"fp8": 0, "fp4": 2}     # MOE_FMT: the experts' operand format (llama_mx_common.FORMATS numbering)
MESHES = {"fp8": (8, 16, 32), "fp4": (4, 8, 16)}
# FP4 block scales: scale = 2^(floor(log2 amax) - p). Weights are quantized offline, so their p is free: 2 is the OCP MX
# convention (E2M1's emax), as the gpt-oss checkpoint ships them, and the block max reaches all 8 magnitudes. The
# activations go through SPAD_REQUANT, whose p is fixed in the RTL (MxRequantizer.scala log2_pmax_floor, gemmini.cc
# log2_pmax; 0 = MXQuant's _po2, decided 2026-09-09): ACT_PMAX must equal the hardware's. (E4M3 does not depend on p.)
W_PMAX_FP4 = 2
ACT_PMAX_FP4 = 0
ROUTER_N = 64                       # router columns (32 experts, padded for the matmul tiling)


def experts_name(fmt: str) -> str:
    return f"moe_experts_{fmt}"


def cfg_name(cfg: str) -> str:
    return f"moe_in_{cfg}"


def expect_name(cfg: str, fmt: str, dim: int) -> str:
    return f"moe_mx_{cfg}_{fmt}_d{dim}_expect"


def vlanes(dim: int) -> int:
    return dim // 2


def bank_bytes(dim: int) -> int:
    return {4: 8192 * 4, 8: 8192 * 8, 16: 4096 * 16, 32: 2048 * 32}[dim]


TB = 16   # tokens per VPU block: SPAD_REQUANT needs rows * N / 32 % 32 == 0, and 2880 / 32 is odd, so 16 rows


def mt(T: int, dim: int, fmt: str) -> int:
    """Token rows: T padded to a multiple of TB (the VPU's token blocks) and of the experts' matmul row tile (DIM, or
    2 * DIM for fp4's row pairs on the quad mesh; the E4M3 router needs only DIM). Kernel: MT."""
    q = max(TB, dim * (2 if fmt == "fp4" else 1))
    return (T + q - 1) // q * q


def chunk(n: int, dim: int) -> int:
    """Column chunk of the VPU phases over a row of n BF16: the largest divisor of n that is a multiple of 64 (whole
    spad rows at every DIM; the 32x32 SPAD_REQUANT wants N % 64 == 0) whose TB-token block [TB][nc] BF16 fits one bank
    and whose per-token row count nc / lanes fits the VPU's 10-bit row counters (RSUM / bcast). RMSNorm's sum of squares
    is RSUM per chunk with the partial sums added in BF16, so this changes the result there: the golden fixes it."""
    best = 0
    for nc in range(64, n + 1, 64):
        if n % nc == 0 and 2 * TB * nc <= bank_bytes(dim) and nc // vlanes(dim) <= 1023:
            best = nc
    assert best, f"no chunk for {n} at DIM {dim}"
    return best


def softmax_width(dim: int) -> int:
    """Lanes per token in the routing softmax: the 4 picked logits, -inf padded to whole spad rows."""
    return max(8, vlanes(dim))
