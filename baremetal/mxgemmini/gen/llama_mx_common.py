"""Shared by gen_llama_layer_mx.py (data) and golden_llama_layer_mx.py (bit-exact golden) for src/llama_layer_mx.c:
one decoder layer of a 7-13B model on MxGemmini, in one of three precisions, on one of four mesh sizes.

  models   llama31_8b, llama2_7b, llama2_13b, qwen25_7b, mistral_7b (+ `tiny`, a bring-up shape)
  formats  fp8  E4M3 everywhere
           fp4  E2M1 projections (Q/K/V, o_proj, gate/up, down) and their inputs; attention + KV cache in E4M3
           fp6  E3M2 through 16-entry LUTs (G = LUT_G), same split: attention + KV cache in E4M3
  meshes   fp8 on 16x16, 8x8, 32x32; fp4 / fp6 on 16x16, 8x8, 4x4

The plan functions below decide every chunking that changes a NUMBER (RMSNorm's partial sums, attention's key
blocks). The golden writes them into the per-mesh expect header and the kernel compiles against those, so the two
cannot disagree.
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
class Model:
    name: str          # what it stands for
    D: int
    F: int
    NH: int
    NKV: int
    HD: int
    eps: float
    theta: float
    bias: bool = False             # Q/K/V projection biases (Qwen2)
    llama3_rope: bool = False      # rope_scaling {"rope_type": "llama3", factor 8, low 1, high 4, orig 8192}
    sk: int = 2048                 # cached keys before the chunk
    m: int = 64                    # query tokens in the chunk


MODELS = {
    "llama31_8b": Model("Llama-3.1-8B", 4096, 14336, 32, 8, 128, 1e-5, 500000.0, llama3_rope=True),
    "llama2_7b":  Model("Llama-2-7B", 4096, 11008, 32, 32, 128, 1e-5, 10000.0),
    "llama2_13b": Model("Llama-2-13B", 5120, 13824, 40, 40, 128, 1e-5, 10000.0),
    "qwen25_7b":  Model("Qwen2.5-7B", 3584, 18944, 28, 4, 128, 1e-6, 1000000.0, bias=True),
    "mistral_7b": Model("Mistral-7B-v0.1", 4096, 14336, 32, 8, 128, 1e-5, 10000.0),
    # bring-up only: every code path of the big ones (GQA, bias, chunked RMSNorm / RoPE / SwiGLU) at a tiny cost
    "tiny":       Model("tiny (bring-up)", 512, 1536, 4, 2, 128, 1e-6, 10000.0, bias=True, sk=256),
}
FORMATS = {"fp8": 0, "fp6": 1, "fp4": 2}    # LMX_FMT: the projections' operand format
MESHES = {"fp8": (16, 8, 32), "fp4": (16, 8, 4), "fp6": (16, 8, 4)}
LUT_G = 4                                   # one LUT per 16 rows (activations) / 16 columns (weights)
LUT_COPIES = 4                              # act tables per tensor: 64 rows >> G (all the same table)


def data_name(model: str, fmt: str) -> str:
    return f"llama_mx_{model}_{fmt}"


def expect_name(model: str, fmt: str, dim: int) -> str:
    return f"llama_mx_{model}_{fmt}_d{dim}_expect"


def vlanes(dim: int) -> int:
    """BF16 VPU lanes per scratchpad row (one row = DIM bytes)."""
    return dim // 2


def bank_bytes(dim: int) -> int:
    """4 banks: 256 KB at DIM 8 / 16 / 32, 128 KB at DIM 4."""
    return {4: 8192 * 4, 8: 8192 * 8, 16: 4096 * 16, 32: 2048 * 32}[dim]


def rms_nc(D: int, dim: int, fmt: str) -> int:
    """RMSNorm column chunk: the largest divisor of D (a multiple of 128) whose 8-token stream fits a bank pair --
    bank 2s: x [8][Nc] + w [Nc] BF16; bank 2s+1: y [8][Nc] BF16, 16 stats rows, codes [8][Nc] -- and whose RSUM
    row length (Nc / lanes rows) fits the VPU's 10-bit counter. The sum of squares is RSUM per chunk, the partial
    sums added in BF16: this choice changes the result, so the kernel takes it from the golden."""
    bb, cb = bank_bytes(dim), (8 if fmt == "fp8" else 4)
    best = 0
    for nc in range(128, D + 1, 128):
        if D % nc == 0 and 18 * nc <= bb and 16 * nc + 16 * dim + cb * nc <= bb and nc // vlanes(dim) <= 1024:
            best = nc
    assert best, f"no RMSNorm chunk for D={D} at DIM {dim}"
    return best


def attn_blocks(m: Model, dim: int) -> tuple[int, int, int]:
    """(BK, BK_FIRST, BK_LAST) of the flash-attention key blocks over sk + m keys; the last block is the chunk itself
    (the causal mask). The 4x4 mesh's 128 KB scratchpad takes 64-key blocks."""
    bk = 64 if dim == 4 else 128
    first = bk
    assert (m.sk + m.m - first - m.m) % bk == 0
    return bk, first, m.m
