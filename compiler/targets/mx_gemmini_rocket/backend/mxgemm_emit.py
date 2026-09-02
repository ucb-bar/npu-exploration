"""Command buffer -> bare-metal C driver for the Rocket-hosted MX (microscaling) Gemmini.

This is the target's codegen: the piece merlin itself would carry if it shipped an mxgemmini
backend. It consumes a merlin command buffer and nothing else — no fixture data, no bring-up paths,
no knowledge of any particular matmul. Operand-specific material lives in ``app/``.

Scope: ONE weight-stationary MX matmul (RES_PACK -> MATMUL_RESIDENT -> COMMIT -> EVICT), BF16 output.
Anything else raises :class:`MxEmitError` rather than emitting C that silently miscomputes — a wrong
tiling does not fault, it returns plausible wrong numbers.

Interface follows the other merlin backends: ``generate_driver(cb, ...) -> str`` (cf.
``gemmini_codegen.generate_driver``, ``muon_codegen.emit_kernel_cpp``), with a private ``_plan(cb)``
deriving the shape/tiling plan (cf. ``muon_codegen._plan``, the radiance backend's ``_plan``).

Provenance
----------
The emitted instruction sequence is reimplemented from
``software/gemmini-rocc-tests/bareMetalC/matmul_tiled_fp8_64x64.c``, which is Rocket-native and
PASSes on spike. The module's *structure* follows the approach in radiance's ``mxgemm_lib.hpp``, but
no code, header, or path from ``radiance-kernels/`` is used (plan sections 1.3 / 2.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

# --- Facts from the target contract / gemmini.h. Nothing here is invented. ------------------------
# Operand format -> the 2-bit CONFIG_EX code (libgemmini/README.md "CONFIG_EX changes").
OPERAND_FMT = {"fp8": 0, "fp6": 1, "fp4": 2}
OUTPUT_FMT = {"fp8": 0, "fp6": 1, "fp4": 2, "bf16": 3}
# Operand dtype -> this target's short format key. Two spellings are accepted on purpose:
#   * the merlin_iface contract grammar uses MLIR's builtin fp8 names (`f8E4M3FN`), which is what
#     the shipped MX capsules declare — block scaling is a TARGET property, not part of the type;
#   * merlin's quant_formats registry names (`mxfp8`) name the block-scaled format directly.
DTYPE_TO_FMT = {
    "f8E4M3FN": "fp8", "f6E3M2FN": "fp6", "f4E2M1FN": "fp4",
    "mxfp8": "fp8", "mxfp6": "fp6", "mxfp4": "fp4",
}
# One E8M0 exponent per 32 K-elements (MxRequantizer.scala).
BLOCK_SCALE_GROUP = 32
# BF16 results drain packed 4 per uint64 word.
BF16_PER_WORD = 4

# [RTL] geometry defaults (gemmini_params.h DIM/BANK_NUM/BANK_ROWS). Overridable per buffer via
# cb["params"] so a re-elaborated mesh needs no code edit.
DEFAULT_GEOMETRY = {"dim": 16, "bank_num": 4, "bank_rows": 4096, "spad_dest": 128, "addr_len": 32}


class MxEmitError(RuntimeError):
    """A command buffer this backend refuses to lower. The caller turns this into a cb-level
    ``declined`` record; emitting a terminator-only program instead would grade as wrong arithmetic
    rather than as an honest decline."""


@dataclass(frozen=True)
class MxGemmPlan:
    """The derived plan for one MX GEMM — the output of :func:`_plan`, not a caller-supplied config.

    Field names double as the knob surface a DSE sweep would drive.
    """

    m: int
    n: int
    k: int
    act_fmt: str
    wgt_fmt: str
    out_fmt: str
    lhs: str
    weight: str
    out: str
    dim: int
    bank_num: int
    bank_rows: int
    spad_dest: int
    addr_len: int

    @property
    def tiles_i(self) -> int:
        return self.m // self.dim

    @property
    def tiles_j(self) -> int:
        return self.n // self.dim

    @property
    def tiles_k(self) -> int:
        return self.k // self.dim

    @property
    def scale_groups(self) -> int:
        """E8M0 groups along K — one scale byte per operand row/col per group."""
        return self.k // BLOCK_SCALE_GROUP

    @property
    def spad_rows(self) -> int:
        return self.bank_num * self.bank_rows

    @property
    def a_base(self) -> int:
        return 0

    @property
    def b_base(self) -> int:
        """B tiles are laid out from the END of the scratchpad, growing down."""
        return self.spad_rows - self.tiles_k * self.tiles_j * self.dim

    @property
    def use_lut(self) -> int:
        """The FP6 path is LUT-indexed on whichever side carries fp6."""
        return int("fp6" in (self.act_fmt, self.wgt_fmt))

    @property
    def out_cols(self) -> int:
        """Width of the packed output row, in uint64 words. The output is [M][N], so this is N —
        NOT M. (Identical when M == N, which is why the reference could use either.)

        BF16 packs 4 per uint64; FP8 codes pack 8.
        """
        per_word = BF16_PER_WORD if self.out_fmt == "bf16" else 2 * BF16_PER_WORD
        return self.n // per_word

    @property
    def out_elem_bytes(self) -> int:
        return 2 if self.out_fmt == "bf16" else 1

    @property
    def out_u16_words(self) -> int:
        """MX shared-memory words to drain — the unit ``MX_READ_SMEM`` (funct 28) counts."""
        return self.m * self.n * self.out_elem_bytes // 2

    @property
    def scale_blocks(self) -> int:
        """E8M0 output scale blocks per row, on the requant path: one per 32 output columns.
        The requantizer writes ``scale_dram + m * scale_blocks + bi``."""
        return self.n // BLOCK_SCALE_GROUP


# --- Plan derivation ------------------------------------------------------------------------------

def _dtype_to_fmt(dtype: str, where: str) -> str:
    if dtype not in DTYPE_TO_FMT:
        raise MxEmitError(
            f"{where} dtype {dtype!r} is not a microscaling operand format "
            f"(expected one of {sorted(DTYPE_TO_FMT)})")
    return DTYPE_TO_FMT[dtype]


def _plan(cb: dict[str, Any]) -> MxGemmPlan:
    """Derive the GEMM plan from the command buffer. Fails closed on anything out of scope."""
    cmds = cb.get("commands", [])
    if cb.get("declined"):
        raise MxEmitError(f"command buffer is already declined: {cb['declined']}")
    packs = [c for c in cmds if c["opcode"] == "RES_PACK"]
    matmuls = [c for c in cmds if c["opcode"] in ("MATMUL_RESIDENT", "MATMUL")]
    commits = [c for c in cmds if c["opcode"] == "COMMIT"]
    if len(packs) != 1:
        raise MxEmitError(f"expected exactly one RES_PACK, got {len(packs)}")
    if len(matmuls) != 1 or len(commits) != 1:
        raise MxEmitError(
            f"scope is a single matmul+commit, got {len(matmuls)} matmuls / {len(commits)} commits")

    tensors = cb.get("tensors") or {}
    weight, resident = packs[0]["operands"]["src"], packs[0]["operands"]["dst"]
    mm = matmuls[0]["operands"]
    if mm["rhs"] != resident:
        raise MxEmitError("the matmul must consume the packed resident weight")
    lhs = mm["lhs"]
    commit = commits[0]
    if commit["operands"]["src"] != mm["dst"]:
        raise MxEmitError("the commit must consume the matmul's accumulator")
    out = commit["operands"]["dst"]

    # Only LEAF tensors appear in the table — the merlin_iface grammar declares inputs and weights,
    # while committed outputs are named by the COMMIT op. So the output shape is DERIVED
    # ("dst rows x resident cols", per interface_grammar.md), not looked up.
    for name in (lhs, weight):
        if name not in tensors:
            raise MxEmitError(f"tensor {name!r} missing from the command buffer's tensor table")
    m, k_a = tensors[lhs]["shape"]
    k_w, n = tensors[weight]["shape"]
    if k_a != k_w:
        raise MxEmitError(f"contraction mismatch: lhs K={k_a} vs weight K={k_w}")

    attrs = commit.get("attributes") or {}
    epilogue = list(attrs.get("epilogue", []))
    if epilogue:
        raise MxEmitError(
            f"epilogue {epilogue} unsupported — the E8M0 requant IS this datapath's scaling, and no "
            "additional epilogue is emitted on the BF16 output path")
    # Output dtype: bf16 (raw accumulator readout) or an MX format (the requantizer's E8M0
    # write-back path, which is what a CHAINED matmul consumes as its next operand).
    out_dtype = attrs.get("output_dtype") or tensors.get(out, {}).get("dtype", "bf16")
    out_fmt = "bf16" if out_dtype == "bf16" else DTYPE_TO_FMT.get(out_dtype)
    if out_fmt is None:
        raise MxEmitError(
            f"output dtype {out_dtype!r} is neither 'bf16' nor a microscaling format "
            f"({sorted(DTYPE_TO_FMT)})")
    if out_fmt in ("fp6", "fp4"):
        raise MxEmitError(f"requant to {out_fmt} not emitted yet (fp8 and bf16 only)")

    geom = dict(DEFAULT_GEOMETRY)
    geom.update({k: v for k, v in (cb.get("params") or {}).items() if k in DEFAULT_GEOMETRY})

    plan = MxGemmPlan(
        m=m, n=n, k=k_a,
        act_fmt=_dtype_to_fmt(tensors[lhs].get("dtype", ""), f"lhs {lhs!r}"),
        wgt_fmt=_dtype_to_fmt(tensors[weight].get("dtype", ""), f"weight {weight!r}"),
        out_fmt=out_fmt,
        lhs=lhs, weight=weight, out=out,
        **geom)
    _validate(plan)
    return plan


def _validate(p: MxGemmPlan) -> None:
    """Enforce the contract's `legality` block."""
    if p.k % BLOCK_SCALE_GROUP:
        raise MxEmitError(
            f"K={p.k} is not a multiple of the E8M0 block-scale group ({BLOCK_SCALE_GROUP}) — "
            "every K group needs its own scale exponent")
    for name, val in (("M", p.m), ("N", p.n), ("K", p.k)):
        if val % p.dim:
            raise MxEmitError(f"{name}={val} is not a whole multiple of the PE tile ({p.dim})")
    if p.b_base <= p.tiles_i * p.tiles_k * p.dim:
        raise MxEmitError(
            f"scratchpad overflow: A needs {p.tiles_i * p.tiles_k * p.dim} rows, B starts at "
            f"{p.b_base} of {p.spad_rows}")
    if p.out_fmt == "bf16" and p.n % BF16_PER_WORD:
        raise MxEmitError(f"N={p.n} must be a multiple of {BF16_PER_WORD} to pack the BF16 output")
    if p.out_fmt != "bf16" and p.n % BLOCK_SCALE_GROUP:
        raise MxEmitError(
            f"N={p.n} must be a multiple of {BLOCK_SCALE_GROUP} on the requant path — the "
            "requantizer emits one E8M0 code per 32 output columns")


def _mx_operands(cb: dict[str, Any], p: MxGemmPlan) -> dict[str, Sequence[Sequence[int]]]:
    """Pull raw MX operand codes + E8M0 scales off the command buffer.

    These ride on the cb as ``mx_operands`` rather than coming from ``materialize_inputs``, because
    the generic tensor table carries DECODED values while the datapath consumes raw codes plus a
    separate block-scale stream that cannot be reconstructed from them. Same side-channel the muon
    MX path uses (``muon_mx_codegen``: "attached to the cb as ``mx_operands``").
    """
    ops = cb.get("mx_operands")
    if not ops:
        raise MxEmitError(
            "command buffer carries no 'mx_operands' — MX needs raw operand codes and E8M0 block "
            "scales, which the decoded tensor table cannot supply")
    want = {"a_codes": (p.m, p.k), "b_codes": (p.k, p.n),
            "a_scales": (p.scale_groups, p.m), "b_scales": (p.scale_groups, p.n)}
    for key, (rows, cols) in want.items():
        if key not in ops:
            raise MxEmitError(f"mx_operands missing {key!r}")
        got = (len(ops[key]), len(ops[key][0]) if ops[key] else 0)
        if got != (rows, cols):
            raise MxEmitError(f"mx_operands[{key!r}] is {got}, expected {(rows, cols)}")
    return ops


# --- Transport seam -------------------------------------------------------------------------------
# The only target-specific axis. The MMIO and RoCC variants of MX-Gemmini share every funct code and
# every rs1/rs2 packing; they differ in how the command word reaches the accelerator and in how
# scales and results move. Also where this target's own V1/V2/V3 output modes differ.
#
# NOTE: no merlin backend has this seam — it is ours, justified by those output modes.

class Transport(Protocol):
    name: str

    def includes(self) -> list[str]: ...
    def emit_load_scales(self, p: MxGemmPlan) -> list[str]: ...
    def emit_drain(self, p: MxGemmPlan) -> list[str]: ...
    def out_dest(self, p: MxGemmPlan) -> str: ...
    def out_flag(self) -> str: ...


@dataclass(frozen=True)
class SpikeSmemTransport:
    """The spike / libgemmini endpoint (oracle tier L1).

    Scales are DMA'd from DRAM by MX_LOAD_SCALES (funct 27); the BF16 result is drained from MX
    shared memory by MX_READ_SMEM (funct 28).
    """

    name: str = "spike_smem"

    def includes(self) -> list[str]:
        return ['#include "include/gemmini_testutils.h"']

    def emit_load_scales(self, p: MxGemmPlan) -> list[str]:
        return [
            "  /* E8M0 block scales: DRAM -> mx_scale_{a,b}_mem. sel 0 = A rows, 1 = B cols. */",
            "  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);",
            "  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);",
        ]

    def out_dest(self, p: MxGemmPlan) -> str:
        return str(p.spad_dest)

    def out_flag(self) -> str:
        # Loop-FSM skip mask; keeps the spad store that the smem drain reads back.
        return "0x38"

    def emit_drain(self, p: MxGemmPlan) -> list[str]:
        return [
            f"  /* Drain the {p.out_fmt} result out of MX shared memory (funct 28); the count is",
            f"     in u16 words, so {p.out_elem_bytes}-byte elements pack {2 // p.out_elem_bytes} per word. */",
            f"  gemmini_mx_read_smem(&C_hw[0][0], {p.spad_dest} * 16, {p.out_u16_words});",
        ]


# --- C emission -----------------------------------------------------------------------------------

def _c_array_2d(ctype: str, name: str, rows: Sequence[Sequence[int]], dims: str) -> str:
    body = ",\n".join("  {" + ",".join(str(int(v)) for v in row) + "}" for row in rows)
    return f"static const {ctype} {name}{dims} = {{\n{body}\n}};"


def _emit_operand_data(p: MxGemmPlan, ops: dict[str, Sequence[Sequence[int]]]) -> list[str]:
    """Bake operands in, so the same bytes reach the reference, the simulator and the device."""
    return [
        _c_array_2d("uint8_t", "A_in", ops["a_codes"], f"[{p.m}][{p.k}]"),
        _c_array_2d("uint8_t", "B_in", ops["b_codes"], f"[{p.k}][{p.n}]"),
        _c_array_2d("uint8_t", "A_scales_row", ops["a_scales"], f"[{p.scale_groups}][{p.m}]"),
        _c_array_2d("uint8_t", "B_scales_col", ops["b_scales"], f"[{p.scale_groups}][{p.n}]"),
    ]


def _emit_mvin(p: MxGemmPlan) -> list[str]:
    """Tile move-in for both operands.

    Addresses and spad slots are DERIVED FROM THE SPIKE MODEL's own indexing (``gemmini.cc``, the
    LOOP_WS MX kernel), not copied from the reference test:

        A_t = A_sp + (i * TK + k_outer) * DIM     -> A slot = i*tiles_K + k
        B_t = B_sp + (k_outer * TJ + j) * DIM     -> B slot = k*tiles_J + j

    The reference test writes the B slot as ``j*tiles_K + k`` and strides BOTH operands by M. All
    three coincide only when M == N == K, which is why its 64x64x64 case passes. Non-square shapes
    crashed spike until this was corrected: A strides by K, B strides by N, and the B slot uses
    tiles_J.
    """
    d = p.dim
    return [
        "  /* MVIN A[M][K]: row stride K; tile (i,k) -> a_base + (i*tiles_K + k)*DIM */",
        f"  gemmini_config_ld({p.k} * sizeof(uint8_t));",
        f"  for (int i = 0; i < {p.tiles_i}; i++) {{",
        f"    for (int k = 0; k < {p.tiles_k}; k++) {{",
        f"      const uint8_t *src = ((const uint8_t *)A_in) + i * {d} * {p.k} + k * {d};",
        f"      uint32_t sp_addr = {p.a_base} + (i * {p.tiles_k} + k) * {d};",
        f"      gemmini_extended_mvin((void *)src, sp_addr, {d}, {d});",
        "    }",
        "  }",
        "",
        "  /* MVIN B[K][N]: row stride N; tile (k,j) -> b_base + (k*tiles_J + j)*DIM */",
        f"  gemmini_config_ld({p.n} * sizeof(uint8_t));",
        f"  for (int k = 0; k < {p.tiles_k}; k++) {{",
        f"    for (int j = 0; j < {p.tiles_j}; j++) {{",
        f"      const uint8_t *src = ((const uint8_t *)B_in) + k * {d} * {p.n} + j * {d};",
        f"      uint32_t sp_addr = {p.b_base} + (k * {p.tiles_j} + j) * {d};",
        f"      gemmini_extended_mvin((void *)src, sp_addr, {d}, {d});",
        "    }",
        "  }",
    ]


def _emit_report(p: MxGemmPlan) -> list[str]:
    """Print the shared merlin console protocol: ``OUT <name> <rows> <cols> v...`` / ``METRIC`` /
    ``DONE`` (``runtime/backends/base.parse_console``).

    BF16 values are reported as **bit patterns**; FP8 as raw **codes**, plus a second OUT line
    carrying the requantizer's per-row per-32-column E8M0 scale codes. Both are the datapath's
    native output, undecoded — nothing is lost on the way out.
    """
    tail = [
        '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));',
        '  printf("METRIC cycle_window_mx_gemmini_region 1\\n");',
        '  printf("DONE\\n");',
    ]
    if p.out_fmt == "bf16":
        return [
            "  /* merlin console protocol — parsed by runtime.backends.base.parse_console. */",
            f'  printf("OUT {p.out} {p.m} {p.n}");',
            f"  for (int i = 0; i < {p.m}; i++)",
            f"    for (int j = 0; j < {p.n}; j++)",
            f"      printf(\" %u\", (unsigned)((C_hw[i][j / {BF16_PER_WORD}]"
            f" >> ((j % {BF16_PER_WORD}) * 16)) & 0xFFFF));",
            '  printf("\\n");',
        ] + tail
    return [
        "  /* merlin console protocol. FP8 requant output: the packed codes, then the E8M0 scale",
        "     codes the requantizer wrote to DRAM (one per row per 32 output columns). */",
        "  { const uint8_t *codes = (const uint8_t *)&C_hw[0][0];",
        f'    printf("OUT {p.out} {p.m} {p.n}");',
        f"    for (long i = 0; i < {p.m} * {p.n}; i++) printf(\" %u\", (unsigned)codes[i]);",
        '    printf("\\n"); }',
        "  { const uint8_t *sc = (const uint8_t *)scale_factors;",
        f'    printf("OUT {p.out}_scales {p.m} {p.scale_blocks}");',
        f"    for (long i = 0; i < {p.m} * {p.scale_blocks}; i++) printf(\" %u\", (unsigned)sc[i]);",
        '    printf("\\n"); }',
    ] + tail


def generate_driver(cb: dict[str, Any], *, transport: Transport | None = None) -> str:
    """Return the C source of the bare-metal MX GEMM driver for ``cb``."""
    p = _plan(cb)
    ops = _mx_operands(cb, p)
    tr = transport or SpikeSmemTransport()

    lines: list[str] = [
        "/* Generated by mxgemm_emit.py for target mx_gemmini_rocket — do not edit. */",
        f"/* {p.m}x{p.n}x{p.k}  A:{p.act_fmt} B:{p.wgt_fmt} -> {p.out_fmt}"
        f"   tiles {p.tiles_i}x{p.tiles_j}x{p.tiles_k}   transport:{tr.name} */",
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <string.h>",
        *tr.includes(),
        "",
        f"#define OUT_COLS {p.out_cols}",
        "typedef uint64_t out_t;",
        "",
        *_emit_operand_data(p, ops),
        "",
        "int main(void) {",
        "  uint64_t c0, c1;",
        f"  static out_t C_hw[{p.m}][OUT_COLS];",
        "  /* Requant scale write-back target. Unused on the BF16 output path, but the mxquant",
        "     config still needs a valid address to point at. */",
        "  static uint32_t scale_factors[512];",
        "  memset(C_hw, 0, sizeof(C_hw));",
        "  memset(scale_factors, 0, sizeof(scale_factors));",
        "",
        "  gemmini_flush(0);",
        "  /* CONFIG_EX: weight-stationary, no sys activation/shift, identity acc scale,",
        "     C_stride=1 A_stride=1, no transpose, then the four MX fields:",
        f"     act={p.act_fmt}({OPERAND_FMT[p.act_fmt]}) wgt={p.wgt_fmt}({OPERAND_FMT[p.wgt_fmt]}) "
        f"out={p.out_fmt}({OUTPUT_FMT[p.out_fmt]}) uselut={p.use_lut} */",
        "  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0,",
        f"                              false, {OPERAND_FMT[p.act_fmt]}, {OPERAND_FMT[p.wgt_fmt]}, "
        f"{OUTPUT_FMT[p.out_fmt]}, {p.use_lut});",
        "",
        *tr.emit_load_scales(p),
        "",
        "  /* Accelerator region: operand move-in, the mesh loop, and the drain. */",
        "  c0 = read_cycles();",
        *_emit_mvin(p),
        "",
        "  /* Output row stride, then the requant/scale-memory config (funct 26). On the requant",
        "     path the reference uses a single-word store stride; the drain is MX_READ_SMEM. */",
        ("  gemmini_config_st(OUT_COLS * sizeof(out_t));" if p.out_fmt == "bf16"
         else "  gemmini_config_st(1 * sizeof(out_t));"),
        f"  gemmini_mxquant_config_mvout((uint64_t)scale_factors, {p.tiles_i}, {p.tiles_j}, "
        f"{p.tiles_k}, 0, 0, 1);",
        "",
        "  /* The MX matmul: LOOP_WS_CONFIG_BOUNDS + LOOP_WS_CONFIG_SPAD_AB + LOOP_WS.",
        "     The SPAD_AB command is what marks the following LOOP_WS as the MX variant. */",
        f"  gemmini_loop_ws_spad({p.tiles_i}, {p.tiles_j}, {p.tiles_k},",
        "                       0, 0, 0,",
        f"                       {p.a_base},",
        f"                       {p.spad_rows},",
        "                       0,",
        f"                       {tr.out_dest(p)},",
        "                       false, false,",
        "                       false, false, false,",
        "                       NO_ACTIVATION,",
        "                       0, 0,",
        "                       false,",
        f"                       {tr.out_flag()});",
        "",
        *tr.emit_drain(p),
        "  gemmini_fence();",
        "  c1 = read_cycles();",
        "",
        *_emit_report(p),
        "  return 0;",
        "}",
    ]
    return "\n".join(lines) + "\n"
