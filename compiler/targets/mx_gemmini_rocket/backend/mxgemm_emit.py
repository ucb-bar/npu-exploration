"""Command buffer -> bare-metal C driver for the Rocket-hosted MX (microscaling) Gemmini.

This is the target's codegen: the piece merlin itself would carry if it shipped an mxgemmini
backend. It consumes a merlin command buffer and nothing else — no fixture data, no bring-up paths,
no knowledge of any particular matmul. Operand-specific material lives in ``app/``.

Scope: a CHAIN of one or more weight-stationary MX matmuls
(RES_PACK -> MATMUL_RESIDENT -> COMMIT -> EVICT, repeated), where stage *i+1*'s lhs is stage *i*'s
committed output. One matmul is the length-1 case and emits byte-for-byte what it always did.
Anything else raises :class:`MxEmitError` rather than emitting C that silently miscomputes — a wrong
tiling does not fault, it returns plausible wrong numbers.

A chained stage's operand does not come back to the host: the requantizer's FP8 write-back is drained
to a DRAM array and moved straight back in as the next stage's A. What the driver must do at that
join — "the seam" — is deliberately as close to nothing as the hardware permits; see
``planning/chain_seam_hw_notes.md`` for every part of it that exists only because of a hardware
property, and what would remove it.

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

from dataclasses import dataclass, replace
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

#: [ABI] Commit attribute naming the exponent shift the NEXT stage's A operand needs, applied on
#: device at the seam. 0 (absent) is the `weight` seam, where the compensation was folded into the
#: following weight's quantization on the host and the device does nothing. Namespaced because it is
#: this target's extension, not part of frozen grammar v0.1 — the grammar carries it as an opaque
#: integer attribute, which `_parse_attr_block` already supports.
CHAIN_CODE_SHIFT_ATTR = "mx_gemmini.chain_code_shift"

#: Unroll the seam's block axis when it is at most this wide. It is a compile-time constant and
#: almost always 2, so unrolling removes the inner loop and all index arithmetic from the only
#: work the driver does between two GEMMs.
_SEAM_UNROLL_MAX = 8


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
    #: Exponent shift the NEXT stage's A operand needs (see :data:`CHAIN_CODE_SHIFT_ATTR`).
    #: 0 means the seam moves no code bytes at all.
    chain_code_shift: int = 0

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

    @property
    def smem_rows(self) -> int:
        """MX shared-memory footprint of this stage's output, in ``spad_dest`` units.

        The mesh writes the full BF16 tile at ``smem_base + m*N + j`` before the requant post-pass
        packs it down in place (``gemmini.cc:1188``), so the footprint is M*N u16 words whatever the
        output format. Chained stages need DISJOINT footprints — see :func:`plan_chain`.
        """
        return -(-(self.m * self.n) // self.dim)


# --- Plan derivation ------------------------------------------------------------------------------

def _dtype_to_fmt(dtype: str, where: str) -> str:
    if dtype not in DTYPE_TO_FMT:
        raise MxEmitError(
            f"{where} dtype {dtype!r} is not a microscaling operand format "
            f"(expected one of {sorted(DTYPE_TO_FMT)})")
    return DTYPE_TO_FMT[dtype]


def plan_chain(cb: dict[str, Any]) -> list[MxGemmPlan]:
    """Derive one plan per matmul, in command order. Fails closed on anything out of scope.

    A chain is N repetitions of RES_PACK -> MATMUL_RESIDENT -> COMMIT (-> EVICT) where stage *i+1*'s
    lhs is stage *i*'s committed output. That is the only multi-matmul shape this backend lowers: a
    graph whose operands come from anywhere else (attention's ``S = Q @ K^T``) has no on-device
    operand path and is refused here rather than mis-emitted.

    Only LEAF tensors appear in the table — the merlin_iface grammar declares inputs and weights,
    while committed outputs are named by the COMMIT op. So an intermediate's shape is DERIVED as it
    is produced ("dst rows x resident cols", per interface_grammar.md), not looked up.
    """
    if cb.get("declined"):
        raise MxEmitError(f"command buffer is already declined: {cb['declined']}")

    tensors = cb.get("tensors") or {}
    geom = dict(DEFAULT_GEOMETRY)
    geom.update({k: v for k, v in (cb.get("params") or {}).items() if k in DEFAULT_GEOMETRY})

    # Shape/dtype of everything nameable, growing as commits produce intermediates.
    shape: dict[str, tuple[int, int]] = {
        n: (int(s["shape"][0]), int(s["shape"][1])) for n, s in tensors.items()}
    dtype: dict[str, str] = {n: s.get("dtype", "") for n, s in tensors.items()}

    res_src: dict[str, str] = {}                     # resident handle -> weight tensor
    pending: dict[str, tuple[str, str]] = {}         # acc -> (lhs, weight)
    plans: list[MxGemmPlan] = []

    for cmd in cb.get("commands", []):
        op, o = cmd["opcode"], cmd.get("operands", {})
        if op == "RES_PACK":
            res_src[o["dst"]] = o["src"]
        elif op in ("MATMUL_RESIDENT", "MATMUL"):
            if o["rhs"] not in res_src:
                raise MxEmitError(f"matmul {o['dst']!r} must consume a packed resident weight, "
                                  f"got {o['rhs']!r}")
            pending[o["dst"]] = (o["lhs"], res_src[o["rhs"]])
        elif op == "COMMIT":
            if o["src"] not in pending:
                raise MxEmitError(f"the commit must consume a matmul's accumulator, "
                                  f"got {o['src']!r}")
            lhs, weight = pending.pop(o["src"])
            plans.append(_plan_stage(len(plans), lhs, weight, o["dst"],
                                     cmd.get("attributes") or {}, shape, dtype, plans, geom))
        elif op == "EVICT":
            res_src.pop(o.get("handle"), None)
        else:
            raise MxEmitError(f"opcode {op!r} is not part of an MX matmul chain")

    if not plans:
        raise MxEmitError("no matmul+commit pair in the command buffer")
    if pending:
        raise MxEmitError(f"{len(pending)} matmul(s) with no commit: {sorted(pending)}")

    # A non-final stage's output IS the next stage's operand, so it has to be committed in an
    # operand encoding. BF16 is a host readout format: the mesh cannot take it back as an A operand.
    for i, (p, nxt) in enumerate(zip(plans, plans[1:])):
        if p.out_fmt == "bf16":
            raise MxEmitError(
                f"stage {i} ({p.out!r}) commits bf16 but stage {i + 1} consumes it — a chained "
                "commit must requantize to an MX format (the bf16 readout is not an operand)")
        if p.out_fmt != nxt.act_fmt:
            raise MxEmitError(
                f"stage {i} commits {p.out_fmt} but stage {i + 1} declares its lhs as "
                f"{nxt.act_fmt} — the requant output IS the next operand, so they must agree")

    _assign_smem(plans)
    return plans


def _plan_stage(i: int, lhs: str, weight: str, out: str, attrs: dict[str, Any],
                shape: dict[str, tuple[int, int]], dtype: dict[str, str],
                prior: list[MxGemmPlan], geom: dict[str, int]) -> MxGemmPlan:
    """One stage's plan, registering its output shape/dtype for the stage that consumes it."""
    for name in (lhs, weight):
        if name not in shape:
            raise MxEmitError(f"tensor {name!r} missing from the command buffer's tensor table")
    if i and lhs != prior[-1].out:
        raise MxEmitError(
            f"stage {i} takes lhs {lhs!r}, but only a CHAIN is lowerable: its lhs must be the "
            f"previous stage's output ({prior[-1].out!r}). An operand computed anywhere else has no "
            "on-device path.")
    m, k_a = shape[lhs]
    k_w, n = shape[weight]
    if k_a != k_w:
        raise MxEmitError(f"contraction mismatch: lhs K={k_a} vs weight K={k_w}")

    epilogue = list(attrs.get("epilogue", []))
    if epilogue:
        raise MxEmitError(
            f"epilogue {epilogue} unsupported — the E8M0 requant IS this datapath's scaling, and no "
            "additional epilogue is emitted on the BF16 output path")
    # Output dtype: bf16 (raw accumulator readout) or an MX format (the requantizer's E8M0
    # write-back path, which is what a CHAINED matmul consumes as its next operand).
    out_dtype = attrs.get("output_dtype") or dtype.get(out, "bf16")
    out_fmt = "bf16" if out_dtype == "bf16" else DTYPE_TO_FMT.get(out_dtype)
    if out_fmt is None:
        raise MxEmitError(
            f"output dtype {out_dtype!r} is neither 'bf16' nor a microscaling format "
            f"({sorted(DTYPE_TO_FMT)})")
    if out_fmt in ("fp6", "fp4"):
        raise MxEmitError(f"requant to {out_fmt} not emitted yet (fp8 and bf16 only)")

    plan = MxGemmPlan(
        m=m, n=n, k=k_a,
        act_fmt=_dtype_to_fmt(dtype[lhs], f"lhs {lhs!r}"),
        wgt_fmt=_dtype_to_fmt(dtype[weight], f"weight {weight!r}"),
        out_fmt=out_fmt,
        lhs=lhs, weight=weight, out=out,
        chain_code_shift=int(attrs.get(CHAIN_CODE_SHIFT_ATTR, 0) or 0),
        **geom)
    _validate(plan)
    shape[out], dtype[out] = (m, n), out_dtype
    return plan


def _assign_smem(plans: list[MxGemmPlan]) -> None:
    """Give every stage a DISJOINT MX shared-memory region, in place.

    Necessary because the mesh ACCUMULATES into smem and nothing clears it
    (``gemmini.cc:1190``: ``prev = bf16_to_f32(mx_smem[idx]); mx_smem[idx] = accum_add(prev, ...)``).
    With one matmul per process that is invisible — ``reset()`` zeroes smem at startup — but fused
    stages sharing a ``spad_dest`` would have stage *i+1* accumulate onto stage *i*'s residue.

    A functional-model workaround, not a hardware fact: see ``chain_seam_hw_notes.md`` §5 before
    carrying it to the Verilator path, where ``spad_dest`` is a real scratchpad address.
    """
    if len(plans) == 1:
        return
    budget = plans[0].spad_rows                      # mx_smem is sp_matrices*DIM*16 u16 words,
    dest = plans[0].spad_dest                        # i.e. bank_num*bank_rows in spad_dest units
    for i, p in enumerate(plans):
        if dest + p.smem_rows > budget:
            raise MxEmitError(
                f"MX shared memory exhausted at stage {i}: stages need "
                f"{dest + p.smem_rows} rows of {budget}. Chained stages must not share a region "
                "(gemmini.cc:1190 accumulates); shrink the chain or the intermediates.")
        plans[i] = replace(p, spad_dest=dest)
        dest += p.smem_rows


def _plan(cb: dict[str, Any]) -> MxGemmPlan:
    """The single-matmul plan. A one-element :func:`plan_chain`; kept because most of this module
    reads one plan at a time."""
    plans = plan_chain(cb)
    if len(plans) != 1:
        raise MxEmitError(f"expected a single matmul, got a chain of {len(plans)}")
    return plans[0]


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


def _mx_operands(cb: dict[str, Any], plans: list[MxGemmPlan]) -> list[dict[str, Any]]:
    """Pull raw MX operand codes + E8M0 scales off the command buffer, one bundle per stage.

    These ride on the cb as ``mx_operands`` rather than coming from ``materialize_inputs``, because
    the generic tensor table carries DECODED values while the datapath consumes raw codes plus a
    separate block-scale stream that cannot be reconstructed from them. Same side-channel the muon
    MX path uses (``muon_mx_codegen``: "attached to the cb as ``mx_operands``").

    Only stage 0 carries ``a_*``: every later stage's A operand is the previous stage's requantizer
    output, which is produced on device and never passes through here.
    """
    ops = cb.get("mx_operands")
    if not ops:
        raise MxEmitError(
            "command buffer carries no 'mx_operands' — MX needs raw operand codes and E8M0 block "
            "scales, which the decoded tensor table cannot supply")
    bundles = list(ops) if isinstance(ops, list) else [ops]
    if len(bundles) != len(plans):
        raise MxEmitError(
            f"mx_operands has {len(bundles)} bundle(s) for {len(plans)} matmul(s) — one per stage")
    for i, (p, bundle) in enumerate(zip(plans, bundles)):
        want = {"b_codes": (p.k, p.n), "b_scales": (p.scale_groups, p.n)}
        if i == 0:
            want |= {"a_codes": (p.m, p.k), "a_scales": (p.scale_groups, p.m)}
        for key, (rows, cols) in want.items():
            if key not in bundle:
                raise MxEmitError(f"mx_operands[{i}] missing {key!r}")
            got = (len(bundle[key]), len(bundle[key][0]) if bundle[key] else 0)
            if got != (rows, cols):
                raise MxEmitError(f"mx_operands[{i}][{key!r}] is {got}, expected {(rows, cols)}")
        if i and ("a_codes" in bundle or "a_scales" in bundle):
            raise MxEmitError(
                f"mx_operands[{i}] carries an A operand, but stage {i}'s A is stage {i - 1}'s "
                "requant output — supplying it from the host would silently ignore the device value")
    return bundles


# --- Transport seam -------------------------------------------------------------------------------
# The only target-specific axis. The MMIO and RoCC variants of MX-Gemmini share every funct code and
# every rs1/rs2 packing; they differ in how the command word reaches the accelerator and in how
# scales and results move. Also where this target's own V1/V2/V3 output modes differ.
#
# NOTE: no merlin backend has this seam — it is ours, justified by those output modes.

class Transport(Protocol):
    name: str

    def includes(self) -> list[str]: ...
    def emit_load_scales(self, p: MxGemmPlan, a_scales: str, b_scales: str) -> list[str]: ...
    def emit_drain(self, p: MxGemmPlan, dst: str) -> list[str]: ...
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

    def emit_load_scales(self, p: MxGemmPlan,
                         a_scales: str = "A_scales_row", b_scales: str = "B_scales_col") -> list[str]:
        return [
            "  /* E8M0 block scales: DRAM -> mx_scale_{a,b}_mem. sel 0 = A rows, 1 = B cols. */",
            f"  gemmini_mx_load_scales((uint64_t)&{a_scales}, sizeof({a_scales}), 0);",
            f"  gemmini_mx_load_scales((uint64_t)&{b_scales}, sizeof({b_scales}), 1);",
        ]

    def out_dest(self, p: MxGemmPlan) -> str:
        return str(p.spad_dest)

    def out_flag(self) -> str:
        # Loop-FSM skip mask; keeps the spad store that the smem drain reads back.
        return "0x38"

    def emit_drain(self, p: MxGemmPlan, dst: str = "C_hw") -> list[str]:
        return [
            f"  /* Drain the {p.out_fmt} result out of MX shared memory (funct 28); the count is",
            f"     in u16 words, so {p.out_elem_bytes}-byte elements pack {2 // p.out_elem_bytes} per word. */",
            f"  gemmini_mx_read_smem(&{dst}[0][0], {p.spad_dest} * 16, {p.out_u16_words});",
        ]


# --- C emission -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Names:
    """C identifiers for one stage's buffers.

    A single-matmul program keeps the historical unsuffixed names (``A_in``, ``B_in``,
    ``A_scales_row``, ``B_scales_col``, ``C_hw``, ``scale_factors``) so its emitted driver is
    byte-for-byte what it has always been. A chain suffixes per stage and points a chained stage's
    A operand at the previous stage's drained output.
    """

    a_codes: str
    a_scales: str
    b_codes: str
    b_scales: str
    out: str                 # drain destination for this stage's result
    out_scales: str          # where the requantizer writes its E8M0 codes


def _stage_names(i: int, plans: list[MxGemmPlan]) -> _Names:
    last, chained = i == len(plans) - 1, i > 0
    if len(plans) == 1:
        return _Names("A_in", "A_scales_row", "B_in", "B_scales_col", "C_hw", "scale_factors")
    return _Names(
        # A chained stage reads the previous stage's drained codes in place — no copy, because
        # MX_READ_SMEM already left them in the [M][K] layout mvin wants (hw notes, intro).
        a_codes=f"T{i - 1}" if chained else "A_in",
        a_scales=_a_scales_name(i, plans) if chained else "A_scales_row",
        b_codes=f"B{i}_in", b_scales=f"B{i}_scales_col",
        out="C_hw" if last else f"T{i}",
        out_scales="scale_factors" if last else f"T{i}_sc")


def _a_scales_name(i: int, plans: list[MxGemmPlan]) -> str:
    """Where stage *i* (>0) reads its A-side E8M0 scales.

    When the previous stage's output is exactly one 32-column block, its ``[M][1]`` scale write IS
    the ``[1][M]`` layout the A-side scale memory reads — same bytes, same order — so the stage reads
    the requantizer's buffer DIRECTLY and the seam moves nothing at all. Otherwise the driver has to
    transpose into a separate buffer (hw notes §2).
    """
    prev = plans[i - 1]
    return f"T{i - 1}_sc" if prev.scale_blocks == 1 and not prev.chain_code_shift else f"A{i}_scales"


def _c_array_2d(ctype: str, name: str, rows: Sequence[Sequence[int]], dims: str) -> str:
    body = ",\n".join("  {" + ",".join(str(int(v)) for v in row) + "}" for row in rows)
    return f"static const {ctype} {name}{dims} = {{\n{body}\n}};"


def _c_array_1d(ctype: str, name: str, vals: Sequence[int], dims: str) -> str:
    """A FLAT initializer. Not `_c_array_2d` with one row: that emits `{ {0,1,...} }`, which C reads
    as a braced initializer for element 0 alone — the rest silently zero-fill, and a table of zeros
    is a table that maps every operand to zero without ever faulting."""
    body = ",".join(str(int(v)) for v in vals)
    return f"static const {ctype} {name}{dims} = {{{body}}};"


def _emit_operand_data(p: MxGemmPlan, ops: dict[str, Sequence[Sequence[int]]],
                       nm: "_Names") -> list[str]:
    """Bake operands in, so the same bytes reach the reference, the simulator and the device.

    Only stage 0 bakes an A operand; a chained stage's A is written by the device at the seam.
    """
    lines = []
    if "a_codes" in ops:
        lines += [
            _c_array_2d("uint8_t", nm.a_codes, ops["a_codes"], f"[{p.m}][{p.k}]"),
            _c_array_2d("uint8_t", nm.b_codes, ops["b_codes"], f"[{p.k}][{p.n}]"),
            _c_array_2d("uint8_t", nm.a_scales, ops["a_scales"], f"[{p.scale_groups}][{p.m}]"),
            _c_array_2d("uint8_t", nm.b_scales, ops["b_scales"], f"[{p.scale_groups}][{p.n}]"),
        ]
    else:
        lines += [
            _c_array_2d("uint8_t", nm.b_codes, ops["b_codes"], f"[{p.k}][{p.n}]"),
            _c_array_2d("uint8_t", nm.b_scales, ops["b_scales"], f"[{p.scale_groups}][{p.n}]"),
        ]
    return lines


def _emit_mvin(p: MxGemmPlan, nm: _Names) -> list[str]:
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
        f"      const uint8_t *src = ((const uint8_t *){nm.a_codes}) + i * {d} * {p.k} + k * {d};",
        f"      uint32_t sp_addr = {p.a_base} + (i * {p.tiles_k} + k) * {d};",
        f"      gemmini_extended_mvin((void *)src, sp_addr, {d}, {d});",
        "    }",
        "  }",
        "",
        "  /* MVIN B[K][N]: row stride N; tile (k,j) -> b_base + (k*tiles_J + j)*DIM */",
        f"  gemmini_config_ld({p.n} * sizeof(uint8_t));",
        f"  for (int k = 0; k < {p.tiles_k}; k++) {{",
        f"    for (int j = 0; j < {p.tiles_j}; j++) {{",
        f"      const uint8_t *src = ((const uint8_t *){nm.b_codes}) + k * {d} * {p.n} + j * {d};",
        f"      uint32_t sp_addr = {p.b_base} + (k * {p.tiles_j} + j) * {d};",
        f"      gemmini_extended_mvin((void *)src, sp_addr, {d}, {d});",
        "    }",
        "  }",
    ]


def _emit_report(p: MxGemmPlan, nm: _Names, tail: list[str]) -> list[str]:
    """Print the shared merlin console protocol: ``OUT <name> <rows> <cols> v...`` / ``METRIC`` /
    ``DONE`` (``runtime/backends/base.parse_console``).

    BF16 values are reported as **bit patterns**; FP8 as raw **codes**, plus a second OUT line
    carrying the requantizer's per-row per-32-column E8M0 scale codes. Both are the datapath's
    native output, undecoded — nothing is lost on the way out.

    In a chain every stage reports, so the per-stage telemetry a host-carried run produced (peak
    code, E8M0 range) survives fusion. All of it runs AFTER the last cycle read, so no printf ever
    lands inside a measured window.
    """
    if p.out_fmt == "bf16":
        return [
            "  /* merlin console protocol — parsed by runtime.backends.base.parse_console. */",
            f'  printf("OUT {p.out} {p.m} {p.n}");',
            f"  for (int i = 0; i < {p.m}; i++)",
            f"    for (int j = 0; j < {p.n}; j++)",
            f"      printf(\" %u\", (unsigned)(({nm.out}[i][j / {BF16_PER_WORD}]"
            f" >> ((j % {BF16_PER_WORD}) * 16)) & 0xFFFF));",
            '  printf("\\n");',
        ] + tail
    return [
        "  /* merlin console protocol. FP8 requant output: the packed codes, then the E8M0 scale",
        "     codes the requantizer wrote to DRAM (one per row per 32 output columns). */",
        f"  {{ const uint8_t *codes = (const uint8_t *)&{nm.out}[0][0];",
        f'    printf("OUT {p.out} {p.m} {p.n}");',
        f"    for (long i = 0; i < {p.m} * {p.n}; i++) printf(\" %u\", (unsigned)codes[i]);",
        '    printf("\\n"); }',
        f"  {{ const uint8_t *sc = (const uint8_t *){nm.out_scales};",
        f'    printf("OUT {p.out}_scales {p.m} {p.scale_blocks}");',
        f"    for (long i = 0; i < {p.m} * {p.scale_blocks}; i++) printf(\" %u\", (unsigned)sc[i]);",
        '    printf("\\n"); }',
    ] + tail


def _emit_seam(i: int, plans: list[MxGemmPlan]) -> list[str]:
    """The handoff from stage *i-1* to stage *i*, on device. Kept as close to nothing as possible.

    The CODES need no work at all: ``MX_READ_SMEM`` already left them as contiguous ``[M][N]`` bytes,
    which is exactly the ``[M][K]`` layout the next ``mvin`` reads — the next stage points straight at
    that buffer. Two things can still force the driver to touch data, and both are hardware
    properties recorded in ``planning/chain_seam_hw_notes.md``:

    * §2 — the requantizer WRITES scales ``[M][K/32]`` while the mesh READS them ``[K/32][M]``.
      Degenerate when ``K == 32``: one block, so the two layouts are the same bytes and this emits
      nothing.
    * §1 — the requantizer's output exponent is hardwired to fill the format
      (``MxRequantizer.scala:35``), so a chained operand may need an exponent shift first. Zero on
      the `weight` seam, where the compensation was folded into the next weight on the host.
    """
    prev, cur = plans[i - 1], plans[i]
    a_scales, blocks, shift = _a_scales_name(i, plans), prev.scale_blocks, prev.chain_code_shift
    if a_scales == f"T{i - 1}_sc":
        return [
            f"  /* ---- seam {i - 1} -> {i}: NOTHING. K={cur.k} is one E8M0 block, so the",
            f"     requantizer's [{prev.m}][1] scale write IS the [1][{prev.m}] layout the A-side",
            "     scale memory reads — same bytes, same order. Codes need no fixup either. */",
        ]
    lines = [
        f"  /* ---- seam {i - 1} -> {i}. Codes: nothing (already [M][K] for mvin). Scales: the",
        f"     requantizer wrote [{prev.m}][{blocks}]; the A-side scale memory reads",
        f"     [{blocks}][{prev.m}] (gemmini.cc:1182). See chain_seam_hw_notes.md §2 — a",
        "     transpose-on-write bit in the requantizer removes this loop entirely. */",
        f"  seam_c0[{i}] = read_cycles();",
    ]
    # The source is contiguous, so walk it with a pointer and unroll the block axis away: it is a
    # compile-time constant and usually 2, which turns the whole seam into M iterations of
    # `blocks` load/stores with no index arithmetic at all.
    body: list[str]
    if blocks <= _SEAM_UNROLL_MAX:
        body = [f"  {{ const uint8_t *sp = T{i - 1}_sc;",
                f"    for (int m = 0; m < {prev.m}; m++) {{"]
        for g in range(blocks):
            body.append(f"      int v{g} = (int)*sp++ + {shift};" if shift else
                        f"      A{i}_scales[{g}][m] = *sp++;")
        if shift:
            for g in range(blocks):
                body.append(f"      A{i}_scales[{g}][m] = (uint8_t)(v{g} > 254 ? 254 : v{g});")
        body += ["    } }"]
    else:
        body = [f"  for (int g = 0; g < {blocks}; g++)",
                f"    for (int m = 0; m < {prev.m}; m++)"]
        body += ([f"    {{ int v = (int)T{i - 1}_sc[m * {blocks} + g] + {shift};",
                  f"      A{i}_scales[g][m] = (uint8_t)(v > 254 ? 254 : v); }}"] if shift else
                 [f"      A{i}_scales[g][m] = T{i - 1}_sc[m * {blocks} + g];"])
    lines += body
    if shift:
        lines += [
            f"  /* ...and the matching code shift. Dividing every code by 2^{shift} while adding",
            "     that exponent to the E8M0 byte above is VALUE-PRESERVING (e4m3 keeps 3 mantissa",
            "     bits at every exponent), but the mesh applies block scales only AFTER its 16-deep",
            "     column pass (gemmini.cc:1184), so the shift is what keeps that pass inside a 4-bit",
            "     accumulator exponent. This whole branch disappears with hw notes §1. */",
            f"  {{ uint8_t *c = (uint8_t *)&T{i - 1}[0][0];",
            f"    for (long t = 0; t < {prev.m} * {prev.n}; t++) c[t] = CODE_SHIFT{i - 1}[c[t]]; }}",
        ]
    return lines + [f"  seam_c1[{i}] = read_cycles();"]


def _emit_config_ex(p: MxGemmPlan) -> list[str]:
    return [
        "  /* CONFIG_EX: weight-stationary, no sys activation/shift, identity acc scale,",
        "     C_stride=1 A_stride=1, no transpose, then the four MX fields:",
        f"     act={p.act_fmt}({OPERAND_FMT[p.act_fmt]}) wgt={p.wgt_fmt}({OPERAND_FMT[p.wgt_fmt]}) "
        f"out={p.out_fmt}({OUTPUT_FMT[p.out_fmt]}) uselut={p.use_lut} */",
        "  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0,",
        f"                              false, {OPERAND_FMT[p.act_fmt]}, {OPERAND_FMT[p.wgt_fmt]}, "
        f"{OUTPUT_FMT[p.out_fmt]}, {p.use_lut});",
    ]


def _emit_mesh(p: MxGemmPlan, nm: _Names, tr: Transport) -> list[str]:
    """Store config, the requant/scale-memory config, and the LOOP_WS itself."""
    return [
        "  /* Output row stride, then the requant/scale-memory config (funct 26). On the requant",
        "     path the reference uses a single-word store stride; the drain is MX_READ_SMEM. */",
        ("  gemmini_config_st(OUT_COLS * sizeof(out_t));" if p.out_fmt == "bf16"
         else "  gemmini_config_st(1 * sizeof(out_t));"),
        f"  gemmini_mxquant_config_mvout((uint64_t){nm.out_scales}, {p.tiles_i}, {p.tiles_j}, "
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
    ]


def _emit_single(p: MxGemmPlan, ops: dict[str, Any], tr: Transport) -> str:
    """The one-matmul driver. Byte-for-byte what this module has always emitted."""
    nm = _Names("A_in", "A_scales_row", "B_in", "B_scales_col", "C_hw", "scale_factors")
    tail = [
        '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));',
        '  printf("METRIC cycle_window_mx_gemmini_region 1\\n");',
        '  printf("DONE\\n");',
    ]
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
        *_emit_operand_data(p, ops, nm),
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
        *_emit_config_ex(p),
        "",
        *tr.emit_load_scales(p, nm.a_scales, nm.b_scales),
        "",
        "  /* Accelerator region: operand move-in, the mesh loop, and the drain. */",
        "  c0 = read_cycles();",
        *_emit_mvin(p, nm),
        "",
        *_emit_mesh(p, nm, tr),
        "",
        *tr.emit_drain(p, nm.out),
        "  gemmini_fence();",
        "  c1 = read_cycles();",
        "",
        *_emit_report(p, nm, tail),
        "  return 0;",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _emit_chain(plans: list[MxGemmPlan], bundles: list[dict[str, Any]], tr: Transport) -> str:
    """The fused driver: every stage of a matmul chain in ONE program, ONE accelerator region.

    Intermediates never leave the device. What sits between two stages is :func:`_emit_seam`, which
    is one byte-transpose at most and frequently nothing at all.
    """
    n, last = len(plans), plans[-1]
    names = [_stage_names(i, plans) for i in range(n)]

    head = ["/* Generated by mxgemm_emit.py for target mx_gemmini_rocket — do not edit. */",
            f"/* CHAIN of {n} MX matmuls fused into one program   transport:{tr.name} */"]
    for i, p in enumerate(plans):
        head.append(f"/*   stage {i}  {p.m}x{p.n}x{p.k}  A:{p.act_fmt} B:{p.wgt_fmt} -> {p.out_fmt}"
                    f"   tiles {p.tiles_i}x{p.tiles_j}x{p.tiles_k}   smem@{p.spad_dest} */")

    # Baked data: stage 0's A, every stage's B, and a code-shift LUT per seam that needs one.
    data: list[str] = []
    for i, (p, nm) in enumerate(zip(plans, names)):
        data += _emit_operand_data(p, bundles[i], nm)
        if p.chain_code_shift:
            lut = bundles[i].get("chain_code_lut")
            if not lut or len(lut) != 256:
                raise MxEmitError(
                    f"stage {i} declares {CHAIN_CODE_SHIFT_ATTR}={p.chain_code_shift} but supplies "
                    "no 256-entry 'chain_code_lut'. The backend will not reimplement the element "
                    "format's encode/decode — the app owns that (app/mxquant.py, transcribed from "
                    "mx_fp_math.h).")
            data.append(_c_array_1d("uint8_t", f"CODE_SHIFT{i}", lut, "[256]"))

    # Buffers. An intermediate's codes land in a plain [M][N] uint8 array, which IS the [M][K]
    # layout the next stage's mvin wants; its scales land flat, as the requantizer writes them.
    buf = [f"  static out_t C_hw[{last.m}][OUT_COLS];",
           "  /* Requant scale write-back target for the final stage. Unused on the BF16 output",
           "     path, but the mxquant config still needs a valid address to point at. */",
           f"  static uint32_t scale_factors[{max(512, -(-last.m * last.scale_blocks // 4))}];"]
    clear = ["  memset(C_hw, 0, sizeof(C_hw));", "  memset(scale_factors, 0, sizeof(scale_factors));"]
    for i, p in enumerate(plans[:-1]):
        buf += [f"  static uint8_t T{i}[{p.m}][{p.n}];",
                f"  static uint8_t T{i}_sc[{p.m} * {p.scale_blocks}];"]
        clear += [f"  memset(T{i}, 0, sizeof(T{i}));",
                  f"  memset(T{i}_sc, 0, sizeof(T{i}_sc));"]
        if names[i + 1].a_scales == f"A{i + 1}_scales":
            buf.append(f"  static uint8_t A{i + 1}_scales[{p.scale_blocks}][{p.m}];")

    body: list[str] = []
    for i, (p, nm) in enumerate(zip(plans, names)):
        if i:
            body += _emit_seam(i, plans) + [""]
        body += [f"  /* ---- stage {i}: {p.m}x{p.n}x{p.k} -> {p.out_fmt} ---- */",
                 f"  st_c0[{i}] = read_cycles();",
                 *_emit_config_ex(p),
                 *tr.emit_load_scales(p, nm.a_scales, nm.b_scales),
                 *_emit_mvin(p, nm),
                 *_emit_mesh(p, nm, tr),
                 *tr.emit_drain(p, nm.out),
                 "  gemmini_fence();",
                 f"  st_c1[{i}] = read_cycles();",
                 ""]

    # Every stage reports, so fusing loses none of the per-stage telemetry. All of it runs after the
    # last cycle read; the tail carries the per-stage and per-seam windows so the seam's cost is
    # MEASURED rather than assumed (that is the number chain_seam_hw_notes.md wants driven to zero).
    tail = ['  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));']
    for i in range(n):
        tail.append(f'  printf("METRIC cycles_stage{i} %lu\\n", '
                    f"(unsigned long)(st_c1[{i}] - st_c0[{i}]));")
    for i in range(1, n):
        tail.append(f'  printf("METRIC seam_cycles_stage{i} %lu\\n", '
                    f"(unsigned long)(seam_c1[{i}] - seam_c0[{i}]));")
    tail += ['  printf("METRIC cycle_window_mx_gemmini_region 1\\n");',
             '  printf("DONE\\n");']

    report: list[str] = []
    for i, (p, nm) in enumerate(zip(plans, names)):
        report += _emit_report(p, nm, tail if i == n - 1 else [])

    return "\n".join([
        *head,
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <string.h>",
        *tr.includes(),
        "",
        f"#define OUT_COLS {last.out_cols}",
        "typedef uint64_t out_t;",
        "",
        *data,
        "",
        "int main(void) {",
        "  uint64_t c0, c1;",
        f"  uint64_t st_c0[{n}] = {{0}}, st_c1[{n}] = {{0}};",
        f"  uint64_t seam_c0[{n}] = {{0}}, seam_c1[{n}] = {{0}};",
        *buf,
        *clear,
        "",
        "  gemmini_flush(0);",
        "",
        "  /* Accelerator region: EVERY stage and every seam between them. Nothing returns to the",
        "     host until the last fence, so this window is the whole chain. */",
        "  c0 = read_cycles();",
        "",
        *body,
        "  c1 = read_cycles();",
        "",
        *report,
        "  return 0;",
        "}",
    ]) + "\n"


def generate_driver(cb: dict[str, Any], *, transport: Transport | None = None) -> str:
    """Return the C source of the bare-metal MX GEMM driver for ``cb``.

    One matmul or a chain of them; the length-1 case emits exactly what it always has.
    """
    plans = plan_chain(cb)
    bundles = _mx_operands(cb, plans)
    tr = transport or SpikeSmemTransport()
    return (_emit_single(plans[0], bundles[0], tr) if len(plans) == 1
            else _emit_chain(plans, bundles, tr))
