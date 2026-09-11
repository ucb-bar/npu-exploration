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
# The format table lives in ONE place, `app/mxformats.py`, because the same facts are needed by the
# quantizer and (eventually) by hardware generation. The backend reads it rather than restating it.
# Imported lazily-ish: the app package is on sys.path in every context this backend runs in, and a
# local fallback would be a second source of truth, which is exactly what the table exists to stop.
from app import mxformats as _fmt                                          # noqa: E402

#: The output-only pseudo-format: the raw bf16 accumulator readout.
BF16 = "bf16"
# One E8M0 exponent per 32 K-elements (MxRequantizer.scala).
BLOCK_SCALE_GROUP = _fmt.BLOCK
# BF16 results drain packed 4 per uint64 word.
BF16_PER_WORD = 4

# [RTL] geometry defaults (gemmini_params.h DIM/BANK_NUM/BANK_ROWS). Overridable per buffer via
# cb["params"] so a re-elaborated mesh needs no code edit.
DEFAULT_GEOMETRY = {"dim": 16, "bank_num": 4, "bank_rows": 4096, "spad_dest": 128, "addr_len": 32}

#: [RTL] LOOP_WS rs2 bit 10 -- deposit the requantized output in the BLOCK-TILED operand-A layout
#: instead of flat row-major, so the next stage reads it in place as its A operand. Spike:
#: `gemmini.cc:1157` `mx_loop_reuse_tiled`, whose tiled address is documented at `:1199` as
#: "identical to the operand-A read". Only meaningful for a requant format; BF16 output stays flat
#: (`:1201` gates on `mx_out_fmt != 3`).
LOOP_WS_REQUANT_TILED = 1 << 10

#: [RTL] CONFIG_SCALE_MEM rs1 bit 63 -- MX_SCALE_RESIDENT. The requantizer also writes this stage's
#: output block-scales into the on-chip ACT-scale window, already transposed to [GN][M]
#: (`gemmini.cc:1325`: `a_off_out = bi*M + m`), so the next stage reads its A-scales in place.
#: Emitted by `gemmini_mxquant_config_mvout_resident`. NOTE `gemmini.h:247` calls it "bit 62" in
#: prose while its own macro shifts by 63; spike decodes 63, so the macro is right and the comment
#: is wrong.
MX_SCALE_RESIDENT_BIT = 63


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
    #: This stage's A operand is the previous stage's requantizer output, still resident in the
    #: scratchpad in the tiled operand layout. It is neither moved in nor scale-loaded.
    chained_in: bool = False
    #: The next stage consumes this stage's output in place: commit it TILED and make its block
    #: scales resident.
    chained_out: bool = False
    #: Scratchpad row where this stage's A operand lives. 0 for stage 0 (freshly moved in); the
    #: previous stage's `spad_dest` when :attr:`chained_in`.
    a_spad: int = 0
    #: Codebook granularity: one 16-entry LUT per 2**lut_g rows/columns. Also the last argument to
    #: gemmini_mxquant_config_mvout.
    lut_g: int = _fmt.LUT_GRANULARITY

    @property
    def fmt_a(self):
        return _fmt.FORMATS[self.act_fmt]

    @property
    def fmt_b(self):
        return _fmt.FORMATS[self.wgt_fmt]

    @property
    def tiles_i(self) -> int:
        """Mesh tiles down M. The nibble formats march 32 rows at a time, not 16
        (``gemmini.cc:1519`` ``const int TM = 32, TN = 32``), so this is format-dependent."""
        return self.m // self.fmt_a.tile_m

    @property
    def tiles_j(self) -> int:
        return self.n // self.fmt_b.tile_n

    @property
    def tiles_k(self) -> int:
        return self.k // self.fmt_a.tile_k

    @property
    def scale_groups(self) -> int:
        """E8M0 groups along K — one scale byte per operand row/col per group."""
        return self.k // BLOCK_SCALE_GROUP

    @property
    def spad_rows(self) -> int:
        return self.bank_num * self.bank_rows

    @property
    def a_base(self) -> int:
        """Where the mesh reads operand A. For a chained stage this IS the previous stage's output
        region -- the whole point of the resident chain."""
        return self.a_spad

    @property
    def b_base(self) -> int:
        """B tiles are laid out from the END of the scratchpad, growing down."""
        return self.spad_rows - self.tiles_k * self.tiles_j * self.dim

    @property
    def a_row_bytes(self) -> int:
        """Bytes per row of the PACKED A operand.

        A packs two m-rows per byte for a 4-bit format (``gemmini.cc:1536``:
        ``spad[A_t + (m >> 1)][kk]``), so the array is ``[M/2][K]`` and a row is still K bytes --
        the packing halves the ROW COUNT, not the row length."""
        return self.k

    @property
    def b_row_bytes(self) -> int:
        """Bytes per row of the PACKED B operand. B packs two n-columns per byte
        (``gemmini.cc:1541``: ``spad[B_t + kk][n >> 1]``), so ``[K][N/2]`` and a row is N/2."""
        return self.n // self.fmt_b.packed_per_byte

    @property
    def use_lut(self) -> int:
        """CONFIG_EX rs1[5]. Set when either operand is codebook-indexed."""
        return int(self.fmt_a.lut or self.fmt_b.lut)

    @property
    def altfmt(self) -> int:
        """CONFIG_EX rs1[6]. Selects the alternate format within a code (E5M2 / E2M3)."""
        a, b = self.fmt_a.altfmt, self.fmt_b.altfmt
        if a != b:
            raise MxEmitError(
                f"operands disagree on altfmt: {self.fmt_a.name}={a} vs {self.fmt_b.name}={b}. "
                "rs1[6] is ONE bit shared by both sides, so a mixed pair is not expressible.")
        return a

    @property
    def out_cols(self) -> int:
        """Width of the packed output row, in uint64 words. The output is [M][N], so this is N —
        NOT M. (Identical when M == N, which is why the reference could use either.)

        BF16 packs 4 per uint64; FP8 codes pack 8.
        """
        per_word = BF16_PER_WORD if self.out_fmt == "bf16" else 2 * BF16_PER_WORD
        return self.n // per_word

    @property
    def out_bits(self) -> int:
        """Wire width of ONE output element. BF16 is 16; a requant output is the element format's
        own width, which is 4 for every nibble format -- so bytes are not a safe unit here."""
        return 16 if self.out_fmt == BF16 else _fmt.FORMATS[self.out_fmt].bits

    @property
    def out_bytes(self) -> int:
        """Total bytes of this stage's output on the wire."""
        return self.m * self.n * self.out_bits // 8

    @property
    def out_byte_rows(self) -> int:
        """Rows of the output viewed as a BYTE array.

        A requant commit packs like operand A -- two m-rows per byte for a nibble format
        (``gemmini.cc:1201-1210``: "hw = m for FP8, m/2 for the nibble formats, exactly as operand A
        is packed") -- so the byte array is ``[M / packed_per_byte][N]``, NOT ``[M][N/2]``. The
        reference reads it back with exactly these dims
        (``matmul_tiled_fp4_64x64_chain.c:151``: ``mvout_detile(C1_hw, SPAD_DEST1, M/2, N)``).
        """
        if self.out_fmt == BF16:
            return self.m
        return self.m // _fmt.FORMATS[self.out_fmt].packed_per_byte

    @property
    def out_byte_cols(self) -> int:
        return self.n

    @property
    def out_elem_bytes(self) -> int:
        if self.out_bits % 8:
            raise MxEmitError(
                f"{self.out_fmt} packs {self.out_bits} bits per element, so an element is not a "
                "whole number of bytes; use out_bytes")
        return self.out_bits // 8

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
    if dtype not in _fmt.ALIASES:
        raise MxEmitError(
            f"{where} dtype {dtype!r} is not a microscaling operand format "
            f"(expected one of {sorted(_fmt.ALIASES)})")
    return _fmt.get(dtype, where=where).name


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
        if p.out_fmt != BF16 and _fmt.FORMATS[p.out_fmt].lut:
            # A codebook chain couples two stages: the requantizer writes indices into the OUTPUT
            # codebook (sel 2) and the next stage reads them as ACTIVATION indices (sel 1), so
            # stage i+1's A codebook must BE stage i's C codebook. The app supplies both; this only
            # checks it, because a mismatch would decode every intermediate against the wrong table
            # and the result would look plausible.
            pass
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
    out_fmt = BF16 if out_dtype == BF16 else (
        _fmt.get(out_dtype, where=f"commit {out!r}").name if out_dtype in _fmt.ALIASES else None)
    if out_fmt is None:
        raise MxEmitError(
            f"output dtype {out_dtype!r} is neither 'bf16' nor a microscaling format "
            f"({sorted(_fmt.ALIASES)})")

    plan = MxGemmPlan(
        m=m, n=n, k=k_a,
        act_fmt=_dtype_to_fmt(dtype[lhs], f"lhs {lhs!r}"),
        wgt_fmt=_dtype_to_fmt(dtype[weight], f"weight {weight!r}"),
        out_fmt=out_fmt,
        lhs=lhs, weight=weight, out=out,
        **geom)
    _validate(plan)
    shape[out], dtype[out] = (m, n), out_dtype
    return plan


def _assign_smem(plans: list[MxGemmPlan]) -> None:
    """Give every stage a DISJOINT MX shared-memory region and wire the residency chain, in place.

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
    dests: list[int] = []
    for i, p in enumerate(plans):
        if dest + p.smem_rows > budget:
            raise MxEmitError(
                f"MX shared memory exhausted at stage {i}: stages need "
                f"{dest + p.smem_rows} rows of {budget}. Chained stages must not share a region "
                "(gemmini.cc:1190 accumulates); shrink the chain or the intermediates.")
        dests.append(dest)
        dest += p.smem_rows

    # A stage's output region must also stay clear of the B tiles, which grow DOWN from the top of
    # the scratchpad and are rewritten every stage. Checked rather than assumed: an overlap would
    # not fault, it would return a plausible wrong answer.
    b_low = min(p.b_base for p in plans)
    if dest > b_low:
        raise MxEmitError(
            f"chain output regions reach row {dest}, overlapping the B tiles at {b_low}. "
            "Shrink the chain, the intermediates, or the weights.")

    for i, p in enumerate(plans):
        plans[i] = replace(
            p,
            spad_dest=dests[i],
            chained_in=i > 0,
            chained_out=i < len(plans) - 1,
            # THE resident chain: stage i reads operand A straight out of stage i-1's output
            # region. The tiled commit layout is byte-identical to the operand-A read
            # (gemmini.cc:1199), so no copy, no transpose, no DRAM.
            a_spad=dests[i - 1] if i > 0 else 0)


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
    # Tile granularity is FORMAT-dependent: the nibble formats march 32x32, not 16x16
    # (gemmini.cc:1519). A shape legal for FP8 is not automatically legal for FP4.
    for name, val, tile, why in (("M", p.m, p.fmt_a.tile_m, p.fmt_a.name),
                                 ("N", p.n, p.fmt_b.tile_n, p.fmt_b.name),
                                 ("K", p.k, p.fmt_a.tile_k, p.fmt_a.name)):
        if val % tile:
            raise MxEmitError(
                f"{name}={val} is not a whole multiple of the {why} mesh tile ({tile})")
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
        # Codes are checked against the PACKED shape -- a 4-bit A is [M/2][K] and a 4-bit B is
        # [K][N/2] (gemmini.cc:1536,1541). Scales never pack: one byte per row/col per K-group.
        want = {"b_codes": (p.k, p.n // p.fmt_b.packed_per_byte),
                "b_scales": (p.scale_groups, p.n)}
        if i == 0:
            want |= {"a_codes": (p.m // p.fmt_a.packed_per_byte, p.k),
                     "a_scales": (p.scale_groups, p.m)}
        for key, (rows, cols) in want.items():
            if key not in bundle:
                raise MxEmitError(f"mx_operands[{i}] missing {key!r}")
            got = (len(bundle[key]), len(bundle[key][0]) if bundle[key] else 0)
            if got != (rows, cols):
                raise MxEmitError(f"mx_operands[{i}][{key!r}] is {got}, expected {(rows, cols)}")
        if p.fmt_a.lut or p.fmt_b.lut:
            words = _fmt.lut_words(p.fmt_a.entry_bits)
            for key, n_lut in (("a_lut", p.m >> p.lut_g), ("b_lut", p.n >> p.lut_g),
                               ("c_lut", p.m >> p.lut_g)):
                if key not in bundle:
                    continue                       # _emit_operand_data raises with the better message
                got = (len(bundle[key]), len(bundle[key][0]) if len(bundle[key]) else 0)
                if got != (n_lut, words):
                    raise MxEmitError(
                        f"mx_operands[{i}][{key!r}] is {got}, expected {(n_lut, words)} "
                        f"({p.fmt_a.entry_bits}-bit entries, G={p.lut_g})")
        if i and p.fmt_a.lut:
            prev_c = bundles[i - 1].get("c_lut")
            mine_a = bundle.get("a_lut")
            if prev_c is None or mine_a is None:
                raise MxEmitError(
                    f"stage {i} chains a codebook format but the codebooks are missing: stage "
                    f"{i - 1} must supply 'c_lut' and stage {i} must supply 'a_lut'")
            if [list(r) for r in prev_c] != [list(r) for r in mine_a]:
                raise MxEmitError(
                    f"stage {i}'s A codebook differs from stage {i - 1}'s C codebook. The "
                    "requantizer writes indices with C and the next stage reads them with A, so "
                    "they MUST be the same table -- otherwise index 7 means one value on the way "
                    "in and another on the way out, and the result is plausible but wrong.")
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
class RoccSpadTransport:
    """The real-RoCC endpoint: one instruction stream for BOTH spike and the standalone RTL.

    Scales are DMA'd from DRAM by MX_LOAD_SCALES (funct 27), and the result is drained from the
    internal scratchpad by a flat contiguous MVOUT (funct 3) -- the sequence every hand-written test
    in ``gemmini-rocc-tests/bareMetalC`` uses, and which they describe as an "identical instruction
    stream on Spike and RTL".

    This replaced a drain via MX_READ_SMEM (funct 28), which works on spike and **cannot work on the
    RTL**: ``GemminiISA.scala:45`` names the opcode and no module in the generator decodes it. An ELF
    drained that way computes the right answer on hardware and then has no way to read it back.
    """

    name: str = "rocc_spad"

    def includes(self) -> list[str]:
        return ['#include "include/gemmini_testutils.h"']

    def emit_load_scales(self, p: MxGemmPlan,
                         a_scales: str = "A_scales_row", b_scales: str = "B_scales_col") -> list[str]:
        b = [f"  gemmini_mx_load_scales((uint64_t)&{b_scales}, sizeof({b_scales}), 1);"]
        # The fence is REQUIRED, and its absence is invisible here: MX_LOAD_SCALES is an
        # asynchronous DMA on the RTL and instantaneous under spike, so without it the mesh can
        # start reading scales that have not landed -- on hardware only. Every reference test
        # fences here for exactly this reason (matmul_tiled_fp8_64x64.c:84: "The fence orders the
        # async scale DMA on the RTL and is a no-op on Spike, so both emit the identical
        # instruction stream").
        fence = ["  gemmini_fence();"]
        if p.chained_in:
            return [
                "  /* A-scales are RESIDENT: the previous stage's requantizer wrote them straight",
                "     into the act-scale window, already transposed to [GN][M] (MX_SCALE_RESIDENT,",
                "     gemmini.cc:1325). Only the B side is loaded. */",
            ] + b + fence
        return [
            "  /* E8M0 block scales: DRAM -> mx_scale_{a,b}_mem. sel 0 = A rows, 1 = B cols. */",
            f"  gemmini_mx_load_scales((uint64_t)&{a_scales}, sizeof({a_scales}), 0);",
        ] + b + fence

    def out_dest(self, p: MxGemmPlan) -> str:
        return str(p.spad_dest)

    def out_flag(self) -> str:
        # Loop-FSM skip mask; keeps the spad store the drain reads back.
        return "0x38"

    def emit_drain(self, p: MxGemmPlan, dst: str = "C_hw") -> list[str]:
        """Flat scratchpad -> DRAM readback, transcribed from the reference tests.

        ``matmul_tiled_fp8_64x64.c:146-160`` (BF16) and ``..._requant.c:170-176`` (FP8) are the same
        loop; only the row count differs, because one spad row holds DIM bytes and an element is
        ``out_elem_bytes`` of them. Contiguous (``config_st`` = DIM), never strided: a strided mvout
        makes the writer DMA emit whole 64-byte cache lines and zero-fill the gaps on RTL, which
        corrupts the readback at N >= 128.
        """
        rows = p.out_bytes // p.dim
        return [
            f"  /* Drain: {p.out_fmt} in the internal scratchpad -> DRAM, {p.out_bits} bit(s)"
            f"/elem => {p.out_bytes} bytes / {p.dim} = {rows} spad rows.",
            "     Flat contiguous MVOUT (funct 3) -- identical instruction stream on Spike and RTL. */",
            "  gemmini_fence();",
            f"  gemmini_config_st({p.dim} * sizeof(uint8_t));",
            f"  {{ uint8_t *c_base = (uint8_t *)&{dst}[0][0];",
            f"    for (int r = 0; r < {rows}; r += {p.dim})",
            f"      gemmini_extended_mvout(c_base + r * {p.dim}, {p.spad_dest} + r, {p.dim}, {p.dim}); }}",
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
    a_lut: str = ""          # codebooks, LUT formats only ("" when the format is direct)
    b_lut: str = ""
    c_lut: str = ""


def _stage_names(i: int, plans: list[MxGemmPlan]) -> _Names:
    """C identifiers for stage *i*.

    A chained stage names NO A buffers: its codes are resident in the scratchpad and its scales are
    resident in the act-scale window, so there is nothing on the host to name. The empty strings are
    load-bearing — :func:`_emit_mvin` and the scale load key off them, and any code that tried to
    read one would fail to compile rather than silently reading a stale buffer.
    """
    last = i == len(plans) - 1
    p = plans[i]
    lut = ("A_lut", "B_lut", "C_lut") if (p.fmt_a.lut or p.fmt_b.lut) else ("", "", "")
    if len(plans) == 1:
        return _Names("A_in", "A_scales_row", "B_in", "B_scales_col", "C_hw", "scale_factors", *lut)
    chained = p.chained_in
    lut = tuple(f"{n}{i}" if n else "" for n in lut)
    return _Names(
        a_codes="" if chained else "A_in",
        a_scales="" if chained else "A_scales_row",
        b_codes=f"B{i}_in", b_scales=f"B{i}_scales_col",
        out="C_hw" if last else f"T{i}",
        out_scales="scale_factors" if last else f"T{i}_sc",
        a_lut=lut[0], b_lut=lut[1], c_lut=lut[2])



def _c_array_2d(ctype: str, name: str, rows: Sequence[Sequence[int]], dims: str) -> str:
    body = ",\n".join("  {" + ",".join(str(int(v)) for v in row) + "}" for row in rows)
    return f"static const {ctype} {name}{dims} = {{\n{body}\n}};"



def _emit_operand_data(p: MxGemmPlan, ops: dict[str, Sequence[Sequence[int]]],
                       nm: "_Names") -> list[str]:
    """Bake operands in, so the same bytes reach the reference, the simulator and the device.

    Only stage 0 bakes an A operand; a chained stage's A is written by the device at the seam.
    """
    # PACKED dimensions: a 4-bit A is [M/2][K] and a 4-bit B is [K][N/2]. Scales are one byte per
    # row/column per K-group whatever the element width, so they never pack.
    a_rows = p.m // p.fmt_a.packed_per_byte
    b_cols = p.n // p.fmt_b.packed_per_byte
    lines = []
    # Codebooks, for a LUT format. 16 entries of `entry_bits`, LE-packed into uint32 words; one
    # codebook per 2**G rows of A / columns of B (gemmini.cc:1392). They are DATA -- built from the
    # tensors by the app, exactly like the operand codes -- so they ride the same side channel.
    if nm.a_lut:
        words = _fmt.lut_words(p.fmt_a.entry_bits)
        for key, name, n_lut in (("a_lut", nm.a_lut, p.m >> p.lut_g),
                                 ("b_lut", nm.b_lut, p.n >> p.lut_g),
                                 ("c_lut", nm.c_lut, p.m >> p.lut_g)):
            if key not in ops:
                raise MxEmitError(
                    f"{p.act_fmt} is codebook-indexed but mx_operands has no {key!r}. The backend "
                    "will not invent a codebook: it is derived from the tensor values (k-means over "
                    "the quantized data), which the app owns.")
            lines.append(_c_array_2d("uint32_t", name, ops[key], f"[{n_lut}][{words}]"))
    if "a_codes" in ops:
        lines += [
            _c_array_2d("uint8_t", nm.a_codes, ops["a_codes"], f"[{a_rows}][{p.k}]"),
            _c_array_2d("uint8_t", nm.b_codes, ops["b_codes"], f"[{p.k}][{b_cols}]"),
            _c_array_2d("uint8_t", nm.a_scales, ops["a_scales"], f"[{p.scale_groups}][{p.m}]"),
            _c_array_2d("uint8_t", nm.b_scales, ops["b_scales"], f"[{p.scale_groups}][{p.n}]"),
        ]
    else:
        lines += [
            _c_array_2d("uint8_t", nm.b_codes, ops["b_codes"], f"[{p.k}][{b_cols}]"),
            _c_array_2d("uint8_t", nm.b_scales, ops["b_scales"], f"[{p.scale_groups}][{p.n}]"),
        ]
    return lines


def _emit_load_luts(p: MxGemmPlan, nm: _Names) -> list[str]:
    """Load the three codebooks (funct 29), for a LUT-indexed format.

    ``sel`` is 0 = B (weight), 1 = A (activation), 2 = C (output); ``entry_bits`` is the codebook's
    element width, 6 for FP6 and 8 for the FP8 codebooks (``gemmini.h:68-76``). Order and arguments
    mirror ``matmul_tiled_fp6_e2m3_lut_64x64.c:67-69``.

    MX_LOAD_LUT also SETS the runtime ``lut_en`` flag, which is what promotes E4M3 from the direct
    8-bit path to the 4-wide quad path (``gemmini.cc:1120``, ``:1366``). The flag is sticky until
    MX_LUT_DISABLE (funct 30) or a reset, so a program that mixed a LUT format with plain E4M3 would
    have to clear it between them. That cannot arise today -- a chain is single-dtype and a chained
    nibble commit is refused -- but it is why the funct exists.
    """
    if not nm.a_lut:
        return []
    e = p.fmt_a.entry_bits
    return [
        f"  /* Codebooks: 16 entries x {e} bits, one per {1 << p.lut_g} row(s)/col(s) (G={p.lut_g}).",
        "     sel: 0 = B (weight), 1 = A (activation), 2 = C (output). */",
        f"  gemmini_mx_load_lut_dt((uint64_t)&{nm.b_lut}[0][0], {p.n >> p.lut_g}, 0, {e});",
        f"  gemmini_mx_load_lut_dt((uint64_t)&{nm.a_lut}[0][0], {p.m >> p.lut_g}, 1, {e});",
        f"  gemmini_mx_load_lut_dt((uint64_t)&{nm.c_lut}[0][0], {p.m >> p.lut_g}, 2, {e});",
    ]


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
    a_lines = [
        f"  /* Operand A is RESIDENT at spad row {p.a_base}: the previous stage committed it there",
        "     in the tiled operand-A layout (LOOP_WS rs2 bit 10), so there is no mvin and no DRAM",
        "     traffic. gemmini.cc:1199 -- the tiled commit address IS the operand-A read address. */",
    ] if p.chained_in else [
        f"  /* MVIN A: packed [{p.m // p.fmt_a.packed_per_byte}][{p.k}] "
        f"({p.fmt_a.packed_per_byte} m-row(s)/byte), row stride {p.a_row_bytes} B;",
        "     tile (i,k) -> a_base + (i*tiles_K + k)*DIM */",
        f"  gemmini_config_ld({p.a_row_bytes} * sizeof(uint8_t));",
        f"  for (int i = 0; i < {p.tiles_i}; i++) {{",
        f"    for (int k = 0; k < {p.tiles_k}; k++) {{",
        f"      const uint8_t *src = ((const uint8_t *){nm.a_codes}) "
        f"+ i * {d} * {p.a_row_bytes} + k * {d};",
        f"      uint32_t sp_addr = {p.a_base} + (i * {p.tiles_k} + k) * {d};",
        f"      gemmini_extended_mvin((void *)src, sp_addr, {d}, {d});",
        "    }",
        "  }",
    ]
    return a_lines + [
        "",
        f"  /* MVIN B: packed [{p.k}][{p.n // p.fmt_b.packed_per_byte}] "
        f"({p.fmt_b.packed_per_byte} n-col(s)/byte), row stride {p.b_row_bytes} B;",
        "     tile (k,j) -> b_base + (k*tiles_J + j)*DIM */",
        f"  gemmini_config_ld({p.b_row_bytes} * sizeof(uint8_t));",
        f"  for (int k = 0; k < {p.tiles_k}; k++) {{",
        f"    for (int j = 0; j < {p.tiles_j}; j++) {{",
        f"      const uint8_t *src = ((const uint8_t *){nm.b_codes}) "
        f"+ k * {d} * {p.b_row_bytes} + j * {d};",
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
        f'    printf("OUT {p.out} {p.out_byte_rows} {p.out_byte_cols}");',
        f"    for (long i = 0; i < {p.out_bytes}; i++) printf(\" %u\", (unsigned)codes[i]);",
        '    printf("\\n"); }',
        f"  {{ const uint8_t *sc = (const uint8_t *){nm.out_scales};",
        f'    printf("OUT {p.out}_scales {p.m} {p.scale_blocks}");',
        f"    for (long i = 0; i < {p.m} * {p.scale_blocks}; i++) printf(\" %u\", (unsigned)sc[i]);",
        '    printf("\\n"); }',
    ] + tail



def _emit_config_ex(p: MxGemmPlan) -> list[str]:
    """CONFIG_EX, including the MX format selector.

    The selector is THREE things, not one enum (``gemmini.cc:460-464`` and the decode at
    ``:1222``/``:1356``): the 2-bit format code per side, the shared ``altfmt`` bit rs1[6], and the
    runtime ``lut_en`` flag that MX_LOAD_LUT sets. ``uselut`` (rs1[5]) is the static half of that.

    ``gemmini_extended3_config_ex`` has no altfmt parameter, so when it is set the instruction is
    emitted longhand -- exactly as ``matmul_tiled_fp8_e5m2_64x64.c:47-64`` does.
    """
    out_code = _fmt.BF16_FMT_CODE if p.out_fmt == BF16 else _fmt.FORMATS[p.out_fmt].fmt_code
    a, b = p.fmt_a, p.fmt_b
    head = [
        "  /* CONFIG_EX: weight-stationary, no sys activation/shift, identity acc scale,",
        "     C_stride=1 A_stride=1, no transpose, then the MX selector:",
        f"     act={a.name}(code {a.fmt_code}) wgt={b.name}(code {b.fmt_code}) "
        f"out={p.out_fmt}(code {out_code}) altfmt={p.altfmt} uselut={p.use_lut} */",
    ]
    if not p.altfmt:
        return head + [
            "  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0,",
            f"                              false, {a.fmt_code}, {b.fmt_code}, "
            f"{out_code}, {p.use_lut});",
        ]
    return head + [
        "  /* altfmt (rs1[6]) has no macro parameter, so this is the longhand form. */",
        "  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,",
        "      ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)ACC_SCALE_IDENTITY) << 32)",
        "    | ((uint64_t)(1) << 16)",
        f"    | ((uint64_t)({out_code}) << 14)",
        f"    | ((uint64_t)({b.fmt_code}) << 12)",
        f"    | ((uint64_t)({a.fmt_code}) << 10)",
        f"    | ((uint64_t)({p.altfmt}) << 6)",
        f"    | ((uint64_t)({p.use_lut}) << 5)",
        "    | ((uint64_t)(WEIGHT_STATIONARY) << 2)",
        "    | CONFIG_EX,",
        "      ((uint64_t)(1) << 48) | (0),",
        "      k_CONFIG);",
    ]


def _emit_mesh(p: MxGemmPlan, nm: _Names, tr: Transport) -> list[str]:
    """Store config, the requant/scale-memory config, and the LOOP_WS itself.

    Two knobs make a chain resident, and both are set on the PRODUCING stage:

    * ``gemmini_mxquant_config_mvout_resident`` (rs1 bit 63) additionally writes the output block
      scales into the on-chip act-scale window, transposed to the layout the next stage reads.
      The DRAM address is still honoured, so the host telemetry costs nothing extra.
    * ``LOOP_WS_REQUANT_TILED`` (rs2 bit 10) commits the codes in the tiled operand-A layout.

    Together they are the whole seam. There is no host work between two stages at all.
    """
    cfg = ("gemmini_mxquant_config_mvout_resident" if p.chained_out
           else "gemmini_mxquant_config_mvout")
    flag = tr.out_flag()
    if p.chained_out:
        flag = f"{flag} | {hex(LOOP_WS_REQUANT_TILED)}"
    return [
        "  /* Output row stride, then the requant/scale-memory config (funct 26). On the requant",
        "     path the reference uses a single-word store stride. */",
        ("  gemmini_config_st(OUT_COLS * sizeof(out_t));" if p.out_fmt == "bf16"
         else "  gemmini_config_st(1 * sizeof(out_t));"),
        *([
            "  /* MX_SCALE_RESIDENT: also write this stage's output scales into the act-scale",
            "     window as the next stage's A-scales -- no DRAM round trip, no host transpose. */",
        ] if p.chained_out else []),
        f"  {cfg}((uint64_t){nm.out_scales}, {p.tiles_i}, {p.tiles_j}, "
        f"{p.tiles_k}, 0, 0, {p.lut_g});",
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
        f"                       {flag});",
    ]


def _emit_single(p: MxGemmPlan, ops: dict[str, Any], tr: Transport) -> str:
    """The one-matmul driver. Byte-for-byte what this module has always emitted."""
    # Via _stage_names, not a literal: the single-matmul driver must pick up every name the chain
    # driver does (the codebooks, most recently), and a second hand-written list silently did not.
    nm = _stage_names(0, [p])
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
        *_emit_load_luts(p, nm),
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


#: Read a BLOCK-TILED resident intermediate back to a flat ``[M][N]`` host buffer. Used ONLY for
#: telemetry, after the measured region has closed -- the chain itself never reads an intermediate.
#:
#: The de-tile is done in software after a CONTIGUOUS mvout, not by a strided mvout. A strided
#: de-tile (16-byte rows N bytes apart) makes the writer DMA emit whole 64-byte cache lines and
#: zero-fill the gaps on RTL, which corrupts the readback at N >= 128 -- rows come back as zero.
#: `matmul_tiled_fp8_64x64_chain.c:82` records the same finding.
_DETILE_HELPER = """\
/* Read a tiled resident intermediate back to a flat [M][N] buffer (telemetry only). */
static void mvout_detile(uint8_t *dst, uint32_t spad_base, int M, int N, uint8_t *tmp) {
  const int tiles_I = M / %(dim)d, tiles_N = N / %(dim)d, total_rows = M * N / %(dim)d;
  gemmini_config_st(%(dim)d * sizeof(uint8_t));
  for (int r = 0; r < total_rows; r += %(dim)d)
    gemmini_extended_mvout(tmp + r * %(dim)d, spad_base + r, %(dim)d, %(dim)d);
  gemmini_fence();
  for (int i = 0; i < tiles_I; i++)
    for (int nt = 0; nt < tiles_N; nt++)
      for (int r = 0; r < %(dim)d; r++)
        for (int c = 0; c < %(dim)d; c++)
          dst[(i * %(dim)d + r) * N + nt * %(dim)d + c] =
              tmp[((i * tiles_N + nt) * %(dim)d + r) * %(dim)d + c];
}"""


def _emit_chain(plans: list[MxGemmPlan], bundles: list[dict[str, Any]], tr: Transport) -> str:
    """The fused driver: every stage of a matmul chain in ONE program, ONE accelerator region.

    Intermediates never leave the device, and **nothing at all sits between two stages**: the
    producing stage commits its codes tiled and its scales resident, and the consuming stage simply
    points operand A at that scratchpad row. No mvin, no scale load, no transpose, no DRAM.

    Intermediates ARE read back once, for telemetry, but only AFTER the measured region closes --
    so the per-stage codes and E8M0 ranges survive without a byte of it landing on the chain's
    critical path.
    """
    n, last = len(plans), plans[-1]
    names = [_stage_names(i, plans) for i in range(n)]

    head = ["/* Generated by mxgemm_emit.py for target mx_gemmini_rocket — do not edit. */",
            f"/* CHAIN of {n} MX matmuls fused into one program   transport:{tr.name} */"]
    for i, p in enumerate(plans):
        head.append(f"/*   stage {i}  {p.m}x{p.n}x{p.k}  A:{p.act_fmt} B:{p.wgt_fmt} -> {p.out_fmt}"
                    f"   tiles {p.tiles_i}x{p.tiles_j}x{p.tiles_k}   smem@{p.spad_dest} */")

    # Baked data: stage 0's A and every stage's B. A chained stage bakes NO A operand -- it has
    # none to bake, which is the point.
    data: list[str] = []
    for i, (p, nm) in enumerate(zip(plans, names)):
        data += _emit_operand_data(p, bundles[i], nm)

    # Buffers. An intermediate's codes land in a plain [M][N] uint8 array, which IS the [M][K]
    # layout the next stage's mvin wants; its scales land flat, as the requantizer writes them.
    detile_bytes = max((p.out_bytes for p in plans[:-1]), default=0)
    buf = [f"  static out_t C_hw[{last.m}][OUT_COLS];",
           f"  static uint8_t detile_tmp[{max(detile_bytes, 1)}];",]
    buf += [
           "  /* Requant scale write-back target for the final stage. Unused on the BF16 output",
           "     path, but the mxquant config still needs a valid address to point at. */",
           f"  static uint32_t scale_factors[{max(512, -(-last.m * last.scale_blocks // 4))}];"]
    clear = ["  memset(C_hw, 0, sizeof(C_hw));", "  memset(scale_factors, 0, sizeof(scale_factors));"]
    # An intermediate needs NO operand buffer -- it is never moved. T{i} is only the destination of
    # the post-region telemetry readback, and T{i}_sc is the DRAM copy the resident scale write
    # makes anyway (the resident config honours its DRAM address as well as the on-chip window).
    for i, p in enumerate(plans[:-1]):
        buf += [f"  static uint8_t T{i}[{p.out_byte_rows}][{p.out_byte_cols}];"
                f"       /* telemetry readback only ({p.out_bits}-bit elements) */",
                f"  static uint8_t T{i}_sc[{p.m} * {p.scale_blocks}];"]
        clear += [f"  memset(T{i}, 0, sizeof(T{i}));",
                  f"  memset(T{i}_sc, 0, sizeof(T{i}_sc));"]

    body: list[str] = []
    for i, (p, nm) in enumerate(zip(plans, names)):
        if i:
            body += [f"  /* ---- {i - 1} -> {i}: NOTHING. Stage {i - 1} committed its codes tiled at",
                     f"     spad row {p.a_base} and its scales into the act-scale window, so stage",
                     "     {i} reads both in place. This is the seam, and it is empty. */".replace(
                         "{i}", str(i)),
                     ""]
        body += [f"  /* ---- stage {i}: {p.m}x{p.n}x{p.k} -> {p.out_fmt} ---- */",
                 f"  st_c0[{i}] = read_cycles();",
                 *_emit_config_ex(p),
                 *_emit_load_luts(p, nm),
                 *tr.emit_load_scales(p, nm.a_scales, nm.b_scales),
                 *_emit_mvin(p, nm),
                 *_emit_mesh(p, nm, tr),
                 # Only the LAST stage drains inside the region. An intermediate stays resident.
                 *([] if p.chained_out else tr.emit_drain(p, nm.out)),
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
    tail += ['  printf("METRIC cycle_window_mx_gemmini_region 1\\n");',
             '  printf("DONE\\n");']

    report: list[str] = []
    for i, (p, nm) in enumerate(zip(plans, names)):
        if p.chained_out:
            report += [
                f"  /* Stage {i}'s codes are still resident and TILED at spad row {p.spad_dest}.",
                "     Read them back flat so the per-stage telemetry survives. This is outside the",
                "     measured region and is the ONLY time an intermediate is touched. */",
                f"  mvout_detile((uint8_t *)&{nm.out}[0][0], {p.spad_dest}, "
                f"{p.out_byte_rows}, {p.out_byte_cols}, detile_tmp);",
            ]
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
        _DETILE_HELPER % {"dim": plans[0].dim},
        "",
        *data,
        "",
        "int main(void) {",
        "  uint64_t c0, c1;",
        f"  uint64_t st_c0[{n}] = {{0}}, st_c1[{n}] = {{0}};",
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

    A cb carrying a ``graph`` side channel is not a chain at all (attention: three live values, a
    computed B operand, a host softmax) and goes to :mod:`.mxgraph_emit` instead. Dispatching here
    keeps one entry point for the backend while leaving this module's chain lowering untouched.
    """
    if cb.get("graph"):
        from .mxgraph_emit import generate_graph_driver
        return generate_graph_driver(cb)

    plans = plan_chain(cb)
    bundles = _mx_operands(cb, plans)
    tr = transport or RoccSpadTransport()
    return (_emit_single(plans[0], bundles[0], tr) if len(plans) == 1
            else _emit_chain(plans, bundles, tr))
