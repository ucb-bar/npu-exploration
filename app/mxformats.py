"""The MX element formats, as one table: what the quantizer, the emitter and the RTL each need.

Every format fact this repo uses lives here, so there is exactly one place to correct when the
hardware changes — and one place to read when asking "can this elaboration run this dtype?".

The fields fall into three groups, deliberately kept in one record because they are *the same
decision* seen from three layers:

* **quantization** — ``mxq`` (the alias MXQuant's ``quantize_mx_block32`` understands),
  ``out_pmax``, ``out_requant``. See :mod:`app.mxq_golden`, the only quantizer.
* **the hardware selector** — ``fmt_code`` / ``altfmt`` / ``lut`` / ``entry_bits``. These are not a
  simple enum: the format the mesh decodes is a function of *three* things, two of them in
  ``config_ex`` and one of them a runtime flag. See :data:`SELECTOR_DOC`.
* **mesh geometry** — ``tile_m/n/k`` and which bit-exact model reproduces it.

Provenance for every value is the current source, re-read 2026-09-08 (the older plans predate the
datatype/requant rebuild and disagree; where they do, the code wins):

* selector decode — ``libgemmini/gemmini.cc:460-464``
* which (code, altfmt, lut) means which format — ``gemmini.cc:1183-1188, 1222-1224, 1356-1368``
* output ``log2_pmax`` per format — ``gemmini.cc:1424-1433`` and the FP4 path at ``:1478``
* codebook width for ``MX_LOAD_LUT`` — ``gemmini.h:68-76`` (``gemmini_mx_load_lut_dt``)
* mesh tile geometry — ``gemmini.cc:1371`` and ``:1519``. The LUT path and the FP4 path BOTH march
  ``TM = TN = 32``; only plain 8-bit E4M3 is 16x16. An earlier draft of this table had the two
  8-bit-codebook formats at 16 and it was simply wrong -- the width follows the 4-bit STORAGE, not
  the codebook's element width.
* which chipyard elaboration carries which format — ``ConfigsFP.scala:331-420``,
  ``chipyard/GemminiConfigs.scala:49-95``

Ported from ``gemmini-rocc-tests/gen_matmul_llama.py:72-100``, which covered fp8/fp6/fp4 only; the
three LUT sub-formats came from a second generator family (``lut_mapping_demo.py``,
``llama_operands.py``) and had no table at all.
"""
from __future__ import annotations

from dataclasses import dataclass

#: How the mesh picks a format. Kept as prose because the shape of it is the surprise: a 2-bit code
#: would suggest four formats, and there are six.
SELECTOR_DOC = """\
config_ex rs1: [11:10] act_fmt, [13:12] wgt_fmt, [15:14] out_fmt, [6] altfmt, [5] uselut.
A RUNTIME flag `lut_en` is set by MX_LOAD_LUT (funct 29) and cleared by MX_LUT_DISABLE (funct 30),
and it is what promotes E4M3 from the direct 8-bit path to the 4-wide quad path. So the decoded
format is (fmt_code, altfmt, lut_en) -- three inputs, not one enum."""


class MxFormatError(RuntimeError):
    """An unsupported format, or one used before it has been proven end to end."""


@dataclass(frozen=True)
class MxFormat:
    """One MX element format, as the quantizer, the emitter and the RTL each see it."""

    name: str            #: our key, e.g. ``"fp8_e4m3"``
    spec: str            #: this repo's own spelling, e.g. ``"fp8:e4m3"`` (the mesh models parse it)
    mxq: str             #: the alias ``quantize_mx_block32`` understands
    mlir: str | None     #: the ``merlin_iface`` dtype spelling, or None if the grammar has no name
    bits: int            #: element width ON THE WIRE (4 for nibble-packed / LUT-indexed operands)

    # --- the hardware selector -------------------------------------------------------------
    fmt_code: int        #: config_ex rs1 [11:10] / [13:12] / [15:14]
    altfmt: int          #: config_ex rs1[6]
    lut: bool            #: operands are 4-bit codebook indices; needs MX_LOAD_LUT
    entry_bits: int | None   #: codebook width for gemmini_mx_load_lut_dt, None when not LUT-indexed

    # --- quantization ----------------------------------------------------------------------
    #: ``log2_pmax`` the requantizer subtracts when it commits OUTPUT blocks.
    #:
    #: **0 for every format as of 2026-09-09.** It used to vary (FP4 2, FP6 4, E5M2 16,
    #: E4M3-quad 8), which made every one of those formats unchainable: the mesh accumulates a
    #: 16-deep column of RAW CODES at exponent width 4, so a chained operand must satisfy
    #: ``16 * |A|max * |B|max <= 2**8``, i.e. codes in [1, 2), i.e. ``out_pmax = 0``. The RTL
    #: (``MxRequantizer.scala:37,180``) and spike (``gemmini.cc:1450,1578``) now both hardwire 0.
    #: Kept as a field rather than deleted because it is an RTL fact this table's job is to mirror.
    out_pmax: int
    #: who requantizes the OUTPUT. ``"mxquant"`` is MXQuant's ``_quantize_elemwise``. ``"model"``
    #: is the hardware's own two-stage quantizer: ``mx_fp_math.h:402-404`` rounds ties differently,
    #: and using MXQuant there disagrees on ~14% of elements in the mantissa LSB (§9.4).
    #: NOTE this is not a violation of "MXQuant only" — that governs how OPERANDS and block scales
    #: are made. It cannot govern how the hardware rounds its own output.
    out_requant: str

    # --- mesh geometry ---------------------------------------------------------------------
    tile_m: int
    tile_n: int
    tile_k: int
    model: str           #: which bit-exact mesh model reproduces this format

    #: Proven end to end (host quantizer == the baremetal headers == spike) in THIS repo.
    #: An unproven format raises rather than emitting C that has never been checked.
    proven: bool = False

    @property
    def packed_per_byte(self) -> int:
        """Elements per wire byte: 1 for the 8-bit formats, 2 for every nibble/index format."""
        return 8 // self.bits

    #: Proven to CHAIN: its requantized output can be re-read as the next stage's operand. Strictly
    #: stronger than `proven`, and for the codebook formats it is not implied by it -- see
    #: `chain_refusal`.
    chain_proven: bool = False

    def require_proven(self, where: str) -> None:
        if not self.proven:
            raise MxFormatError(
                f"{where}: format {self.name!r} is declared but NOT proven in this repo yet — its "
                "operand packing, LUT handling and requant path are Step 5 of "
                "planning/merlin_glue_port_plan.md. Refusing to emit an unchecked datapath.")


def chain_refusal(fmt: "MxFormat") -> str | None:
    """Why ``fmt`` cannot be the intermediate of a chain, or None if it can.

    **This currently refuses nothing, and that is a result rather than an oversight.** It is kept as
    a live invariant because the condition it tests is one a future format can walk straight back
    into.

    The requantizer normalizes its output to ``[2**out_pmax, 2**(out_pmax+1))``. The next stage
    reads those values through a codebook, and the hardware's nearest-entry finder compares in a
    fixed-point domain whose width mask makes large magnitudes ALIAS onto small ones
    (``app/mxlut._fixed_point``). If the requant range starts ABOVE the largest value the finder can
    represent faithfully, every element aliases and the chain returns noise. That is exactly what
    the codebook formats used to do:

    ======  ==============  ===============  ============  ======================
    format  out_pmax (was)  requant range    finder max    2-stage mlp2, then
    ======  ==============  ===============  ============  ======================
    E2M3    2               [4, 8)           7.5           21.6%  -- fitted
    E3M2    4               [16, 32)         14            62%    -- aliased
    E5M2    16              [65536, 131072)  57344         NaN    -- aliased
    ======  ==============  ===============  ============  ======================

    The RTL and spike now use ``log2_pmax = 0`` for every format, so every requant range is
    ``[1, 2)`` and no finder can alias. Every ``out_pmax`` in the table below is 0 and the range arm
    is dormant; a format that reintroduces a nonzero one will trip it again.
    """
    if not fmt.lut:
        return None
    from . import mxlut
    import numpy as np

    lo = 2.0 ** fmt.out_pmax
    finder_max = float(np.abs(mxlut.codebook_values(fmt)).max())
    if finder_max < lo:
        return (f"{fmt.name} requantizes into [{lo:g}, {2*lo:g}) but its nearest-entry finder "
                f"cannot faithfully represent anything above {finder_max:g} -- every intermediate "
                "value would alias onto a smaller one. This is a datapath property, not a missing "
                "feature; see app/mxformats.chain_refusal.")
    if not fmt.chain_proven:
        return (f"{fmt.name} passes the range check but its chain is UNEXPLAINED: a 2-stage mlp2 "
                "measures ~99% vs fp32 where ~20% is expected, and the cause is not yet found. "
                "Refusing rather than shipping a chain that is quietly wrong.")
    return None


#: Block-scale group size along K. E8M0, one code per 32 elements.
BLOCK = 32
#: E8M0 exponent bias.
E8M0_BIAS = 127

#: The output-only pseudo-format: the raw bf16 accumulator readout, no requantization.
BF16_FMT_CODE = 3

#: Codebook granularity ``G``: one 16-entry LUT per ``2**G`` rows of A / columns of B
#: (``gemmini.cc:1392`` ``lut_idx = (i*TM + m) >> G``). Every shipped LUT test uses 1, and it is
#: also what ``gemmini_mxquant_config_mvout``'s last argument carries.
LUT_GRANULARITY = 1

#: A codebook is 16 entries of ``entry_bits``, LE-packed into 32-bit words.
def lut_words(entry_bits: int) -> int:
    """uint32 words per codebook: 3 for 6-bit entries (96 bits), 4 for 8-bit (128)."""
    return (16 * entry_bits + 31) // 32

FORMATS: dict[str, MxFormat] = {
    # ---- FP8 -------------------------------------------------------------------------------
    # The only format proven end to end here: it is what every kernel in kernels/registry.py runs
    # on today, and what the baseline in planning/merlin_glue_port_plan.md §4.1 was measured with.
    "fp8_e4m3": MxFormat(
        name="fp8_e4m3", spec="fp8:e4m3", mxq="MXFP8_E4M3", mlir="f8E4M3FN", bits=8,
        fmt_code=0, altfmt=0, lut=False, entry_bits=None,
        out_pmax=0, out_requant="mxquant",
        tile_m=16, tile_n=16, tile_k=16, model="fp8_matmul_model", proven=True, chain_proven=True),

    # Same element format, but operands are 4-bit indices into an 8-bit codebook. Selected by the
    # RUNTIME lut_en flag, not by a different fmt_code — gemmini.cc:1222-1224.
    "fp8_e4m3_quad": MxFormat(
        name="fp8_e4m3_quad", spec="fp8:e4m3", mxq="MXFP8_E4M3", mlir=None, bits=4,
        fmt_code=0, altfmt=0, lut=True, entry_bits=8,
        out_pmax=0, out_requant="mxquant",
        tile_m=32, tile_n=32, tile_k=16, model="lut_fp8_matmul_model", proven=True, chain_proven=True),

    "fp8_e5m2": MxFormat(
        name="fp8_e5m2", spec="fp8:e5m2", mxq="MXFP8_E5M2", mlir=None, bits=4,
        fmt_code=0, altfmt=1, lut=True, entry_bits=8,
        out_pmax=0, out_requant="model",
        tile_m=32, tile_n=32, tile_k=16, model="lut_fp8_matmul_model", proven=True, chain_proven=True),

    # ---- FP6 -------------------------------------------------------------------------------
    "fp6_e3m2": MxFormat(
        name="fp6_e3m2", spec="fp6:e3m2", mxq="MXFP6_E3M2", mlir="f6E3M2FN", bits=4,
        fmt_code=1, altfmt=0, lut=True, entry_bits=6,
        out_pmax=0, out_requant="model",
        tile_m=32, tile_n=32, tile_k=16, model="fp4_matmul_model", proven=True, chain_proven=True),

    "fp6_e2m3": MxFormat(
        name="fp6_e2m3", spec="fp6:e2m3", mxq="MXFP6_E2M3", mlir=None, bits=4,
        fmt_code=1, altfmt=1, lut=True, entry_bits=6,
        out_pmax=0, out_requant="model",
        tile_m=32, tile_n=32, tile_k=16, model="fp4_matmul_model", proven=True, chain_proven=True),

    # ---- FP4 -------------------------------------------------------------------------------
    # Direct 4-bit codes, NOT LUT indices: the only nibble format that is not codebook-indexed.
    "fp4_e2m1": MxFormat(
        name="fp4_e2m1", spec="fp4:e2m1", mxq="MXFP4", mlir="f4E2M1FN", bits=4,
        fmt_code=2, altfmt=0, lut=False, entry_bits=None,
        out_pmax=0, out_requant="model",
        tile_m=32, tile_n=32, tile_k=16, model="fp4_matmul_model", proven=True, chain_proven=True),
}

#: Accepted spellings -> our key. The interface grammar uses MLIR's builtin fp8 names, older command
#: buffers used ``mxfp8``, and this table's own keys are used directly by the front end.
ALIASES: dict[str, str] = {
    **{f.name: f.name for f in FORMATS.values()},
    **{f.mlir: f.name for f in FORMATS.values() if f.mlir},
    "mxfp8": "fp8_e4m3", "fp8": "fp8_e4m3", "e4m3": "fp8_e4m3",
    "mxfp6": "fp6_e3m2", "fp6": "fp6_e3m2", "e3m2": "fp6_e3m2",
    "mxfp4": "fp4_e2m1", "fp4": "fp4_e2m1", "e2m1": "fp4_e2m1",
    "e5m2": "fp8_e5m2", "e2m3": "fp6_e2m3",
}


def get(dtype: str, *, where: str = "format lookup", proven_only: bool = True) -> MxFormat:
    """Resolve a dtype spelling to its :class:`MxFormat`, failing closed.

    ``proven_only`` refuses a format this repo has declared but not yet checked end to end, which
    is the whole point of the ``proven`` flag: an unchecked datapath must not silently emit.
    """
    key = ALIASES.get(dtype) or ALIASES.get(str(dtype).lower())
    if key is None:
        raise MxFormatError(
            f"{where}: unknown MX dtype {dtype!r}. Known: {sorted(set(ALIASES))}")
    f = FORMATS[key]
    if proven_only:
        f.require_proven(where)
    return f


#: Which chipyard elaboration can run which formats. "Any datatype" is gated by the RTL build, not
#: by software — ConfigsFP.scala:331-420 elaborates only the decode logic a config declares, so a
#: format absent here is not slow on that build, it is absent.
ELABORATIONS: dict[str, tuple[str, ...]] = {
    "MxGemminiRocketConfig":        ("fp8_e4m3",),
    "MxE5M2GemminiRocketConfig":    ("fp8_e5m2",),
    "MxAllGemminiRocketConfig":     ("fp4_e2m1", "fp6_e3m2", "fp6_e2m3",
                                     "fp8_e4m3", "fp8_e4m3_quad", "fp8_e5m2"),
    "MxE4M3LutGemminiRocketConfig": ("fp4_e2m1", "fp6_e3m2", "fp6_e2m3",
                                     "fp8_e4m3", "fp8_e4m3_quad", "fp8_e5m2"),
    "MxFp4OnlyGemminiRocketConfig": ("fp4_e2m1",),
    "MxE3M2OnlyGemminiRocketConfig": ("fp6_e3m2",),
    "MxE2M3OnlyGemminiRocketConfig": ("fp6_e2m3",),
    "MxE4M3OnlyGemminiRocketConfig": ("fp8_e4m3", "fp8_e4m3_quad"),
    "MxE5M2OnlyGemminiRocketConfig": ("fp8_e5m2",),
}

#: The elaboration the spike model behaves as. Spike decodes every format regardless of config, so
#: it is the permissive one — which is exactly why a run passing on spike is not evidence the
#: target build can run it. :func:`check_elaboration` is what closes that gap.
SPIKE_ELABORATION = "MxAllGemminiRocketConfig"


def check_elaboration(dtypes, config: str) -> None:
    """Refuse a (formats, RTL config) pair the elaboration cannot run.

    Spike would run it anyway — it models every format — so without this check a kernel could pass
    on spike and be unrunnable on the build it claims to target.
    """
    if config not in ELABORATIONS:
        raise MxFormatError(
            f"unknown elaboration {config!r}. Known: {sorted(ELABORATIONS)}")
    have = set(ELABORATIONS[config])
    want = {get(d, where="elaboration check", proven_only=False).name for d in dtypes}
    missing = sorted(want - have)
    if missing:
        raise MxFormatError(
            f"{config} does not elaborate {missing}; it carries {sorted(have)}. "
            "Pick another config (see ELABORATIONS) or another dtype.")
