#!/usr/bin/env python3
"""Generate the MX bisection ladder: `data/mx_ladder_<rung>.h`, one tiny header per rung.

WHY THIS EXISTS. `llama_attention` PASSES on spike and fails on the FPGA with *every* mesh stage
wrong, including the first projection -- while `matmul_tiled_fp8_64x64_chain` and the 128x128 fp8
test PASS on the same bitstream. So the difference is not "MX fp8 on RTL"; it is something the
llama matmuls do that the ISA tests do not. The ladder walks from the passing ISA shape to the
failing llama shape ONE DELTA AT A TIME, so the first rung that fails names the feature.

The deltas, derived by diffing `llama_attention.c`'s first projection (`Q = Xn @ Wq`, M=32 D=2048
H=64 -> I=2, J=4, TK=128) against `matmul_tiled_fp8_64x64.c` (I=J=K=4):

  * the tile grid is NON-SQUARE (I != J), which every passing ISA test avoids;
  * the reduction is 128 k-tiles deep in ONE loop_ws call, against 4 (or 8 at 128x128);
  * the B-side scale window is 4096 bytes, against 128;
  * K-tiled accumulation, output-region reuse, the requant->spad resident seam and a strided A
    mvin are all used further down the kernel and by the BANK_ROWS=2048 build.

Every golden here comes from `fp8_matmul_model.tiled_matmul_hwlike` via `gen_matmul_llama._run_mesh`
-- the same model that generates the llama headers -- so "bit-exact" means the same thing it means
there, and a ladder rung passing on spike is a real statement about the instruction stream.

Operands are deterministic pseudo-random (seeded), not captured llama tensors: the ladder is testing
STRUCTURE, not numerics, and a self-contained generator does not need the capture npz or the venv's
torch-side HuggingFace cache. Only the CODES and the E8M0 scales are baked -- the float operands
never reach the device -- so a header costs one byte per element.

    cd gen && ../../../.venv/bin/python3 gen_mx_ladder.py          # every rung
    cd gen && ../../../.venv/bin/python3 gen_mx_ladder.py mxl4     # just one
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

#: gen/ -> mxgemmini/ -> baremetal/ -> the repo root. Same walk as gen_llama_layer.py.
HERE = Path(__file__).resolve().parent
NPU = HERE.parents[2]
DATA = HERE.parent / "data"
ROCC = NPU.parent / "software" / "gemmini-rocc-tests"
if not (ROCC / "gen_matmul_llama.py").exists():
    raise SystemExit(f"gen_matmul_llama.py not found in {ROCC}")
sys.path.insert(0, str(NPU))
sys.path.insert(0, str(ROCC))
DATA.mkdir(parents=True, exist_ok=True)

import gen_matmul_llama as G  # noqa: E402  -- FORMATS, quantize(), _run_mesh(), _requant(), _rows()

FMT = G.FORMATS["fp8"]
BLOCK = 32

#: The master operands every rung slices. A is [64][2048] so the widest rung (M=32, K=2048) and the
#: 64-row control both come out of it; B is [2048][64] because no rung needs N > 64 (llama's
#: projections are all head_dim- or NCHUNK-wide, and a wider N only makes the header bigger).
#:
#: Block scales are computed ONCE on the masters and SLICED, which is exact: an E8M0 scale belongs
#: to a (row, k-block) pair for the A axis and a (k-block, col) pair for the B axis, so taking whole
#: rows and whole 32-blocks selects scales without recomputing an amax over different elements.
MASTER_M, MASTER_K, MASTER_N = 64, 2048, 64

#: A SECOND B master, 128 columns wide, for the rungs that need a big B-side scale window at a
#: shallow K (window bytes = K/32 * N, so widening N is the only way to grow it without growing K).
#: It is drawn from its OWN generator rather than by widening the master above, because the master's
#: draw order fixes every existing rung's golden -- widening `MASTER_N` would silently change
#: mxl0..mxl9 and invalidate results already measured against the built ELFs.
WIDE_N = 128
WIDE_SEED_OFFSET = 7

#: std 0.5 keeps a 2048-deep reduction inside the narrow per-lane accumulators (acc_e = 4 for lanes
#: 0..14). `_run_mesh` raises if that is ever violated, so this is checked, not assumed.
OPERAND_STD = 0.5
SEED = 20260920


class Rung:
    """One rung: a name, a shape, and the kind of C driver that consumes its header."""

    def __init__(self, name: str, kind: str, M: int, K: int, N: int, why: str, **kw):
        self.name, self.kind = name, kind
        self.M, self.K, self.N = M, K, N
        self.why = why
        self.__dict__.update(kw)


#: The ladder. Order is the bisection order: the first rung that fails on the FPGA names the bug.
RUNGS = [
    Rung("mxl0", "plain", 64, 64, 64,
         "control -- the shape matmul_tiled_fp8_64x64 already passes, driven by llama's own "
         "helpers. A failure here is in the helpers or the config, not in the shapes."),
    Rung("mxl1", "plain", 32, 64, 64,
         "NON-SQUARE tile grid: I=2, J=4. The only delta from mxl0. Every ISA test that passes "
         "on this bitstream is square."),
    Rung("mxl2", "plain", 32, 256, 64,
         "deeper reduction: TK=16 k-tiles in one loop_ws call (mxl1 has 4)."),
    Rung("mxl3", "plain", 32, 1024, 64,
         "TK=64, and a 2048-byte B-side scale window (the ISA tests load 128 bytes)."),
    Rung("mxl4", "plain", 32, 2048, 64,
         "EXACTLY llama's Q = Xn @ Wq: I=2, J=4, TK=128, A scales 2048 B, B scales 4096 B. If "
         "mxl0..mxl3 pass and this fails, the cliff is between TK=64 and TK=128."),
    Rung("mxl5", "ktile", 32, 2048, 64,
         "the SAME matmul as mxl4, split into 2 accumulating K-tiles of 1024 (ex_accumulate = 0 "
         "then 1). Graded against mxl4's golden, so it tests ex_accumulate on RTL -- unverified "
         "per planning/llama_layer_hw_plan.md 10.3. This is the schedule a BANK_ROWS=2048 build "
         "picks for every projection.", ktiles=2),
    Rung("mxl6", "reuse", 32, 64, 64,
         "output-region REUSE: two different matmuls into one C_spad, both with ex_accumulate = 0. "
         "If the RTL never clears its bf16 shadow accumulator the second result is the sum of "
         "both. This is what o_proj's N-chunk loop depends on."),
    Rung("mxl7", "requant", 32, 32, 64,
         "the resident seam: O = A @ B requantized to FP8 straight into the scratchpad in the "
         "operand-A tiled layout, its E8M0 codes written into the act-scale window, then read IN "
         "PLACE as the next matmul's A. llama_attention's P@V -> o_proj.", n2=64),
    Rung("mxl8", "strided", 32, 1024, 64,
         "a STRIDED A mvin: the same matmul as mxl3, but A is a column slice of a [32][2048] "
         "array, so the DMA walks a 2048-byte row pitch for a 16-byte tile. This is how a K-tile "
         "of Xn moves in without being copied out first.", stride=2048, acol=1024),
    Rung("mxl9", "host", 32, 2048, 64,
         "NO MESH AT ALL: mx_host.h's fp32 glue (RMSNorm, block quantize, causal softmax, RoPE) "
         "against the generator's golden. If the host's operand codes are wrong on the FPGA then "
         "every mesh stage differs for a reason that has nothing to do with the mesh."),

    # --- the second tier, added 2026-09-20 after the first RTL run ----------------------------
    #
    # MEASURED on MxGemminiRocketConfig (VCS): mxl0..mxl3 PASS, mxl4 FAILS 2036/2048, with values
    # of the right sign and order of magnitude but 5-60% wrong. So a non-square tile grid is NOT
    # the bug (mxl1 passes), and the cliff is between K=1024 and K=2048. Four things change
    # together across it, and no rung above separates them:
    #
    #   E8M0 groups along K            32 ->   64 B  <- crosses 32 EXACTLY, so a 5-bit group index
    #   K in elements                1024 -> 2048       would wrap here -- and that would produce
    #   A-side scale window          1024 -> 2048 B     precisely this symptom.
    #   B-side scale window          2048 -> 4096 B
    #
    # These five vary them independently. Read as a truth table, against mxl3 PASS / mxl4 FAIL:
    #
    #   cause                mxl10  mxl11  mxl12  mxl13  mxl14
    #   > 32 K-groups        FAIL   FAIL   FAIL   pass   pass
    #   B window > 2048 B    pass   pass   pass   FAIL   pass
    #   A window > 1024 B    FAIL   FAIL   FAIL   pass   FAIL
    #
    # "> 32 groups" and "A window > 1024 B" are the SAME STATEMENT at M=32, since the window is
    # groups*M bytes. mxl14 exists to separate them: M=64 puts a 2048-byte A window on a 32-group
    # reduction.
    Rung("mxl10", "plain", 32, 1088, 64,
         "K=1088 = 34 E8M0 groups, ONE group past mxl3's 32. If the cliff is a 5-bit group index "
         "this is the first rung to fail -- and it fails by a hair rather than by half the "
         "reduction, so the MAGNITUDE of the error here is itself evidence."),
    Rung("mxl11", "plain", 32, 1536, 64,
         "K=1536 = 48 groups, midway between the passing 32 and the failing 64. Brackets the "
         "cliff with mxl10 whichever way that one goes."),
    Rung("mxl12", "plain", 32, 2048, 32,
         "64 groups exactly like mxl4, but N=32 HALVES the B-side scale window to 1024 bytes. If "
         "mxl4's failure is the B window this PASSES; if it is the group count or the A window it "
         "fails exactly like mxl4."),
    Rung("mxl13", "plain", 32, 1024, 128,
         "32 groups like the PASSING mxl3, but N=128 gives it mxl4's 4096-byte B-side scale "
         "window. The only rung that can convict the B window on its own.", wide=True),
    Rung("mxl14", "plain", 64, 1024, 64,
         "32 groups like the passing mxl3, but M=64 gives it mxl4's 2048-byte A-side scale "
         "window. Separates 'the A window is too big' from 'there are too many groups'."),

    # --- the third tier: the requantizer's OUTPUT SCALES ---------------------------------------
    #
    # MEASURED: mxl7 FAILS with `0/2048 codes, 36/64 scales` -- the requantizer computes the right
    # FP8 values and deposits them in the right tiled spad layout, and gets the E8M0 bytes wrong.
    # Y then fails as a consequence (it reads those scales resident).
    #
    # `matmul_tiled_fp8_64x64_chain` PASSES on the same bitstream and checks the DRAM scales the
    # SAME way mxl7 does (`sf[i*GN + b] == C1_scales_out[i][b]`, i.e. [M][GN]). So this is not a
    # plain transpose -- that would break the chain test too. What differs is the tile counts:
    #
    #             M    K    N     I    J    Kt   GN
    #   chain    64   64   64     4    4    4     2    PASSES
    #   mxl7     32   32   64     2    4    2     2    FAILS
    #
    # so `I` and `Kt` both moved. These four separate them, holding GN = 2 throughout (N=64) or
    # dropping it to 1 (N=32) only in the J rung.
    Rung("mxl15", "requant", 64, 64, 64,
         "requant CONTROL: I=J=Kt=4, the chain test's own shape, through this ladder's helpers. "
         "Must PASS -- if it does not, the fault is in the helpers or in this driver, and mxl7 "
         "says nothing about the hardware.", n2=64),
    Rung("mxl16", "requant", 32, 64, 64,
         "one delta from mxl15: M=32, so I=2 while J and Kt stay 4. Convicts I alone.", n2=64),
    Rung("mxl17", "requant", 64, 32, 64,
         "one delta from mxl15: K=32, so Kt=2 while I and J stay 4. Convicts Kt alone.", n2=64),
    Rung("mxl18", "requant", 64, 64, 32,
         "one delta from mxl15: N=32, so J=2 and the output has ONE E8M0 block per row instead of "
         "two. Convicts J, and separately tests whether GN=1 is handled.", n2=64),
]
BY_NAME = {r.name: r for r in RUNGS}


def masters() -> dict:
    """Quantize the master operands once. Everything below is a slice of these."""
    rng = np.random.default_rng(SEED)
    A = (rng.standard_normal((MASTER_M, MASTER_K)) * OPERAND_STD).astype(np.float32)
    B = (rng.standard_normal((MASTER_K, MASTER_N)) * OPERAND_STD).astype(np.float32)
    a_codes, a_scales, a_P = G.quantize(A, axis="row", f=FMT)   # scales [M][K/32]
    b_codes, b_scales, b_P = G.quantize(B, axis="col", f=FMT)   # scales [K/32][N]
    # Drawn after, from a separate generator, so the two above are untouched.
    wrng = np.random.default_rng(SEED + WIDE_SEED_OFFSET)
    BW = (wrng.standard_normal((MASTER_K, WIDE_N)) * OPERAND_STD).astype(np.float32)
    w_codes, w_scales, w_P = G.quantize(BW, axis="col", f=FMT)
    return dict(a_codes=a_codes, a_scales=a_scales, a_P=a_P,
                b_codes=b_codes, b_scales=b_scales, b_P=b_P,
                w_codes=w_codes, w_scales=w_scales, w_P=w_P)


def slice_A(m: dict, M: int, K: int, k0: int = 0):
    """A[:M, k0:k0+K] and its scales, as (codes, scales_MxG, P). Whole 32-blocks only."""
    assert k0 % BLOCK == 0 and K % BLOCK == 0
    g0, g1 = k0 // BLOCK, (k0 + K) // BLOCK
    return (m["a_codes"][:M, k0:k0 + K], m["a_scales"][:M, g0:g1], m["a_P"][:M, k0:k0 + K])


def slice_B(m: dict, K: int, N: int, k0: int = 0, wide: bool = False):
    """B[k0:k0+K, :N] and its scales, as (codes, scales_GxN, P).

    `wide` picks the 128-column master, for the rungs that need N > 64.
    """
    assert k0 % BLOCK == 0 and K % BLOCK == 0
    g0, g1 = k0 // BLOCK, (k0 + K) // BLOCK
    p = "w" if wide else "b"
    return (m[p + "_codes"][k0:k0 + K, :N], m[p + "_scales"][g0:g1, :N],
            m[p + "_P"][k0:k0 + K, :N])


def golden(aP, asc, bP, bsc) -> np.ndarray:
    """One matmul through the bit-exact mesh model. `asc`/`bsc` are the UNTRANSPOSED scale arrays,
    exactly as `gen_llama_layer.mesh()` passes them -- the transpose below is a WIRE layout, not a
    model input, and confusing the two silently grades against the wrong thing."""
    return G._run_mesh(aP, asc, bP, bsc, FMT)


# --- emission ----------------------------------------------------------------------------------

def _hdr(rung: Rung, body: str) -> str:
    guard = f"INCLUDE_MX_LADDER_{rung.name.upper()}_H"
    return f"""// GENERATED by gen/gen_mx_ladder.py -- do not edit. Rung {rung.name} of the MX bisection ladder.
//
// {rung.why}
//
// Operands: deterministic pseudo-random (seed {SEED}), quantized by MXQuant's own
// quantize_mx_block32 (block {BLOCK}, {FMT.name}); mesh goldens from
// fp8_matmul_model.tiled_matmul_hwlike at the datapath's precision schedule.
#ifndef {guard}
#define {guard}

#define LAD_NAME "{rung.name}"
// The first sentence of the rung's reason, for the ELF's own banner. Split on ". " rather than
// "." so an abbreviation ("mx_host.h") does not truncate the line mid-word.
#define LAD_WHY  "{rung.why.split('. ')[0].rstrip('.')}"

{body}
#endif  // {guard}
"""


def _scale_arrays(name: str, a_scales: np.ndarray, b_scales: np.ndarray, M: int, N: int, GK: int,
                  r) -> str:
    """The two scale WINDOWS, in the layout `gemmini_mx_load_scales` wants.

    The A window is indexed `a_off = group * M + row`, so it is [K/32][M] -- the TRANSPOSE of the
    [M][K/32] the quantizer returns. The B window is `b_off = group * N + col`, i.e. [K/32][N],
    which is already the quantizer's shape for axis="col". Getting this backwards is the bug
    planning/llama_layer_hw_plan.md 10.4 records, so both are spelled out rather than inferred.
    """
    return f"""static const uint8_t {name}_A_SCALES[{GK}][{M}] = {{
{r(a_scales.T, 2)}
}};

static const uint8_t {name}_B_SCALES[{GK}][{N}] = {{
{r(b_scales, 2)}
}};
"""


def emit_plain(rung: Rung, m: dict) -> str:
    """plain / ktile / strided all consume the same arrays; the driver differs, not the data."""
    r, M, K, N = G._rows, rung.M, rung.K, rung.N
    GK = K // BLOCK
    # The strided rung's A is the slice the DMA has to gather, so its golden must be that slice --
    # not the leading K columns. Getting this wrong makes the rung fail everywhere, including spike.
    k0 = rung.acol if rung.kind == "strided" else 0
    ac, asc, aP = slice_A(m, M, K, k0)
    bc, bsc, bP = slice_B(m, K, N, wide=getattr(rung, "wide", False))
    C = G.bf16_bits(golden(aP, asc, bP, bsc))

    extra = ""
    a_decl = f"""static const uint8_t LAD_A[{M}][{K}] = {{
{r(ac, 2)}
}};
"""
    if rung.kind == "strided":
        # A is a COLUMN SLICE of a wider array, so the device must mvin with the wide row pitch.
        # The untouched columns are filled from the master's other half rather than zeroed: a zero
        # background would make an off-by-one stride read plausible data and pass.
        stride, acol = rung.stride, rung.acol
        wide = np.zeros((M, stride), dtype=np.uint8)
        wide[:, :] = m["a_codes"][:M, :stride]
        assert np.array_equal(wide[:, acol:acol + K], ac), "strided master does not carry the slice"
        a_decl = f"""// A lives as a column slice at [.., {acol}:{acol + K}] of a {stride}-wide array: the mvin DMA must
// walk a {stride}-byte row pitch to collect each {16}-byte tile row.
#define LAD_A_STRIDE {stride}
#define LAD_A_COL    {acol}
static const uint8_t LAD_A_WIDE[{M}][{stride}] = {{
{r(wide, 2)}
}};
"""
    if rung.kind == "ktile":
        extra = f"#define LAD_KTILES {rung.ktiles}\n"

    return f"""#define LAD_M {M}
#define LAD_K {K}
#define LAD_N {N}
#define LAD_GK {GK}
{extra}
{a_decl}
static const uint8_t LAD_B[{K}][{N}] = {{
{r(bc, 2)}
}};

{_scale_arrays('LAD', asc, bsc, M, N, GK, r)}
static const uint16_t LAD_C_BF16[{M}][{N}] = {{
{r(C, 4)}
}};
"""


def emit_reuse(rung: Rung, m: dict) -> str:
    """Two DISTINCT matmuls of the same shape, to be run into the same C_spad."""
    r, M, K, N = G._rows, rung.M, rung.K, rung.N
    GK = K // BLOCK
    out = [f"#define LAD_M {M}\n#define LAD_K {K}\n#define LAD_N {N}\n#define LAD_GK {GK}\n"]
    for idx, k0 in enumerate((0, K)):
        s = "" if idx == 0 else "2"
        ac, asc, aP = slice_A(m, M, K, k0)
        bc, bsc, bP = slice_B(m, K, N, k0)
        C = G.bf16_bits(golden(aP, asc, bP, bsc))
        out.append(f"""static const uint8_t LAD{s}_A[{M}][{K}] = {{
{r(ac, 2)}
}};

static const uint8_t LAD{s}_B[{K}][{N}] = {{
{r(bc, 2)}
}};

{_scale_arrays('LAD' + s, asc, bsc, M, N, GK, r)}
static const uint16_t LAD{s}_C_BF16[{M}][{N}] = {{
{r(C, 4)}
}};
""")
    return "\n".join(out)


def emit_requant(rung: Rung, m: dict) -> str:
    """O = A @ B requantized to FP8 in the scratchpad, then Y = O @ W2 reading O in place."""
    r, M, K, N, N2 = G._rows, rung.M, rung.K, rung.N, rung.n2
    GK, GN = K // BLOCK, N // BLOCK
    ac, asc, aP = slice_A(m, M, K)
    bc, bsc, bP = slice_B(m, K, N)
    O_bf16 = golden(aP, asc, bP, bsc)
    o_codes, o_scales, o_P = G._requant(O_bf16, FMT)          # codes [M][N], scales [M][N/32]

    # The second matmul's B is a DIFFERENT slice of the master, so a stale-operand failure cannot
    # masquerade as a pass.
    w2c, w2sc, w2P = slice_B(m, N, N2, k0=2 * BLOCK)
    Y = G.bf16_bits(golden(o_P, o_scales, w2P, w2sc))

    return f"""#define LAD_M  {M}
#define LAD_K  {K}
#define LAD_N  {N}
#define LAD_N2 {N2}
#define LAD_GK {GK}
#define LAD_GN {GN}

static const uint8_t LAD_A[{M}][{K}] = {{
{r(ac, 2)}
}};

static const uint8_t LAD_B[{K}][{N}] = {{
{r(bc, 2)}
}};

{_scale_arrays('LAD', asc, bsc, M, N, GK, r)}
// What the requantizer must produce: the FP8 codes of O, and its per-block E8M0 bytes. The
// hardware writes the scales [M][N/32] to DRAM and, in resident mode, the SAME bytes transposed
// into the act-scale window ([N/32][M], a_off = group * M + row) for the next matmul to read.
static const uint8_t LAD_O_CODES[{M}][{N}] = {{
{r(o_codes, 2)}
}};

static const uint8_t LAD_O_SCALES[{M}][{GN}] = {{
{r(o_scales, 2)}
}};

// Y = O @ W2, with O and its scales read IN PLACE -- no A mvin, no A-scale load.
static const uint8_t LAD_W2[{N}][{N2}] = {{
{r(w2c, 2)}
}};

static const uint8_t LAD_W2_SCALES[{GN}][{N2}] = {{
{r(w2sc, 2)}
}};

static const uint16_t LAD_Y_BF16[{M}][{N2}] = {{
{r(Y, 4)}
}};
"""


def emit_host(rung: Rung, m: dict) -> str:
    """No mesh: the fp32 host glue, against goldens computed the same way the llama headers are.

    Every stage here feeds the mesh in the real kernel, so a host disagreement explains "every mesh
    stage differs" without any mesh being at fault. The inputs are BF16 bit patterns because that is
    what the kernel reads (`H_PRE_BF16`), and BF16 is exactly representable in fp32, so the C and
    the numpy start from identical values.
    """
    from app.capture_llama_layer import rmsnorm, softmax_causal, rope

    r, M, D, H = G._rows, rung.M, rung.K, rung.N
    GD, GH, GM = D // BLOCK, H // BLOCK, M // BLOCK
    eps = 1e-5
    rng = np.random.default_rng(SEED + 1)

    def as_bf16(x):
        bits = G.bf16_bits(x)
        return bits, (bits.astype(np.uint32) << 16).view(np.float32).reshape(x.shape)

    h_bits, h = as_bf16((rng.standard_normal((M, D)) * 0.02).astype(np.float32))
    w_bits, w = as_bf16((1.0 + rng.standard_normal(D) * 0.1).astype(np.float32))

    xn = rmsnorm(h, w, eps)
    xn_codes, xn_scales, _ = G.quantize(xn, axis="row", f=FMT)

    # 1/sqrt(H) then the causal mask then the row softmax -- the same order, and the same scale,
    # that mx_softmax_causal applies in C (`mx_softmax_causal(S, M, 1/sqrtf(H), p)`).
    s_bits, s = as_bf16((rng.standard_normal((M, M)) * 2.0).astype(np.float32))
    p = softmax_causal(s / np.sqrt(np.float32(H)))
    p_codes, p_scales, _ = G.quantize(p, axis="row", f=FMT)

    q_bits, q = as_bf16((rng.standard_normal((M, H)) * 0.5).astype(np.float32))
    cos = np.cos(np.arange(M)[:, None] * (10000.0 ** (-np.arange(H) / H))[None, :]).astype(np.float32)
    sin = np.sin(np.arange(M)[:, None] * (10000.0 ** (-np.arange(H) / H))[None, :]).astype(np.float32)
    qr = rope(q, cos, sin)
    qr_codes, qr_scales, _ = G.quantize(qr, axis="row", f=FMT)

    def f32_rows(a):
        u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
        return ",\n".join("    { " + ", ".join("0x%08x" % int(v) for v in row) + " }" for row in u)

    return f"""#define LAD_M  {M}
#define LAD_D  {D}
#define LAD_H  {H}
#define LAD_GD {GD}
#define LAD_GH {GH}
#define LAD_GM {GM}
#define LAD_EPS {eps:g}f

// --- RMSNorm + block quantize -------------------------------------------------------------
static const uint16_t LAD_H_PRE_BF16[{M}][{D}] = {{
{r(h_bits, 4)}
}};

static const uint16_t LAD_W_LN_BF16[{D}] = {{
    {", ".join("0x%04x" % int(v) for v in w_bits)}
}};

static const uint8_t LAD_XN_CODES[{M}][{D}] = {{
{r(xn_codes, 2)}
}};

static const uint8_t LAD_XN_SCALES[{GD}][{M}] = {{
{r(xn_scales.T, 2)}
}};

// --- causal softmax + block quantize ------------------------------------------------------
static const uint16_t LAD_S_BF16[{M}][{M}] = {{
{r(s_bits, 4)}
}};

static const uint8_t LAD_P_CODES[{M}][{M}] = {{
{r(p_codes, 2)}
}};

static const uint8_t LAD_P_SCALES[{GM}][{M}] = {{
{r(p_scales.T, 2)}
}};

// --- RoPE + block quantize ----------------------------------------------------------------
static const uint16_t LAD_Q_BF16[{M}][{H}] = {{
{r(q_bits, 4)}
}};

static const uint32_t LAD_ROPE_COS[{M}][{H}] = {{
{f32_rows(cos)}
}};

static const uint32_t LAD_ROPE_SIN[{M}][{H}] = {{
{f32_rows(sin)}
}};

static const uint8_t LAD_QR_CODES[{M}][{H}] = {{
{r(qr_codes, 2)}
}};

static const uint8_t LAD_QR_SCALES[{GH}][{M}] = {{
{r(qr_scales.T, 2)}
}};
"""


EMIT = {"plain": emit_plain, "ktile": emit_plain, "strided": emit_plain,
        "reuse": emit_reuse, "requant": emit_requant, "host": emit_host}


def build(names: list[str]) -> int:
    m = masters()
    for name in names:
        rung = BY_NAME[name]
        body = EMIT[rung.kind](rung, m)
        path = DATA / f"mx_ladder_{name}.h"
        path.write_text(_hdr(rung, body))
        print(f"  {name:6s} {rung.kind:8s} M={rung.M} K={rung.K} N={rung.N}  ->  "
              f"{path.name} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rungs", nargs="*", default=[], help=f"one or more of {list(BY_NAME)}")
    a = ap.parse_args()
    names = a.rungs or list(BY_NAME)
    bad = [n for n in names if n not in BY_NAME]
    if bad:
        raise SystemExit(f"unknown rung(s) {bad}; known: {list(BY_NAME)}")
    print(f"MX bisection ladder -> {DATA}")
    return build(names)


if __name__ == "__main__":
    raise SystemExit(main())
