#!/usr/bin/env python3
"""Generate include/llama_mlp.h -- a REAL TinyLlama MLP, back to back, at llama dimensions.

Operands come from `npu-exploration/app/capture_llama_layer.py`, which captures one real decoder
layer from one real forward pass: the residual stream, both RMSNorm weights, and the actual
gate/up/down weights, all index-consistent (unlike the `data_evalrun_512` tiles, whose random
per-projection offsets cannot be composed -- see `planning/llama_layer_hw_plan.md` section 1).

By default the hidden size D = 2048 is kept FULL, so RMSNorm and every projection *input* are
exact; the slice is on the output side, F of the 5632 FFN neurons. `gate`/`up` are therefore exact
real llama values and `down` is an honest partial sum over those F neurons -- and the fp32 reference
emitted here is truncated the same way, so it grades what the device actually computes.

`--d D` additionally cuts the HIDDEN axis, which makes an RTL run affordable (the host's fp32 glue is
99% of the cycles and scales linearly in D) at the cost of that exactness: RMSNorm then normalizes
over D features and every projection becomes a partial sum over them, so the values are real
arithmetic on real weights but NOT the numbers the model computes. `slice_capture` recomputes the
fp32 references on the slice so the grade still describes the device exactly, and every generated
header states which of the two it is.

What runs where (user, 2026-09-05: the host glue runs in fp32 on Rocket and is NOT mirrored
bit-for-bit here):

    host   xn = rmsnorm(h_mid, w_post_ln)          fp32, then MX-quantized to fp8 + E8M0
    mesh   G  = Xn @ Wg      [M,D]x[D,F] -> bf16
    mesh   U  = Xn @ Wu      [M,D]x[D,F] -> bf16
    host   h  = silu(G) * U                        fp32, then MX-quantized
    mesh   Y  = H  @ Wd      [M,F]x[F,D] -> bf16   (emitted in two N-chunks; see the .c)
    host   out = h_mid + Y                         the residual

Every mesh stage is modelled by `fp8_matmul_model.tiled_matmul_hwlike` at the datapath's own
precision schedule, so its golden is bit-exact given the same operand codes. The host stages are
modelled in plain fp32 numpy; the header carries their golden codes too, so the C can report how far
its own fp32 differs (expected: not at all -- e4m3's 3 mantissa bits absorb a last-ulp difference).

Run it with the npu-exploration venv:

    cd generators/gemmini/software/gemmini-rocc-tests
    PATH=../../npu-exploration/.venv/bin:$PATH \
      ../../npu-exploration/.venv/bin/python3 gen_llama_layer.py mlp
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

#: gen/ -> mxgemmini/ -> baremetal/ -> the repo root.
HERE = Path(__file__).resolve().parent
NPU = HERE.parents[2]
#: Generated data lands next to the kernels that include it, not next to the generator.
DATA = HERE.parent / "data"
#: gen_matmul_llama.py stays in gemmini-rocc-tests: it generates the matmul_*.h data for the ISA
#: tests themselves, so it belongs with them. We borrow its FORMATS/quantize/mesh-model helpers.
ROCC = NPU.parent / "software" / "gemmini-rocc-tests"
if not (NPU / "app" / "mxq_golden.py").exists():
    raise SystemExit(f"npu-exploration not found at {NPU}")
if not (ROCC / "gen_matmul_llama.py").exists():
    raise SystemExit(f"gen_matmul_llama.py not found in {ROCC}")
sys.path.insert(0, str(NPU))
sys.path.insert(0, str(ROCC))
DATA.mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402
import gen_matmul_llama as G  # noqa: E402  -- PROD/ACC precision, quantize(), bf16_bits(), _rows()
from app.capture_llama_layer import rmsnorm, silu  # noqa: E402  -- ONE definition of the host math
from app.mxwire import e8m0_decode  # noqa: E402

CAPTURE = NPU / "out" / "layer_capture"
FMT = G.FORMATS["fp8"]
BLOCK = 32


def load_capture(path: Path | None = None) -> dict:
    """The newest layer capture, or a named one."""
    if path is None:
        cands = sorted(CAPTURE.glob("layer*.npz"))
        if not cands:
            raise SystemExit(
                f"no capture in {CAPTURE}\nRun it first:\n"
                f"    cd {NPU} && .venv/bin/python3 -m app.capture_llama_layer")
        path = cands[-1]
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d["_path"] = path
    #: The capture's own D, kept across a slice so a header can state what it was cut from.
    d["meta_d_model_full"] = d["meta_d_model"]
    return d


#: Tensors carrying a hidden-size axis, and which axis that is. `rope_cos`/`rope_sin` are indexed by
#: head dim, not by D, so they are deliberately absent. `ref_mlp`/`ref_attn` are absent too: a sliced
#: reference is not the slice of a reference, so they are RECOMPUTED (see `slice_capture`).
_D_AXIS = {"h_pre": 1, "h_mid": 1, "h_out": 1, "w_in_ln": 0, "w_post_ln": 0,
           "Wq": 0, "Wk": 0, "Wv": 0, "Wg": 0, "Wu": 0, "Wo": 1, "Wd": 1}


def slice_capture(cap: dict, d: int) -> dict:
    """The same real layer with the hidden size cut to the first `d` features.

    Why this is honest, and exactly how far. The full-D capture already grades PARTIAL SUMS: `Wo`
    covers 1 of 32 heads and `Wd` covers F of 5632 neurons, with the fp32 reference truncated the
    same way (`capture_llama_layer.py` module docstring). Cutting D extends that truncation to the
    two axes it had left whole:

      * RMSNorm's mean is now over `d` features, not 2048 -- so `xn` is NOT the real llama
        normalized activation, and this is the one quantity a D-slice stops being exact;
      * Q/K/V and gate/up become partial sums over `d` of 2048 input features, joining `Wo`/`Wd`;
      * `Wo`/`Wd` lose output columns, so the result is a `d`-wide slice of the layer output.

    Every OPERAND is still a real llama value at its real index, and `reference_{attn,mlp}` are
    recomputed from the sliced tensors rather than sliced from the stored full-D references, so the
    fp32 grade describes exactly what the device computes. What this buys is the point: the host's
    fp32 glue is 99.9% of the cycles and scales linearly in D, so d=256 is ~8x less RTL simulation
    for a chain with every structural feature of the full one intact.
    """
    from app.capture_llama_layer import reference_attn, reference_mlp

    D = int(cap["meta_d_model"])
    if d == D:
        return cap
    if not 0 < d <= D:
        raise SystemExit(f"--d {d} out of range: the capture has D={D}")
    if d % 32:
        raise SystemExit(f"--d {d} must be a multiple of 32 (the E8M0 block)")

    out = dict(cap)
    for key, axis in _D_AXIS.items():
        out[key] = np.ascontiguousarray(cap[key][:d] if axis == 0 else cap[key][..., :d])
    out["meta_d_model"] = np.array(d)

    eps = float(cap["meta_rms_eps"])
    out["ref_mlp"] = reference_mlp(out, eps)
    out["ref_attn"] = reference_attn(out, eps)
    return out


def _exactness_note_mlp(cap: dict, D: int, F: int) -> str:
    """The MLP's equivalent of `_exactness_note`."""
    full = int(cap["meta_d_model_full"])
    if not _sliced(cap, D):
        return (f"// gate/up are EXACT real llama values (the projection input is the full {D});\n"
                f"// down_proj is an honest PARTIAL SUM over the {F} captured neurons, and REF_MLP is\n"
                f"// truncated the same way.")
    return (f"// NOT the model's own values. Every weight is real and at its real index, but the\n"
            f"// hidden axis is cut to {D} of {full}: RMSNorm normalizes over {D} features and\n"
            f"// gate/up are partial sums over {D} of {full} inputs, on top of down_proj's existing\n"
            f"// truncation to {F} neurons. REF_MLP is recomputed on the SAME slice, so it grades\n"
            f"// what the device computes; this is a structural test, not a model result.")


def bf16_exact(x: np.ndarray, what: str) -> np.ndarray:
    """BF16 bit patterns for `x`, asserting the round trip is LOSSLESS.

    The model was loaded in bfloat16, so its weights and activations are exactly representable and
    baking them as 2 bytes costs nothing. If that ever stops being true the assertion says so rather
    than quietly halving the precision of an operand.
    """
    bits = G.bf16_bits(x)
    back = (bits.astype(np.uint32) << 16).view(np.float32).reshape(x.shape)
    if not np.array_equal(back, x.astype(np.float32)):
        i = int(np.argmax((back != x).ravel()))
        raise AssertionError(f"{what} is not exactly BF16 at flat index {i}: "
                             f"{x.ravel()[i]!r} -> {back.ravel()[i]!r}")
    return bits


def mesh(A_P: np.ndarray, A_scales: np.ndarray, B_P: np.ndarray, B_scales: np.ndarray) -> np.ndarray:
    """One matmul through the bit-exact mesh model: raw code products, block scales applied after."""
    C = G._run_mesh(A_P, A_scales, B_P, B_scales, FMT)
    return C


def build_mlp(cap: dict) -> dict:
    """The whole MLP chain: host stages in fp32, mesh stages through the datapath model."""
    eps = float(cap["meta_rms_eps"])
    h_mid, w_ln = cap["h_mid"], cap["w_post_ln"]
    Wg, Wu, Wd = cap["Wg"], cap["Wu"], cap["Wd"]
    M, D = h_mid.shape
    F = Wg.shape[1]
    print(f"  shape  M={M} D={D} F={F}   layer={int(cap['meta_layer'])} "
          f"neurons [{int(cap['meta_neuron0'])}, {int(cap['meta_neuron0']) + F})")

    # --- host stage 0: RMSNorm, then MX-quantize as the mesh's A operand ---
    xn = rmsnorm(h_mid, w_ln, eps)
    xn_codes, xn_scales, xn_P = G.quantize(xn, axis="row", f=FMT)          # scales [M][D/32]

    # --- mesh stages 1 and 2: the two projections, BF16 out (the host consumes them in fp32) ---
    Wg_codes, Wg_scales, Wg_P = G.quantize(Wg, axis="col", f=FMT)          # scales [D/32][F]
    Wu_codes, Wu_scales, Wu_P = G.quantize(Wu, axis="col", f=FMT)
    t0 = time.time()
    G_bf16 = mesh(xn_P, xn_scales, Wg_P, Wg_scales)
    U_bf16 = mesh(xn_P, xn_scales, Wu_P, Wu_scales)
    print(f"  mesh   G,U {G_bf16.shape} |max|={np.abs(G_bf16).max():.6g},"
          f"{np.abs(U_bf16).max():.6g}   ({time.time() - t0:.1f}s)")

    # --- host stage 3: SwiGLU on the values the device actually produced, then quantize ---
    h = silu(G_bf16) * U_bf16
    h_codes, h_scales, h_P = G.quantize(h, axis="row", f=FMT)              # scales [M][F/32]

    # --- mesh stage 4: down_proj ---
    Wd_codes, Wd_scales, Wd_P = G.quantize(Wd, axis="col", f=FMT)          # scales [F/32][D]
    t0 = time.time()
    Y_bf16 = mesh(h_P, h_scales, Wd_P, Wd_scales)
    print(f"  mesh   Y {Y_bf16.shape} |max|={np.abs(Y_bf16).max():.6g}   ({time.time() - t0:.1f}s)")

    # --- host stage 5: the residual ---
    out = h_mid + Y_bf16

    ref = cap["ref_mlp"]
    rel = float(np.linalg.norm(Y_bf16 - ref) / np.linalg.norm(ref))
    print(f"  grade  MX chain vs fp32 reference: rel_fro = {100 * rel:.4f}%")

    return dict(M=M, D=D, F=F, eps=eps, rel=rel,
                h_mid=h_mid, w_ln=w_ln,
                xn_codes=xn_codes, xn_scales=xn_scales,
                Wg_codes=Wg_codes, Wg_scales=Wg_scales,
                Wu_codes=Wu_codes, Wu_scales=Wu_scales,
                Wd_codes=Wd_codes, Wd_scales=Wd_scales,
                G_bf16=G_bf16, U_bf16=U_bf16,
                h_codes=h_codes, h_scales=h_scales,
                Y_bf16=Y_bf16, ref=ref, out=out + 0.0)


def _sliced(cap: dict, D: int) -> bool:
    return D != int(cap["meta_d_model_full"])


def _hidden_note(cap: dict, D: int) -> str:
    """How to describe the hidden axis in a generated header: full, or a stated slice."""
    full = int(cap["meta_d_model_full"])
    if not _sliced(cap, D):
        return f"hidden size {D} kept FULL"
    return (f"hidden size SLICED to the first {D} of {full} -- RMSNorm normalizes over {D}\n"
            f"// features and every projection is a partial sum over them, with the fp32 reference\n"
            f"// recomputed the same way (gen_llama_layer.slice_capture)")


def _d_comment(cap: dict, D: int) -> str:
    """The trailing comment on `#define LLAMA_D`. A slice must not call itself full."""
    full = int(cap["meta_d_model_full"])
    return "hidden size (full)" if not _sliced(cap, D) else f"hidden size (SLICED from {full})"


def _exactness_note(cap: dict, D: int, what: str, heads: int) -> str:
    """What the header may honestly claim about the projections it carries.

    At the full D the projection input is the whole residual stream, so the projected values ARE the
    model's own. Under a D-slice they are not: the normalization and the reduction both run over a
    subset, so the values are internally consistent real arithmetic on real weights -- but not the
    numbers TinyLlama computes.
    """
    full = int(cap["meta_d_model_full"])
    if not _sliced(cap, D):
        return (f"// {what} are EXACT real llama values -- the projection input is the full {D} --\n"
                f"// so S is the real score matrix for this head. Y is that head's own contribution\n"
                f"// to the layer output, an honest partial sum over 1 of {heads} heads, and the\n"
                f"// reference is truncated the same way.")
    return (f"// NOT the model's own values. Every WEIGHT and every input activation is real and at\n"
            f"// its real index, but the hidden axis is cut to {D} of {full}: RMSNorm normalizes over\n"
            f"// {D} features rather than {full}, so `xn` is not llama's normalized activation, and\n"
            f"// {what} are partial sums over {D} of {full} input features -- so S is not the real\n"
            f"// score matrix. Y is a {D}-wide slice of one of {heads} heads' contribution. The fp32\n"
            f"// reference is recomputed on the SAME slice, so it grades exactly what the device\n"
            f"// computes; this header is a structural test of the datapath, not a model result.")


def emit_mlp(cap: dict, d: dict, tag: str = "") -> Path:
    M, D, F = d["M"], d["D"], d["F"]
    GD, GF = D // BLOCK, F // BLOCK
    path = DATA / f"llama_mlp{tag}.h"
    guard = f"INCLUDE_LLAMA_MLP{tag.upper()}_H"
    r = G._rows
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by gen_llama_layer.py mlp -- do not edit by hand.
//
// A REAL TinyLlama MLP, back to back, at llama dimensions. Captured from one real forward pass by
// npu-exploration/app/capture_llama_layer.py: layer {int(cap['meta_layer'])}, FFN neurons
// [{int(cap['meta_neuron0'])}, {int(cap['meta_neuron0']) + F}) of {int(cap['meta_intermediate'])},
// {M} real tokens of wikitext2, {_hidden_note(cap, D)}. Quantized by MXQuant itself
// (quantize_mx_block32, block 32, {FMT.name}); mesh goldens from fp8_matmul_model.tiled_matmul_hwlike
// at the datapath's own precision schedule.
//
//   host   xn = rmsnorm(h_mid, w_post_ln)   fp32 -> MX codes + E8M0
//   mesh   G  = Xn @ Wg     [{M},{D}]x[{D},{F}] -> bf16
//   mesh   U  = Xn @ Wu
//   host   h  = silu(G) * U                 fp32 -> MX codes + E8M0
//   mesh   Y  = H  @ Wd     [{M},{F}]x[{F},{D}] -> bf16
//   host   out = h_mid + Y
//
{_exactness_note_mlp(cap, D, F)}
// The MX chain
// modelled here lands at rel_fro {100 * d['rel']:.4f}% of that fp32 reference.
#ifndef {guard}
#define {guard}

#include <stdint.h>

#define LLAMA_M   {M}      // tokens
#define LLAMA_D   {D}      // {_d_comment(cap, D)}
#define LLAMA_F   {F}      // FFN neurons captured
#define LLAMA_GD  {GD}      // D / 32, E8M0 blocks along the hidden axis
#define LLAMA_GF  {GF}      // F / 32
#define LLAMA_RMS_EPS {d['eps']:.10g}f

// ---- host inputs: the residual stream entering the MLP, and the RMSNorm weight ----
// BF16 bit patterns, LOSSLESS -- the model itself is bfloat16.
static const uint16_t H_MID_BF16[LLAMA_M][LLAMA_D] = {{
{r(bf16_exact(d['h_mid'], 'h_mid'), 4)}
}};

static const uint16_t W_POST_LN_BF16[LLAMA_D] = {{
    {", ".join("0x%04x" % int(v) for v in bf16_exact(d['w_ln'], 'w_post_ln'))}
}};

// ---- mesh operands: gate_proj and up_proj weights, [D][F] fp8 codes ----
static const uint8_t WG_IN[LLAMA_D][LLAMA_F] = {{
{r(d['Wg_codes'], 2)}
}};

// B-side scales: per column, per 32-element K group -- b_off = group * N + col
static const uint8_t WG_SCALES_COL[LLAMA_GD][LLAMA_F] = {{
{r(d['Wg_scales'], 2)}
}};

static const uint8_t WU_IN[LLAMA_D][LLAMA_F] = {{
{r(d['Wu_codes'], 2)}
}};

static const uint8_t WU_SCALES_COL[LLAMA_GD][LLAMA_F] = {{
{r(d['Wu_scales'], 2)}
}};

// ---- mesh operand: down_proj weight, [F][D] ----
static const uint8_t WD_IN[LLAMA_F][LLAMA_D] = {{
{r(d['Wd_codes'], 2)}
}};

static const uint8_t WD_SCALES_COL[LLAMA_GF][LLAMA_D] = {{
{r(d['Wd_scales'], 2)}
}};

// ---- golden for the HOST stages: what the reference fp32 produced, quantized ----
// Not a pass criterion -- the C computes its own and reports how many bytes differ. Expected 0:
// e4m3 keeps 3 mantissa bits, which absorbs a last-ulp difference between newlib and numpy.
static const uint8_t XN_CODES[LLAMA_M][LLAMA_D] = {{
{r(d['xn_codes'], 2)}
}};

// A-side scales: per row, per 32-element K group -- a_off = group * M + row (the TRANSPOSE of the
// [M][GD] layout the quantizer produces).
static const uint8_t XN_SCALES_ROW[LLAMA_GD][LLAMA_M] = {{
{r(d['xn_scales'].T, 2)}
}};

static const uint8_t H_CODES[LLAMA_M][LLAMA_F] = {{
{r(d['h_codes'], 2)}
}};

static const uint8_t H_SCALES_ROW[LLAMA_GF][LLAMA_M] = {{
{r(d['h_scales'].T, 2)}
}};

// ---- golden for the MESH stages: bit-exact, given the operand codes above ----
static const uint16_t G_OUT_BF16[LLAMA_M][LLAMA_F] = {{
{r(G.bf16_bits(d['G_bf16']), 4)}
}};

static const uint16_t U_OUT_BF16[LLAMA_M][LLAMA_F] = {{
{r(G.bf16_bits(d['U_bf16']), 4)}
}};

static const uint16_t Y_OUT_BF16[LLAMA_M][LLAMA_D] = {{
{r(G.bf16_bits(d['Y_bf16']), 4)}
}};

// ---- fp32 reference (bf16-rounded, ~0.4% -- far finer than the ~5% MX error it grades) ----
// The SLICED computation in fp32: same neurons, same truncated reduction as the device.
static const uint16_t REF_MLP_BF16[LLAMA_M][LLAMA_D] = {{
{r(G.bf16_bits(d['ref']), 4)}
}};

// h_mid + REF_MLP: the residual-stream output of this MLP slice.
static const uint16_t REF_OUT_BF16[LLAMA_M][LLAMA_D] = {{
{r(G.bf16_bits(d['h_mid'] + d['ref']), 4)}
}};

#endif // {guard}
""")
    return path


# --- attention -----------------------------------------------------------------------------------
#
# Six matmuls on the mesh, with the host in three of the five seams:
#
#   host   xn = rmsnorm(h_pre, w_in_ln)                     -> MX
#   mesh   Q, K, V = Xn @ Wq/Wk/Wv    [M,D]x[D,H] -> bf16   (Xn resident across all three)
#   host   RoPE on Q and K; K is TRANSPOSED here, because mx_loop_ws_spad ignores the transpose
#          bits that gemmini_loop_ws_spad's signature carries (gemmini.cc:1144 does `(void)rs1`)
#   mesh   S = Q @ K^T                [M,H]x[H,M] -> bf16
#   host   S/sqrt(H), causal mask, softmax                  -> MX
#   mesh   O = P @ V                  [M,M]x[M,H] -> FP8 REQUANT, tiled, straight to the scratchpad
#   mesh   Y = O @ Wo                 [M,H]x[H,D] -> bf16   <- O and its scales read IN PLACE
#   host   out = h_pre + Y
#
# The last seam is the point of this kernel: no host op sits between `P@V` and `O@Wo`, so the
# intermediate never leaves the device. The requantizer writes O into the scratchpad in the
# operand-A tiled layout and its E8M0 codes into the act-scale window, and o_proj reads both where
# they already are. That is the one place a real attention layer can use the chain path, and it is
# why O is requantized to FP8 here while every other intermediate comes back as BF16.


def _softmax_causal_bf16(S_bf16: np.ndarray, head_dim: int) -> np.ndarray:
    """1/sqrt(H), causal mask, row softmax -- in fp32 over the values the mesh actually produced."""
    from app.capture_llama_layer import softmax_causal
    return softmax_causal(S_bf16.astype(np.float32) / np.sqrt(np.float32(head_dim)))


def build_attn(cap: dict) -> dict:
    from app.capture_llama_layer import rope
    eps = float(cap["meta_rms_eps"])
    h_pre, w_ln = cap["h_pre"], cap["w_in_ln"]
    Wq, Wk, Wv, Wo = cap["Wq"], cap["Wk"], cap["Wv"], cap["Wo"]
    cos, sin = cap["rope_cos"], cap["rope_sin"]
    M, D = h_pre.shape
    H = Wq.shape[1]
    print(f"  shape  M={M} D={D} H={H}   layer={int(cap['meta_layer'])} "
          f"head={int(cap['meta_head'])} (kv head {int(cap['meta_kv_head'])})")

    # --- host: RMSNorm ---
    xn = rmsnorm(h_pre, w_ln, eps)
    xn_codes, xn_scales, xn_P = G.quantize(xn, axis="row", f=FMT)

    # --- mesh: the three projections, Xn resident ---
    qkv = {}
    for name, W in (("Q", Wq), ("K", Wk), ("V", Wv)):
        w_codes, w_scales, w_P = G.quantize(W, axis="col", f=FMT)
        qkv[name] = dict(codes=w_codes, scales=w_scales,
                         out=mesh(xn_P, xn_scales, w_P, w_scales))
    print(f"  mesh   Q,K,V {qkv['Q']['out'].shape} |max|="
          f"{np.abs(qkv['Q']['out']).max():.4g},{np.abs(qkv['K']['out']).max():.4g},"
          f"{np.abs(qkv['V']['out']).max():.4g}")

    # --- host: RoPE, then the operands for S = Q @ K^T ---
    q_rope = rope(qkv["Q"]["out"], cos, sin)
    k_rope = rope(qkv["K"]["out"], cos, sin)
    q_codes, q_scales, q_P = G.quantize(q_rope, axis="row", f=FMT)          # A: [M][H], [M][H/32]
    kt = np.ascontiguousarray(k_rope.T)                                     # [H][M]
    kt_codes, kt_scales, kt_P = G.quantize(kt, axis="col", f=FMT)           # B: [H][M], [H/32][M]

    # --- mesh: the scores ---
    S_bf16 = mesh(q_P, q_scales, kt_P, kt_scales)

    # --- host: scale, mask, softmax; V becomes a B operand ---
    P = _softmax_causal_bf16(S_bf16, H)
    p_codes, p_scales, p_P = G.quantize(P, axis="row", f=FMT)               # A: [M][M], [M][M/32]
    v_codes, v_scales, v_P = G.quantize(qkv["V"]["out"], axis="col", f=FMT)  # B: [M][H], [M/32][H]

    # --- mesh: O = P @ V, requantized to FP8 by the hardware (this is the resident operand) ---
    O_bf16 = mesh(p_P, p_scales, v_P, v_scales)
    o_codes, o_scales, o_P = G._requant(O_bf16, FMT)                        # [M][H], [M][H/32]
    print(f"  mesh   O = P@V requant: peak code 0x{int(np.max(o_codes & 0x7F)):02X}, "
          f"E8M0 {int(o_scales.min())}..{int(o_scales.max())}")

    # --- mesh: o_proj, reading O and its scales in place ---
    wo_codes, wo_scales, wo_P = G.quantize(Wo, axis="col", f=FMT)
    Y_bf16 = mesh(o_P, o_scales, wo_P, wo_scales)

    ref = cap["ref_attn"]
    rel = float(np.linalg.norm(Y_bf16 - ref) / np.linalg.norm(ref))
    print(f"  grade  MX chain vs fp32 reference: rel_fro = {100 * rel:.4f}%")

    return dict(M=M, D=D, H=H, eps=eps, rel=rel, h_pre=h_pre, w_ln=w_ln, cos=cos, sin=sin,
                xn_codes=xn_codes, xn_scales=xn_scales,
                qkv=qkv, q_rope=q_rope, k_rope=k_rope,
                q_codes=q_codes, q_scales=q_scales,
                kt_codes=kt_codes, kt_scales=kt_scales,
                S_bf16=S_bf16, P=P, p_codes=p_codes, p_scales=p_scales,
                v_codes=v_codes, v_scales=v_scales,
                O_bf16=O_bf16, o_codes=o_codes, o_scales=o_scales,
                wo_codes=wo_codes, wo_scales=wo_scales,
                Y_bf16=Y_bf16, ref=ref)


def _f32_rows(a: np.ndarray) -> str:
    """fp32 bit patterns, as C initializer rows. Used where BF16 would not be lossless."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ",\n".join("    { " + ", ".join("0x%08x" % int(v) for v in row) + " }" for row in u)


def emit_attn(cap: dict, d: dict, tag: str = "") -> Path:
    M, D, H = d["M"], d["D"], d["H"]
    GD, GH, GM = D // BLOCK, H // BLOCK, M // BLOCK
    path = DATA / f"llama_attn{tag}.h"
    guard = f"INCLUDE_LLAMA_ATTN{tag.upper()}_H"
    r = G._rows
    b = G.bf16_bits
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by gen_llama_layer.py attn -- do not edit by hand.
//
// A REAL TinyLlama attention head, back to back, at llama dimensions. Captured from one real
// forward pass by npu-exploration/app/capture_llama_layer.py: layer {int(cap['meta_layer'])},
// query head {int(cap['meta_head'])} of {int(cap['meta_n_heads'])} (GQA kv head
// {int(cap['meta_kv_head'])} of {int(cap['meta_n_kv_heads'])}), {M} real tokens of wikitext2,
// {_hidden_note(cap, D)}. Quantized by MXQuant itself (block 32, {FMT.name}); mesh goldens from
// fp8_matmul_model.tiled_matmul_hwlike at the datapath's own precision schedule.
//
//   host   xn = rmsnorm(h_pre, w_in_ln)      -> MX
//   mesh   Q,K,V = Xn @ Wq/Wk/Wv   [{M},{D}]x[{D},{H}] -> bf16   (Xn resident for all three)
//   host   RoPE on Q and K, and the K TRANSPOSE (the MX loop ignores the transpose bits)
//   mesh   S = Q @ K^T             [{M},{H}]x[{H},{M}] -> bf16
//   host   S/sqrt({H}), causal mask, softmax -> MX
//   mesh   O = P @ V               [{M},{M}]x[{M},{H}] -> FP8 requant, TILED, in the scratchpad
//   mesh   Y = O @ Wo              [{M},{H}]x[{H},{D}] -> bf16, O read IN PLACE
//   host   out = h_pre + Y
//
{_exactness_note(cap, D, "Q/K/V", int(cap['meta_n_heads']))}
// The MX chain modelled here lands at rel_fro {100 * d['rel']:.4f}% of that fp32 reference.
#ifndef {guard}
#define {guard}

#include <stdint.h>

#define LLAMA_M   {M}      // tokens (also the causal context)
#define LLAMA_D   {D}      // {_d_comment(cap, D)}
#define LLAMA_H   {H}      // head dim
#define LLAMA_GD  {GD}      // D / 32
#define LLAMA_GH  {GH}      // H / 32
#define LLAMA_GM  {GM}      // M / 32
#define LLAMA_RMS_EPS {d['eps']:.10g}f

// ---- host inputs ----
static const uint16_t H_PRE_BF16[LLAMA_M][LLAMA_D] = {{
{r(bf16_exact(d['h_pre'], 'h_pre'), 4)}
}};

static const uint16_t W_IN_LN_BF16[LLAMA_D] = {{
    {", ".join("0x%04x" % int(v) for v in bf16_exact(d['w_ln'], 'w_in_ln'))}
}};

// The model's own rotary tables for these positions, as fp32 bit patterns -- transformers computes
// them in fp32 and they need not be exactly BF16, so they are not forced through it.
static const uint32_t ROPE_COS_F32[LLAMA_M][LLAMA_H] = {{
{_f32_rows(d['cos'])}
}};

static const uint32_t ROPE_SIN_F32[LLAMA_M][LLAMA_H] = {{
{_f32_rows(d['sin'])}
}};

// ---- mesh operands: the four projection weights ----
static const uint8_t WQ_IN[LLAMA_D][LLAMA_H] = {{
{r(d['qkv']['Q']['codes'], 2)}
}};

static const uint8_t WQ_SCALES_COL[LLAMA_GD][LLAMA_H] = {{
{r(d['qkv']['Q']['scales'], 2)}
}};

static const uint8_t WK_IN[LLAMA_D][LLAMA_H] = {{
{r(d['qkv']['K']['codes'], 2)}
}};

static const uint8_t WK_SCALES_COL[LLAMA_GD][LLAMA_H] = {{
{r(d['qkv']['K']['scales'], 2)}
}};

static const uint8_t WV_IN[LLAMA_D][LLAMA_H] = {{
{r(d['qkv']['V']['codes'], 2)}
}};

static const uint8_t WV_SCALES_COL[LLAMA_GD][LLAMA_H] = {{
{r(d['qkv']['V']['scales'], 2)}
}};

static const uint8_t WO_IN[LLAMA_H][LLAMA_D] = {{
{r(d['wo_codes'], 2)}
}};

static const uint8_t WO_SCALES_COL[LLAMA_GH][LLAMA_D] = {{
{r(d['wo_scales'], 2)}
}};

// ---- goldens for the HOST stages (reported, not a pass criterion) ----
static const uint8_t XN_CODES[LLAMA_M][LLAMA_D] = {{
{r(d['xn_codes'], 2)}
}};

static const uint8_t XN_SCALES_ROW[LLAMA_GD][LLAMA_M] = {{
{r(d['xn_scales'].T, 2)}
}};

// Q after RoPE, as the mesh's A operand.
static const uint8_t Q_CODES[LLAMA_M][LLAMA_H] = {{
{r(d['q_codes'], 2)}
}};

static const uint8_t Q_SCALES_ROW[LLAMA_GH][LLAMA_M] = {{
{r(d['q_scales'].T, 2)}
}};

// K after RoPE AND transposed, as the mesh's B operand: [H][M], scales [H/32][M].
static const uint8_t KT_IN[LLAMA_H][LLAMA_M] = {{
{r(d['kt_codes'], 2)}
}};

static const uint8_t KT_SCALES_COL[LLAMA_GH][LLAMA_M] = {{
{r(d['kt_scales'], 2)}
}};

// The softmax output, as the mesh's A operand.
static const uint8_t P_CODES[LLAMA_M][LLAMA_M] = {{
{r(d['p_codes'], 2)}
}};

static const uint8_t P_SCALES_ROW[LLAMA_GM][LLAMA_M] = {{
{r(d['p_scales'].T, 2)}
}};

// V as the mesh's B operand: blocked along the token axis, one scale per (group, head-dim column).
static const uint8_t V_IN[LLAMA_M][LLAMA_H] = {{
{r(d['v_codes'], 2)}
}};

static const uint8_t V_SCALES_COL[LLAMA_GM][LLAMA_H] = {{
{r(d['v_scales'], 2)}
}};

// ---- goldens for the MESH stages: bit-exact, given the operand codes above ----
static const uint16_t Q_OUT_BF16[LLAMA_M][LLAMA_H] = {{
{r(b(d['qkv']['Q']['out']), 4)}
}};

static const uint16_t K_OUT_BF16[LLAMA_M][LLAMA_H] = {{
{r(b(d['qkv']['K']['out']), 4)}
}};

static const uint16_t V_OUT_BF16[LLAMA_M][LLAMA_H] = {{
{r(b(d['qkv']['V']['out']), 4)}
}};

static const uint16_t S_OUT_BF16[LLAMA_M][LLAMA_M] = {{
{r(b(d['S_bf16']), 4)}
}};

// O = P @ V as the REQUANTIZER emits it: FP8 codes plus one E8M0 byte per row per 32 output
// columns. These are what must be resident in the scratchpad and in the act-scale window for
// o_proj to read them in place -- C1_out / C1_scales_out in matmul_tiled_fp8_64x64_chain.c terms.
static const uint8_t O_OUT[LLAMA_M][LLAMA_H] = {{
{r(d['o_codes'], 2)}
}};

static const uint8_t O_SCALES_OUT[LLAMA_M][LLAMA_GH] = {{
{r(d['o_scales'], 2)}
}};

static const uint16_t O_OUT_BF16[LLAMA_M][LLAMA_H] = {{
{r(b(d['O_bf16']), 4)}
}};

static const uint16_t Y_OUT_BF16[LLAMA_M][LLAMA_D] = {{
{r(b(d['Y_bf16']), 4)}
}};

// ---- fp32 reference, truncated to this head exactly as the device is ----
static const uint16_t REF_ATTN_BF16[LLAMA_M][LLAMA_D] = {{
{r(b(d['ref']), 4)}
}};

static const uint16_t REF_OUT_BF16[LLAMA_M][LLAMA_D] = {{
{r(b(d['h_pre'] + d['ref']), 4)}
}};

#endif // {guard}
""")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", default=["mlp", "attn"],
                    help="which headers to emit: mlp, attn (default: both)")
    ap.add_argument("--d", type=int, default=None, metavar="D",
                    help="cut the hidden size to the first D features (multiple of 32). The host's "
                         "fp32 glue is 99%% of the cycles and scales linearly in D, so this is the "
                         "knob that makes an RTL run affordable. Default: the capture's full D.")
    ap.add_argument("--tag", default=None, metavar="SUFFIX",
                    help="suffix for the emitted header names, e.g. --tag _small writes "
                         "include/llama_attn_small.h. Defaults to _d<D> when --d is given.")
    args = ap.parse_args()
    want = args.targets or ["mlp", "attn"]
    unknown = [w for w in want if w not in ("mlp", "attn")]
    if unknown:
        raise SystemExit(f"unknown target(s) {unknown}; known: mlp, attn")

    cap = load_capture()
    tag = ""
    if args.d is not None and args.d != int(cap["meta_d_model"]):
        cap = slice_capture(cap, args.d)
        tag = args.tag if args.tag is not None else f"_d{args.d}"
    elif args.tag:
        tag = args.tag

    for target in want:
        print(f"llama_{target}{tag}.h  from {cap['_path'].name}")
        if target == "mlp":
            p = emit_mlp(cap, build_mlp(cap), tag)
        else:
            p = emit_attn(cap, build_attn(cap), tag)
        print(f"  wrote {p.relative_to(DATA.parent)}  ({p.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
