#!/usr/bin/env python3
"""Generate the data for `src/llama_mlp_full.c` -- a COMPLETE TinyLlama MLP sub-layer: all 5632
FFN neurons, the full 2048x5632 gate/up and 5632x2048 down projections.

Why this one is different from `llama_mlp.h`. That kernel runs 64 of 5632 neurons, so `down_proj`
is a partial sum over ~1% of the reduction and the only available reference is a numpy
reimplementation of that slice. With every neuron present nothing is truncated, so the result is the
layer's REAL MLP output -- and the capture now carries `mlp_torch`, the tensor TinyLlama's own `mlp`
module produced on the same tokens. That is the MLP's counterpart to what `--all-heads` bought
attention, and it is a reference this repo could not previously write down.

TWO THINGS MAKE THIS A NEW SHAPE, not a bigger `llama_mlp.h`:

  1. `down_proj` contracts over K = 5632, deeper than the B-side scale window permits in ONE call:
     `SCALE_ROWS(5632, 64) = 704` against the 256 rows `ScaleFactorMem` holds
     (`planning/rtl_fault_b_kdepth.md`). So it MUST be split into accumulating K-tiles -- the first
     kernel here forced into that at full scale. `mxl5` proved the split is bit-exact on RTL.
  2. `[M][F]` at F = 5632 is 22528 scratchpad rows of BF16 output against 16384 available, so the
     gate/up projections cannot land whole either and are chunked on F.

Neither is a limitation of the data: an output column depends only on its own column of B, so the
unchunked model here is the golden for any chunking the C picks.

GOLDENS RUN IN PARALLEL. The datapath model is 0.42 MMAC/s and this chain is ~1.1 GMAC -- 44
minutes on one core. `mesh_par.mesh_parallel` splits the output columns across every core and is
checked bit-identical to the serial path (`mesh_par.py` self-test).

    PATH=../../.venv/bin:$PATH ../../.venv/bin/python3 gen_llama_mlp_full.py

Needs an --all-neurons capture:
    cd ../../.. && .venv/bin/python3 -m kernels.captures.llama_layer --all-heads --all-neurons
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

#: gen/ -> mxgemmini/ -> baremetal/ -> the repo root.
HERE = Path(__file__).resolve().parent
NPU = HERE.parents[2]
DATA = HERE.parent / "data"
ROCC = NPU.parent / "software" / "gemmini-rocc-tests"
if not (NPU / "compiler" / "operands.py").exists():
    raise SystemExit(f"npu-exploration not found at {NPU}")
sys.path.insert(0, str(NPU))
sys.path.insert(0, str(ROCC))
sys.path.insert(0, str(HERE))
DATA.mkdir(parents=True, exist_ok=True)

import gen_matmul_llama as G      # noqa: E402  -- quantize(), bf16_bits()
import gen_llama_layer as GL      # noqa: E402  -- FMT, bf16_exact()
from mesh_par import mesh_parallel  # noqa: E402
from kernels.host_math import rmsnorm, silu  # noqa: E402  -- the device's host math, bit-exact

FMT = GL.FMT
BLOCK = 32
CAPTURE = NPU / "out" / "layer_capture"


def load_all_neurons() -> dict:
    """The newest --all-neurons capture. A sliced one cannot be used: `Wd` covers F of 5632."""
    cands = sorted(CAPTURE.glob("layer*_alln_*.npz"))
    if not cands:
        raise SystemExit(
            f"no --all-neurons capture in {CAPTURE}\nRun it first:\n"
            f"    cd {NPU} && .venv/bin/python3 -m kernels.captures.llama_layer "
            f"--all-heads --all-neurons")
    with np.load(cands[-1]) as z:
        d = {k: z[k] for k in z.files}
    d["_path"] = cands[-1]
    return d


class Blob:
    """Append-only binary image; every tensor 64-byte aligned so the DMA never straddles oddly.

    Same contract as gen_llama_attn_full.py's: `add` records an offset the header turns into
    `#define LLAMA_OFF_<NAME>`, and the C reads `LLAMA_BLOB + LLAMA_OFF_<NAME>`.
    """

    def __init__(self) -> None:
        self.buf = bytearray()
        self.off: dict[str, int] = {}
        self.desc: list[tuple[str, int, int, str]] = []

    def add(self, name: str, arr: np.ndarray, dtype) -> int:
        a = np.ascontiguousarray(arr, dtype=dtype)
        pad = (-len(self.buf)) % 64
        self.buf.extend(b"\0" * pad)
        o = len(self.buf)
        self.off[name] = o
        self.buf.extend(a.tobytes())
        self.desc.append((name, o, a.nbytes, str(a.shape)))
        return o


def mlp_chain(cap: dict, b: Blob, x_in: np.ndarray, w_ln: np.ndarray,
              prefix: str = "") -> tuple[np.ndarray, dict]:
    """The whole MLP sub-layer, from a residual-stream input to `Y = H @ Wd`.

    Factored out of `build` for the same reason as `attention_chain`: a complete decoder layer runs
    this as its SECOND half, on the residual the device itself produced, and a second copy of the
    chain would drift. `prefix` namespaces the blob entries -- see `attention_chain` on why a
    duplicate name is a silent corruption rather than an error.

    Appends every operand and per-stage golden to `b`; returns (Y, dims).
    """
    def add(n, a, dt):
        return b.add(prefix + n, a, dt)
    M, D = x_in.shape
    F = cap["Wg"].shape[1]
    eps = float(cap["meta_rms_eps"])
    print(f"  shape  M={M} D={D} F={F} (ALL neurons)   layer={int(cap['meta_layer'])}")
    assert cap["Wg"].shape == (D, F) and cap["Wd"].shape == (F, D), \
        f"capture is not --all-neurons: Wg {cap['Wg'].shape}, Wd {cap['Wd'].shape}"
    assert F == int(cap["meta_intermediate"]), \
        f"F {F} is not the model's intermediate_size {int(cap['meta_intermediate'])}"

    # --- host: RMSNorm over the FULL D, then the mesh's A operand ---
    xn = rmsnorm(x_in, w_ln, eps)
    xn_codes, xn_scales, xn_P = G.quantize(xn, axis="row", f=FMT)      # scales [M][D/32]
    add("XN_CODES", xn_codes, np.uint8)
    add("XN_SCALES", xn_scales.T, np.uint8)                            # A window: [D/32][M]

    # --- mesh: gate and up, [M,D]x[D,F]. F-chunked in the C; the unchunked model is the golden. ---
    outs = {}
    for name, W in (("WG", cap["Wg"]), ("WU", cap["Wu"])):
        w_codes, w_scales, w_P = G.quantize(W, axis="col", f=FMT)      # scales [D/32][F]
        add(f"{name}_CODES", w_codes, np.uint8)
        add(f"{name}_SCALES", w_scales, np.uint8)
        outs[name] = mesh_parallel(xn_P, xn_scales, w_P, w_scales, FMT,
                                   label=f"{name[1]} = Xn @ {name}")
    G_bf16, U_bf16 = outs["WG"], outs["WU"]
    add("G_OUT", G.bf16_bits(G_bf16), np.uint16)
    add("U_OUT", G.bf16_bits(U_bf16), np.uint16)

    # --- host: SwiGLU on the values the device actually produced, then quantize as operand A ---
    h = silu(G_bf16) * U_bf16
    h_codes, h_scales, h_P = G.quantize(h, axis="row", f=FMT)          # scales [M][F/32]
    add("H_CODES", h_codes, np.uint8)
    add("H_SCALES", h_scales.T, np.uint8)                              # A window: [F/32][M]

    # --- mesh: down_proj, [M,F]x[F,D]. K = 5632 exceeds the scale window, so the C splits it into
    #     accumulating K-tiles; mxl5 proved that split bit-exact, so the whole-K model is the
    #     golden for it exactly as the unchunked projections are for their F-chunking. ---
    wd_codes, wd_scales, wd_P = G.quantize(cap["Wd"], axis="col", f=FMT)   # scales [F/32][D]
    add("WD_CODES", wd_codes, np.uint8)
    add("WD_SCALES", wd_scales, np.uint8)
    Y = mesh_parallel(h_P, h_scales, wd_P, wd_scales, FMT, label="Y = H @ Wd")
    add("Y_OUT", G.bf16_bits(Y), np.uint16)

    return Y, dict(M=M, D=D, F=F, eps=eps, layer=int(cap["meta_layer"]))


def build(cap: dict) -> tuple[Blob, dict]:
    """The standalone MLP kernel's blob: the chain, plus the references it is graded on."""
    b = Blob()
    b.add("H_MID", GL.bf16_exact(cap["h_mid"], "h_mid"), np.uint16)
    b.add("W_POST_LN", GL.bf16_exact(cap["w_post_ln"], "w_post_ln"), np.uint16)
    Y, d = mlp_chain(cap, b, cap["h_mid"], cap["w_post_ln"])

    ref = cap["ref_mlp"]
    mt = cap["mlp_torch"] if "mlp_torch" in cap else ref
    b.add("REF_MLP", G.bf16_bits(ref), np.uint16)
    b.add("MLP_TORCH", G.bf16_bits(mt), np.uint16)
    rel = float(np.linalg.norm(Y - ref) / np.linalg.norm(ref))
    rel_t = float(np.linalg.norm(Y - mt) / np.linalg.norm(mt))
    print(f"  grade  MX chain vs fp32 reference        : rel_fro = {100 * rel:.4f}%")
    print(f"  grade  MX chain vs the MODEL's own output: rel_fro = {100 * rel_t:.4f}%")

    d.update(rel=rel, rel_torch=rel_t)
    return b, d


def emit(b: Blob, d: dict) -> tuple[Path, Path]:
    bin_path = DATA / "llama_mlp_full.bin"
    hdr_path = DATA / "llama_mlp_full.h"
    bin_path.write_bytes(bytes(b.buf))

    offs = "\n".join(f"#define LLAMA_OFF_{n:<12s} {o}u" for n, o in b.off.items())
    table = "\n".join(f"//   {n:<12s} @ {o:>9d}  {sz:>9d} B  {sh}" for n, o, sz, sh in b.desc)
    with open(hdr_path, "w") as fh:
        fh.write(f"""// GENERATED by gen_llama_mlp_full.py -- do not edit by hand.
//
// A COMPLETE TinyLlama MLP sub-layer: ALL {d['F']} FFN neurons, the full {d['D']}x{d['F']} gate/up
// and {d['F']}x{d['D']} down projections. Layer {d['layer']}, {d['M']} real tokens of wikitext2,
// captured by kernels/captures/llama_layer.py --all-neurons. Quantized by MXQuant (block 32,
// {FMT.name}); mesh goldens from fp8_matmul_model.tiled_matmul_hwlike at the datapath's own
// precision schedule, run column-parallel (mesh_par.py, bit-identical to the serial path).
//
// NOTHING IS TRUNCATED. Unlike llama_mlp.h -- 64 of {d['F']} neurons, so down_proj is a partial sum
// over ~1% of the reduction -- every neuron is present and the hidden size is full, so the result
// IS this layer's real MLP output. MLP_TORCH is what TinyLlama's own mlp module produced on these
// tokens, which makes this gradeable against the model rather than against a numpy slice.
//
//   host   xn = rmsnorm(h_mid, w_post_ln)                   -> MX
//   mesh   G = Xn @ Wg   [{d['M']},{d['D']}]x[{d['D']},{d['F']}]      (F-chunked in the C)
//   mesh   U = Xn @ Wu   [{d['M']},{d['D']}]x[{d['D']},{d['F']}]      (Xn resident across both)
//   host   H = silu(G) * U                                  -> MX
//   mesh   Y = H @ Wd    [{d['M']},{d['F']}]x[{d['F']},{d['D']}]      (K-TILED: K={d['F']} exceeds
//                                                             the scale window in one call)
//   host   out = h_mid + Y
//
// The MX chain lands at rel_fro {100 * d['rel']:.4f}% of the fp32 reference, and
// {100 * d['rel_torch']:.4f}% of the model's own bf16 MLP output.
//
// DATA LIVES IN llama_mlp_full.bin, linked as a binary section -- {len(b.buf) / 1e6:.1f} MB, which as C
// initializers would be ~{6 * len(b.buf) / 1e6:.0f} MB of source. Offsets are into that blob:
//
{table}
#ifndef INCLUDE_LLAMA_MLP_FULL_H
#define INCLUDE_LLAMA_MLP_FULL_H

#include <stdint.h>

#define LLAMA_M   {d['M']}      // tokens
#define LLAMA_D   {d['D']}      // hidden size, FULL
#define LLAMA_F   {d['F']}      // FFN neurons -- ALL of them
#define LLAMA_GD  {d['D'] // BLOCK}      // D / 32
#define LLAMA_GF  {d['F'] // BLOCK}      // F / 32
#define LLAMA_GM  {d['M'] // BLOCK}      // M / 32
#define LLAMA_RMS_EPS {d['eps']:.10g}f

// The blob, linked by objcopy -I binary (see the Makefile).
extern const uint8_t _binary_llama_mlp_full_bin_start[];
#define LLAMA_BLOB _binary_llama_mlp_full_bin_start
#define LLAMA_AT(off, type) ((const type *) (LLAMA_BLOB + (off)))

{offs}

#endif // INCLUDE_LLAMA_MLP_FULL_H
""")
    return bin_path, hdr_path


def main() -> int:
    cap = load_all_neurons()
    print(f"llama_mlp_full  from {cap['_path'].name}")
    t0 = time.time()
    b, d = build(cap)
    bp, hp = emit(b, d)
    print(f"  wrote {bp.relative_to(DATA.parent)}  ({bp.stat().st_size / 1e6:.1f} MB)")
    print(f"  wrote {hp.relative_to(DATA.parent)}  ({hp.stat().st_size / 1e3:.1f} kB)")
    print(f"  total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
