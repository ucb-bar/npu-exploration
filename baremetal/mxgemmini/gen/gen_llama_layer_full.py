#!/usr/bin/env python3
"""Generate the data for `src/llama_layer_full.c` -- one COMPLETE TinyLlama decoder layer.

This is the first kernel here that is a LAYER rather than a sub-layer: attention (all 32 heads) and
the MLP (all 5632 neurons) joined through both RMSNorms and both residuals, exactly as
`LlamaDecoderLayer.forward` does it:

    h_mid = h_pre + attn(rmsnorm(h_pre,  w_in_ln))
    h_out = h_mid + mlp (rmsnorm(h_mid,  w_post_ln))

WHAT MAKES IT MORE THAN TWO HEADERS CONCATENATED. `llama_mlp_full` is given `h_mid` from the
capture -- torch's own value. Here `h_mid` is what the DEVICE produced: `h_pre` plus an attention
output that carries the full MX error of 32 heads and four matmul stages. So every MLP-side golden
has to be re-derived from that value, and none of `llama_mlp_full.bin`'s bytes can be reused. The
RMSNorm of a perturbed input is a different vector, its fp8 codes are different, and the errors
COMPOUND across the seam rather than adding. That compounding is the quantity this kernel exists to
measure, and it is the reason a stacked model cannot be predicted from per-sub-layer numbers.

THE SEAM MUST BE BIT-EXACT, NOT MERELY CLOSE. `h_mid` is quantized immediately afterwards, so a
one-ulp difference between this golden's residual and the kernel's can flip an E4M3 code and cascade
through the whole MLP. `_bf16_rne` below therefore mirrors `mx_host.h:mx_f32_to_bf16_rne` exactly --
including its treatment of -0.0, where `gen_matmul_llama.bf16_bits` deliberately differs (it forces
+0). That difference has never mattered before because no golden had ever fed a residual RESULT back
into a quantizer; here one does.

The grading target is `h_out` from the capture -- the layer's true output, straight out of
TinyLlama's own forward pass. Neither sub-layer kernel could be graded against it: one produces
attention alone, the other an MLP over torch's residual.

    PATH=../../.venv/bin:$PATH ../../.venv/bin/python3 gen_llama_layer_full.py

Needs a capture with BOTH slices removed:
    cd ../../.. && .venv/bin/python3 -m app.capture_llama_layer --all-heads --all-neurons
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

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

import gen_matmul_llama as G          # noqa: E402
import gen_llama_layer as GL          # noqa: E402  -- FMT, bf16_exact()
import gen_llama_attn_full as GA      # noqa: E402  -- Blob, attention_chain()
import gen_llama_mlp_full as GM       # noqa: E402  -- mlp_chain()

FMT = GL.FMT
BLOCK = 32
CAPTURE = NPU / "out" / "layer_capture"


def load_full_layer() -> dict:
    """The newest capture with BOTH slices removed. A half-sliced one cannot build a whole layer."""
    cands = sorted(CAPTURE.glob("layer*_allheads_alln_*.npz"))
    if not cands:
        raise SystemExit(
            f"no --all-heads --all-neurons capture in {CAPTURE}\nRun it first:\n"
            f"    cd {NPU} && .venv/bin/python3 -m app.capture_llama_layer "
            f"--all-heads --all-neurons")
    with np.load(cands[-1]) as z:
        d = {k: z[k] for k in z.files}
    d["_path"] = cands[-1]
    return d


def _bf16_rne(x: np.ndarray) -> np.ndarray:
    """fp32 -> BF16 bits, mirroring mx_host.h:mx_f32_to_bf16_rne EXACTLY.

    `gen_matmul_llama.bf16_bits` is the same function plus `where(x == 0, +0)`, which maps -0.0 to
    0x0000 where the C produces 0x8000. Harmless for a mesh golden; NOT harmless for a residual
    whose result is re-quantized, which is the only place this is used.
    """
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    nan = ((u >> 23) & 0xFF) == 0xFF
    lsb = (u >> 16) & 1
    r = np.where(nan, u >> 16, (u + 0x7FFF + lsb) >> 16)
    return (r & 0xFFFF).astype(np.uint16)


def _bf16_val(bits: np.ndarray) -> np.ndarray:
    """BF16 bits -> the fp32 value they denote, as the host's mx_bf16_to_f32 reads them back."""
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(bits.shape)


def residual(x_bits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """`out = bf16(f32(x) + f32(y))` -- the C's residual, to the bit.

    Both operands are already BF16-representable (`x` as stored bits, `y` as a mesh output), the sum
    is taken in fp32 and rounded once. Returns BF16 bits.
    """
    return _bf16_rne(_bf16_val(x_bits) + np.asarray(y, dtype=np.float32))


def build(cap: dict) -> tuple[GA.Blob, dict]:
    M, D = cap["h_pre"].shape
    F = cap["Wg"].shape[1]
    eps = float(cap["meta_rms_eps"])

    b = GA.Blob()
    h_pre_bits = GL.bf16_exact(cap["h_pre"], "h_pre")
    b.add("H_PRE", h_pre_bits, np.uint16)
    b.add("W_IN_LN", GL.bf16_exact(cap["w_in_ln"], "w_in_ln"), np.uint16)
    b.add("W_POST_LN", GL.bf16_exact(cap["w_post_ln"], "w_post_ln"), np.uint16)

    # ---- first half: attention, on the layer's input ----
    print("  [attention]")
    t0 = time.time()
    Yattn, da = GA.attention_chain(cap, b, cap["h_pre"], cap["w_in_ln"], prefix="A_")
    print(f"  [attention] done ({time.time() - t0:.1f}s)")

    # ---- the seam: residual 1, computed the way the DEVICE computes it ----
    h_mid_bits = residual(h_pre_bits, Yattn)
    h_mid = _bf16_val(h_mid_bits)
    b.add("H_MID_OUT", h_mid_bits, np.uint16)          # golden for the kernel's own residual
    drift = float(np.linalg.norm(h_mid - cap["h_mid"]) / np.linalg.norm(cap["h_mid"]))
    print(f"  seam   h_mid = h_pre + Yattn  vs the model's own h_mid: rel_fro = {100 * drift:.4f}%"
          f"   <- the MLP half runs on THIS, not on the capture's")

    # ---- second half: the MLP, on the residual the device produced ----
    print("  [mlp]")
    t0 = time.time()
    Ymlp, dm = GM.mlp_chain(cap, b, h_mid, cap["w_post_ln"], prefix="M_")
    print(f"  [mlp] done ({time.time() - t0:.1f}s)")

    # ---- residual 2, and the layer's output ----
    h_out_bits = residual(h_mid_bits, Ymlp)
    h_out = _bf16_val(h_out_bits)
    b.add("H_OUT_OUT", h_out_bits, np.uint16)
    b.add("H_OUT_TORCH", G.bf16_bits(cap["h_out"]), np.uint16)
    # Context, not gates: what each half would have scored on its own.
    b.add("ATTN_TORCH", G.bf16_bits(cap["attn_torch"]), np.uint16)
    b.add("MLP_TORCH", G.bf16_bits(cap["mlp_torch"]), np.uint16)

    rel_out = float(np.linalg.norm(h_out - cap["h_out"]) / np.linalg.norm(cap["h_out"]))
    rel_a = float(np.linalg.norm(Yattn - cap["attn_torch"]) / np.linalg.norm(cap["attn_torch"]))
    rel_m = float(np.linalg.norm(Ymlp - cap["mlp_torch"]) / np.linalg.norm(cap["mlp_torch"]))
    print(f"  grade  attention out vs the model's        : rel_fro = {100 * rel_a:.4f}%")
    print(f"  grade  MLP out       vs the model's        : rel_fro = {100 * rel_m:.4f}%   "
          f"(on the DEVICE's h_mid, so above llama_mlp_full's own number)")
    print(f"  grade  LAYER out h_out vs THE MODEL's h_out: rel_fro = {100 * rel_out:.4f}%")
    print(f"         (the residual stream is {np.linalg.norm(cap['h_pre']) / np.linalg.norm(Ymlp):.0f}x "
          f"larger than the MLP output it carries, which is why this is far below either half)")

    return b, dict(M=M, D=D, F=F, H=da["H"], NH=da["NH"], NKV=da["NKV"], PER=da["PER"],
                   QD=da["QD"], KVD=da["KVD"], eps=eps, layer=int(cap["meta_layer"]),
                   rel_out=rel_out, rel_a=rel_a, rel_m=rel_m, seam_drift=drift)


def emit(b: GA.Blob, d: dict) -> tuple[Path, Path]:
    bin_path = DATA / "llama_layer_full.bin"
    hdr_path = DATA / "llama_layer_full.h"
    bin_path.write_bytes(bytes(b.buf))

    offs = "\n".join(f"#define LLAMA_OFF_{n:<14s} {o}u" for n, o in b.off.items())
    table = "\n".join(f"//   {n:<14s} @ {o:>9d}  {sz:>9d} B  {sh}" for n, o, sz, sh in b.desc)
    with open(hdr_path, "w") as fh:
        fh.write(f"""// GENERATED by gen_llama_layer_full.py -- do not edit by hand.
//
// ONE COMPLETE TinyLlama DECODER LAYER: attention (all {d['NH']} heads) and the MLP (all {d['F']}
// neurons) joined by both RMSNorms and both residuals. Layer {d['layer']}, {d['M']} real tokens of
// wikitext2. Quantized by MXQuant (block 32, {FMT.name}); mesh goldens from
// fp8_matmul_model.tiled_matmul_hwlike, run column-parallel (mesh_par.py).
//
//   host   xn1   = rmsnorm(h_pre, w_in_ln)                   -> MX
//   mesh   Q,K,V = Xn1 @ Wq/Wk/Wv        (Xn1 resident)
//   host   RoPE per head; K transposed per kv head
//   mesh   per head  S_h = Q_h @ K_kv^T ; host causal softmax ; O_h = P_h @ V_kv -> FP8 requant
//   mesh   Yattn = sum_h O_h @ Wo_h      (accumulated in smem across heads)
//   host   h_mid = h_pre + Yattn                             <-- RESIDUAL 1, the seam
//   host   xn2   = rmsnorm(h_mid, w_post_ln)                 -> MX
//   mesh   G,U   = Xn2 @ Wg/Wu           (F-chunked, Xn2 resident)
//   host   H     = silu(G) * U                               -> MX
//   mesh   Ymlp  = H @ Wd                (K-tiled: K={d['F']} exceeds the scale window)
//   host   h_out = h_mid + Ymlp                              <-- RESIDUAL 2
//
// THE SEAM IS THE POINT. The MLP half runs on the h_mid the DEVICE produced, not on the capture's
// -- so its operands carry attention's MX error and the two halves' errors COMPOUND. Every M_*
// golden here is re-derived from that value and none of llama_mlp_full.bin's bytes are reusable.
// Measured: h_mid drifts {100 * d['seam_drift']:.4f}% from the model's own, and the MLP then scores
// {100 * d['rel_m']:.4f}% where llama_mlp_full (given torch's h_mid) scores 11.26%.
//
// H_OUT_TORCH is the layer's TRUE output, from TinyLlama's own forward pass -- a reference neither
// sub-layer kernel could be graded against. The MX layer lands at rel_fro {100 * d['rel_out']:.4f}%
// of it.
//
// DATA LIVES IN llama_layer_full.bin, linked as a binary section -- {len(b.buf) / 1e6:.1f} MB.
// A_* entries are the attention half, M_* the MLP half (the two chains share operand names, so
// they are namespaced; an unprefixed collision would silently alias one onto the other).
//
{table}
#ifndef INCLUDE_LLAMA_LAYER_FULL_H
#define INCLUDE_LLAMA_LAYER_FULL_H

#include <stdint.h>

#define LLAMA_M   {d['M']}      // tokens (also the causal context)
#define LLAMA_D   {d['D']}      // hidden size, FULL
#define LLAMA_F   {d['F']}      // FFN neurons -- ALL of them
#define LLAMA_H   {d['H']}      // head dim
#define LLAMA_NH  {d['NH']}      // query heads -- ALL of them
#define LLAMA_NKV {d['NKV']}      // GQA kv heads
#define LLAMA_PER {d['PER']}      // query heads per kv head
#define LLAMA_QD  {d['QD']}      // NH * H, = D
#define LLAMA_KVD {d['KVD']}      // NKV * H
#define LLAMA_GD  {d['D'] // BLOCK}      // D / 32
#define LLAMA_GF  {d['F'] // BLOCK}      // F / 32
#define LLAMA_GH  {d['H'] // BLOCK}      // H / 32
#define LLAMA_GM  {d['M'] // BLOCK}      // M / 32
#define LLAMA_RMS_EPS {d['eps']:.10g}f

// The blob, linked by objcopy -I binary (see the Makefile).
extern const uint8_t _binary_llama_layer_full_bin_start[];
#define LLAMA_BLOB _binary_llama_layer_full_bin_start
#define LLAMA_AT(off, type) ((const type *) (LLAMA_BLOB + (off)))

{offs}

#endif // INCLUDE_LLAMA_LAYER_FULL_H
""")
    return bin_path, hdr_path


def main() -> int:
    cap = load_full_layer()
    print(f"llama_layer_full  from {cap['_path'].name}")
    t0 = time.time()
    b, d = build(cap)
    bp, hp = emit(b, d)
    print(f"  wrote {bp.relative_to(DATA.parent)}  ({bp.stat().st_size / 1e6:.1f} MB)")
    print(f"  wrote {hp.relative_to(DATA.parent)}  ({hp.stat().st_size / 1e3:.1f} kB)")
    print(f"  total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
