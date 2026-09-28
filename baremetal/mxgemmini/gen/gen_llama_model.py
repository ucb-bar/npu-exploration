#!/usr/bin/env python3
"""Generate the data for `src/llama_model.c` -- N STACKED TinyLlama layers, optionally the whole
model with its lm_head, run as one chain in MXFP8.

Each layer consumes the residual stream the PREVIOUS layer's DEVICE output produced, so the error
compounds exactly as it does in the hardware. `llama_layer_full` measured one such seam (the MLP
went 11.26% -> 16.60% on a 0.32%-perturbed input); this measures what N of them do, which is the
one quantity no per-layer number predicts.

    python3 gen_llama_model.py                 # every captured layer + the head
    python3 gen_llama_model.py --layers 2      # the first two layers, no head (fast, for bring-up)

LAYOUT: A UNIFORM PER-LAYER STRIDE. Each layer's operands and goldens are built by the SAME chain
functions the single-layer kernel uses (`attention_chain`, `mlp_chain`), into a per-layer `Blob`.
Because every layer has identical shapes, every per-layer blob has an identical internal layout --
which is asserted, not assumed -- so the emitted header carries ONE set of relative offsets plus a
stride, and the kernel addresses layer n as `LAYERS + n * STRIDE + LOFF_<name>`. The alternative,
22 x ~40 absolute offsets, would put the layer count in the header's shape and make a 2-layer
build and a 22-layer build structurally different C.

THE FULL PER-STAGE BIT-EXACT GATE IS KEPT, for every layer. The plan had assumed goldens would have
to be dropped at this scale; they do not. They are ACTIVATION-sized ([M] x something at M=32), not
weight-sized: ~1.2 MB per layer against ~45 MB of weights, so all 22 layers' goldens cost ~26 MB on
a ~1.05 GB blob -- 2.4%. Keeping them means a divergence is localized to a stage of a layer instead
of only being visible in the final logits.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPU = HERE.parents[2]
DATA = HERE.parent / "data"
ROCC = NPU.parent / "software" / "gemmini-rocc-tests"
sys.path.insert(0, str(NPU))
sys.path.insert(0, str(ROCC))
sys.path.insert(0, str(HERE))
DATA.mkdir(parents=True, exist_ok=True)

import gen_matmul_llama as G            # noqa: E402
import gen_llama_layer as GL            # noqa: E402  -- FMT, bf16_exact()
import gen_llama_attn_full as GA        # noqa: E402  -- Blob, attention_chain()
import gen_llama_mlp_full as GM         # noqa: E402  -- mlp_chain()
from gen_llama_layer_full import _bf16_rne, _bf16_val, residual  # noqa: E402
from mesh_par import mesh_parallel      # noqa: E402
from kernels.captures.llama_layer import rmsnorm  # noqa: E402

FMT = GL.FMT
BLOCK = 32
CAPTURE = NPU / "out" / "model_capture"


def load_layer(n: int) -> dict:
    p = CAPTURE / f"layer{n}.npz"
    if not p.exists():
        raise SystemExit(f"{p} missing -- run:\n"
                         f"    cd {NPU} && .venv/bin/python3 -m kernels.captures.llama_model")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def load_model() -> dict:
    p = CAPTURE / "model.npz"
    if not p.exists():
        raise SystemExit(f"{p} missing -- run kernels.captures.llama_model")
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def build_layer_blob(cap: dict, x_bits: np.ndarray) -> tuple[GA.Blob, np.ndarray, dict]:
    """One decoder layer: both halves and both residuals, into its own Blob.

    `x_bits` is the residual stream entering this layer, as BF16 bits -- for layer 0 the embedding
    output, thereafter the PREVIOUS layer's device result. Returns (blob, h_out bits, stats).
    """
    x = _bf16_val(x_bits)
    b = GA.Blob()
    b.add("W_IN_LN", GL.bf16_exact(cap["w_in_ln"], "w_in_ln"), np.uint16)
    b.add("W_POST_LN", GL.bf16_exact(cap["w_post_ln"], "w_post_ln"), np.uint16)

    Yattn, da = GA.attention_chain(cap, b, x, cap["w_in_ln"], prefix="A_")
    h_mid_bits = residual(x_bits, Yattn)
    b.add("H_MID_OUT", h_mid_bits, np.uint16)

    Ymlp, dm = GM.mlp_chain(cap, b, _bf16_val(h_mid_bits), cap["w_post_ln"], prefix="M_")
    h_out_bits = residual(h_mid_bits, Ymlp)
    b.add("H_OUT_OUT", h_out_bits, np.uint16)
    # The model's own value for this layer, so a divergence can be located rather than just seen.
    b.add("H_OUT_REF", G.bf16_bits(cap["h_out"]), np.uint16)

    h_out = _bf16_val(h_out_bits)
    drift_in = float(np.linalg.norm(x - cap["h_pre"]) / np.linalg.norm(cap["h_pre"]))
    drift_out = float(np.linalg.norm(h_out - cap["h_out"]) / np.linalg.norm(cap["h_out"]))
    return b, h_out_bits, dict(drift_in=drift_in, drift_out=drift_out, dims=da | dm)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=None,
                    help="how many layers to stack, from layer 0. Default: every captured one.")
    ap.add_argument("--no-head", action="store_true",
                    help="stop after the last layer: no final RMSNorm, no lm_head, no logits. "
                         "The kernel then grades the residual stream instead of the logits.")
    ap.add_argument("--tag", default="", help="suffix for the emitted data files")
    args = ap.parse_args()

    have = sorted(int(p.stem[5:]) for p in CAPTURE.glob("layer*.npz"))
    if not have:
        raise SystemExit(f"no layer captures in {CAPTURE} -- run kernels.captures.llama_model")
    NL = args.layers if args.layers else len(have)
    if have[:NL] != list(range(NL)):
        raise SystemExit(f"need layers 0..{NL - 1} captured consecutively; have {have}")
    head = not args.no_head
    if head and not (CAPTURE / "model.npz").exists():
        raise SystemExit("model.npz missing (needed for the head); use --no-head or re-capture")

    mdl = load_model() if head or (CAPTURE / "model.npz").exists() else None
    print(f"llama_model: stacking {NL} layer(s){' + head' if head else ''}")

    # THE HEAD ON A TRUNCATED STACK IS NOT THE MODEL. `lm_head` reads the hidden state after ALL
    # n_layers; applied to layer NL-1's output for NL < n_layers it produces well-formed logits of
    # nothing, and their perplexity is meaningless (measured: 35026 vs the model's 28.5 at NL=2).
    # Kept available anyway, because it is the only way to exercise the head's code path cheaply
    # during bring-up -- the per-stage bit-exact gate is still valid, only the GRADE lines are not.
    n_total = int(mdl["meta_n_layers"]) if mdl is not None and "meta_n_layers" in mdl else NL
    truncated_head = head and NL < n_total
    if truncated_head:
        print(f"  WARNING: head over {NL} of {n_total} layers -- the logits, nll and ppl below are "
              f"NOT the model's and must not be read as accuracy. Bit-exactness still gates.")

    t_all = time.time()
    x_bits = G.bf16_bits(mdl["embed_out"]) if mdl is not None else None
    if x_bits is None:
        x_bits = G.bf16_bits(load_layer(0)["h_pre"])

    layer_bufs: list[bytes] = []
    layer_off: dict[str, int] | None = None
    layer_desc = None
    stats = []
    for n in range(NL):
        t0 = time.time()
        cap = load_layer(n)
        b, x_bits, st = build_layer_blob(cap, x_bits)
        # Every layer must lay out identically, or one stride cannot address them all.
        if layer_off is None:
            layer_off, layer_desc = b.off, b.desc
        elif b.off != layer_off:
            raise SystemExit(f"layer {n} has a different blob layout than layer 0 -- the uniform "
                             f"stride assumption is broken; shapes must vary somewhere")
        layer_bufs.append(bytes(b.buf))
        stats.append(st)
        print(f"  layer {n:2d}  in-drift {100 * st['drift_in']:7.4f}%  ->  "
              f"out-drift {100 * st['drift_out']:7.4f}%   ({time.time() - t0:.0f}s)", flush=True)

    stride = max(len(x) for x in layer_bufs)
    stride = (stride + 63) // 64 * 64
    print(f"  per-layer stride {stride} B ({stride / 1e6:.1f} MB); "
          f"{NL} layers = {NL * stride / 1e6:.0f} MB")

    # ---- assemble: globals first, then the uniform layer array, then the head ----
    b = GA.Blob()
    b.add("EMBED_OUT", _bf16_rne(_bf16_val(G.bf16_bits(
        mdl["embed_out"] if mdl is not None else load_layer(0)["h_pre"]))), np.uint16)
    layers_off = len(b.buf) + (-len(b.buf)) % 64
    b.buf.extend(b"\0" * (layers_off - len(b.buf)))
    b.off["LAYERS"] = layers_off
    b.desc.append(("LAYERS", layers_off, NL * stride, f"({NL} x {stride} B)"))
    for buf in layer_bufs:
        b.buf.extend(buf)
        b.buf.extend(b"\0" * (stride - len(buf)))

    d = dict(NL=NL, M=0, D=0, F=0, stride=stride, head=int(head))
    dims = stats[0]["dims"]
    d.update(M=dims["M"], D=dims["D"], F=dims["F"], H=dims["H"], NH=dims["NH"],
             NKV=dims["NKV"], PER=dims["PER"], QD=dims["QD"], KVD=dims["KVD"],
             eps=dims["eps"])

    # ---- the head: final RMSNorm, then lm_head ----
    if head:
        M, D = d["M"], d["D"]
        V = int(mdl["lm_head"].shape[1])
        h_last = _bf16_val(x_bits)
        b.add("W_FINAL_LN", GL.bf16_exact(mdl["w_final_ln"], "w_final_ln"), np.uint16)
        xf = rmsnorm(h_last, mdl["w_final_ln"], d["eps"])
        xf_codes, xf_scales, xf_P = G.quantize(xf, axis="row", f=FMT)
        b.add("XF_CODES", xf_codes, np.uint8)
        b.add("XF_SCALES", xf_scales.T, np.uint8)
        lmh_codes, lmh_scales, lmh_P = G.quantize(mdl["lm_head"], axis="col", f=FMT)
        b.add("LMH_CODES", lmh_codes, np.uint8)
        b.add("LMH_SCALES", lmh_scales, np.uint8)
        t0 = time.time()
        logits = mesh_parallel(xf_P, xf_scales, lmh_P, lmh_scales, FMT, label="logits = Xf @ Wlm")
        b.add("LOGITS_OUT", G.bf16_bits(logits), np.uint16)
        b.add("LOGITS_TORCH", mdl["logits"], np.float32)
        b.add("LABELS", mdl["labels"], np.int32)
        d.update(V=V)

        lab = mdl["labels"].astype(np.int64)
        def _nll(lg):
            z = lg.astype(np.float32)
            z = z - z.max(axis=-1, keepdims=True)
            lse = np.log(np.exp(z).sum(axis=-1))
            return float(np.mean(lse - z[np.arange(len(lab)), lab]))
        nll_mx, nll_t = _nll(logits), _nll(mdl["logits"])
        agree = int((logits.argmax(-1) == mdl["logits"].argmax(-1)).sum())
        rel_l = float(np.linalg.norm(logits - mdl["logits"]) / np.linalg.norm(mdl["logits"]))
        d.update(nll_mx=nll_mx, nll_t=nll_t, ppl_mx=float(np.exp(nll_mx)),
                 ppl_t=float(np.exp(nll_t)), agree=agree, rel_logits=rel_l)
        print(f"  head   logits {logits.shape}   rel_fro vs torch = {100 * rel_l:.4f}%   "
              f"({time.time() - t0:.0f}s)")
        print(f"  GRADE  argmax agrees on {agree}/{M} tokens")
        print(f"  GRADE  nll  MX {nll_mx:.6f}  vs torch {nll_t:.6f}")
        print(f"  GRADE  ppl  MX {np.exp(nll_mx):.4f}  vs torch {np.exp(nll_t):.4f}")
    else:
        b.add("H_FINAL_REF", G.bf16_bits(load_layer(NL - 1)["h_out"]), np.uint16)

    d.update(rel_last=stats[-1]["drift_out"], truncated=int(truncated_head), n_total=n_total)
    emit(b, d, layer_off, layer_desc, args.tag)
    print(f"  total {time.time() - t_all:.0f}s")
    return 0


def emit(b: GA.Blob, d: dict, loff: dict, ldesc, tag: str) -> None:
    name = f"llama_model{tag}"
    bin_path, hdr_path = DATA / f"{name}.bin", DATA / f"{name}.h"
    bin_path.write_bytes(bytes(b.buf))

    offs = "\n".join(f"#define LLAMA_OFF_{n:<14s} {o}u" for n, o in b.off.items())
    loffs = "\n".join(f"#define LOFF_{n:<14s} {o}u" for n, o in loff.items())
    table = "\n".join(f"//   {n:<14s} @ {o:>11d}  {sz:>11d} B  {sh}" for n, o, sz, sh in b.desc)
    ltable = "\n".join(f"//   {n:<14s} + {o:>9d}  {sz:>11d} B  {sh}" for n, o, sz, sh in ldesc)
    grade = ""
    if d.get("head") and d.get("truncated"):
        grade = (f"// !! THE HEAD HERE IS OVER {d['NL']} OF {d['n_total']} LAYERS. lm_head reads the hidden state after\n"
                 f"// ALL layers, so these logits are well-formed logits of nothing: ppl {d['ppl_mx']:.1f} against the\n"
                 f"// model's {d['ppl_t']:.4f}. This build exists to exercise the head's CODE PATH during bring-up;\n"
                 f"// its bit-exact gate is valid, its accuracy numbers are not.\n//\n")
    elif d.get("head"):
        grade = (f"// Logits land at rel_fro {100 * d['rel_logits']:.4f}% of torch's, argmax agrees on\n"
                 f"// {d['agree']}/{d['M']} tokens, and wikitext2 perplexity is {d['ppl_mx']:.4f} against the\n"
                 f"// fp32 model's {d['ppl_t']:.4f}.\n//\n")
    with open(hdr_path, "w") as fh:
        fh.write(f"""// GENERATED by gen_llama_model.py -- do not edit by hand.
//
// {d['NL']} STACKED TinyLlama decoder layers{' + the final RMSNorm and lm_head' if d.get('head') else ''},
// as ONE MXFP8 chain. Each layer consumes the residual stream the PREVIOUS layer's DEVICE output
// produced, so the quantization error compounds exactly as it does in hardware.
// {d['M']} real tokens of wikitext2. Quantized by MXQuant (block 32, {FMT.name}); mesh goldens from
// fp8_matmul_model.tiled_matmul_hwlike, run column-parallel (mesh_par.py).
//
{grade}// THE LAYER ARRAY HAS A UNIFORM STRIDE. Every layer's operands and per-stage goldens sit in an
// identically-laid-out block of {d['stride']} B, so layer n is addressed as
//     LLAMA_BLOB + LLAMA_OFF_LAYERS + n * LLAMA_LAYER_STRIDE + LOFF_<name>
// and the C is the same source for 2 layers or {d['NL']}. The identical layout is ASSERTED at
// generation time, not assumed.
//
// Globals:
{table}
//
// Per-layer block (offsets RELATIVE to the layer's base):
{ltable}
#ifndef INCLUDE_LLAMA_MODEL_H
#define INCLUDE_LLAMA_MODEL_H

#include <stdint.h>

#define LLAMA_NL  {d['NL']}      // decoder layers stacked
#define LLAMA_M   {d['M']}      // tokens
#define LLAMA_D   {d['D']}      // hidden size
#define LLAMA_F   {d['F']}      // FFN neurons
#define LLAMA_H   {d['H']}      // head dim
#define LLAMA_NH  {d['NH']}      // query heads
#define LLAMA_NKV {d['NKV']}      // GQA kv heads
#define LLAMA_PER {d['PER']}
#define LLAMA_QD  {d['QD']}
#define LLAMA_KVD {d['KVD']}
#define LLAMA_GD  {d['D'] // BLOCK}
#define LLAMA_GF  {d['F'] // BLOCK}
#define LLAMA_GH  {d['H'] // BLOCK}
#define LLAMA_GM  {d['M'] // BLOCK}
#define LLAMA_RMS_EPS {d['eps']:.10g}f
#define LLAMA_LAYER_STRIDE {d['stride']}u
#define LLAMA_HAS_HEAD {d.get('head', 0)}
{f"#define LLAMA_V   {d['V']}      // vocab" if d.get('head') else ""}
{f"#define LLAMA_GV  {d['V'] // BLOCK}" if d.get('head') else ""}

extern const uint8_t _binary_{name}_bin_start[];
#define LLAMA_BLOB _binary_{name}_bin_start
#define LLAMA_AT(off, type) ((const type *) (LLAMA_BLOB + (off)))
// Layer n's base, and one of its tensors.
#define LLAMA_LAYER(n) (LLAMA_OFF_LAYERS + (unsigned long) (n) * LLAMA_LAYER_STRIDE)
#define LLAMA_LAT(n, loff, type) ((const type *) (LLAMA_BLOB + LLAMA_LAYER(n) + (loff)))

{offs}

{loffs}

#endif // INCLUDE_LLAMA_MODEL_H
""")
    print(f"  wrote {bin_path.relative_to(DATA.parent)}  ({bin_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  wrote {hdr_path.relative_to(DATA.parent)}")


if __name__ == "__main__":
    raise SystemExit(main())
