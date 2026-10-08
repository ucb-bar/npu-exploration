#!/usr/bin/env python3
"""Data for src/llama_layer_e2e.c: one whole TinyLlama decoder layer on MxGemmini, every stage on the device.

Exactly TinyLlama's layer for a 64-token prefill chunk (tokens 2048..2111 of wikitext-2) after a 2048-token KV cache
(tokens 0..2047): causal attention over all 2112 keys, the chunk's own K/V appended to the cache by the device. One
fp32 forward pass of TinyLlama-1.1B gives layer --layer's input h_pre, its weights, the rotary tables and the cache
(K after RoPE, V); the cache's last 64 key slots are NaN poison until the device writes them.

Device layouts:
  Wq / Wk output columns are permuted to [x1 of every head | x2 of every head] (x1 / x2 = the head's two rotary
  halves), so RoPE is six elementwise VPU passes over two contiguous halves. Within a head the order is unchanged, so
  head h's Q is [x1_h | x2_h] = the model's order, and the cache's K^T keeps the model's head-dim order.
  Weights: E4M3 codes [K][N] with E8M0 scales [K/32][N] (one per 32 along K). Cache (2112 keys): K^T [kv][64][2112]
  codes, [kv][2][2112] scales; V [kv][2112][64], [kv][66][64]. MASK [128][64] BF16: the causal mask of the last
  key block (the chunk), repeated for the two packed heads.
  ROPE_{C,S}Q [64][1024] / ROPE_{C,S}K [64][128] BF16: cos / sin of each token's position per permuted column.

References (fp64, unquantized fp32 weights, from the BF16 h_pre the device gets): every stage's output, for
per-stage accuracy; H_MODEL = the model's own hidden_states[layer + 1]. The reference code is checked against the model
first (from the fp32 h_pre it must reproduce hidden_states[layer + 1]).

    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ../../../.venv/bin/python3 gen_llama_layer_e2e.py
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import gen_llama_layer as L  # noqa: E402
import gen_llama_attn_full as GA  # noqa: E402  -- Blob

G, FMT = L.G, L.FMT
DATA = Path(__file__).resolve().parent.parent / "data"


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(axis=1, keepdims=True) + eps) * w


def rope(x, cos, sin):   # HF: x * cos + rotate_half(x) * sin, per head of 64
    h = x.shape[-1] // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    return x * cos + rot * sin


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layer_ref(h, W, ln1, ln2, eps, cos_q, sin_q, Kc, Vc, NH, NKV, hd):
    """fp64 decoder layer for the chunk h (rows) after the cache Kc/Vc [NKV][T][hd] (K after RoPE): the chunk's own
    K/V are appended and attention is causal. Returns every stage."""
    r = {}
    r["xn1"] = rms(h, ln1, eps)
    q = r["xn1"] @ W["q"]
    M = h.shape[0]
    qh = q.reshape(M, NH, hd).transpose(1, 0, 2)
    qh = rope(qh, cos_q[None], sin_q[None])
    r["q"] = qh                                            # [NH][M][hd]
    kn = rope((r["xn1"] @ W["k"]).reshape(M, NKV, hd).transpose(1, 0, 2), cos_q[None], sin_q[None])
    vn = (r["xn1"] @ W["v"]).reshape(M, NKV, hd).transpose(1, 0, 2)
    causal_q0 = Kc.shape[1]
    Kc, Vc = np.concatenate([Kc, kn], axis=1), np.concatenate([Vc, vn], axis=1)
    per = NH // NKV
    o = np.zeros((NH, M, hd))
    for hh in range(NH):
        s = qh[hh] @ Kc[hh // per].T / np.sqrt(hd)
        s = np.where(np.arange(Kc.shape[1])[None, :] > (causal_q0 + np.arange(M))[:, None], -np.inf, s)
        o[hh] = softmax(s) @ Vc[hh // per]
    r["o"] = o.transpose(1, 0, 2).reshape(M, NH * hd)
    r["yattn"] = r["o"] @ W["o"]
    r["hmid"] = h + r["yattn"]
    r["xn2"] = rms(r["hmid"], ln2, eps)
    g = r["xn2"] @ W["g"]
    u = r["xn2"] @ W["u"]
    r["hact"] = g / (1 + np.exp(-g)) * u
    r["ymlp"] = r["hact"] @ W["d"]
    r["hout"] = r["hmid"] + r["ymlp"]
    return r


def perm_cols(W, nh, hd):
    """[x1 of every head | x2 of every head]: new column j <- original column."""
    h2 = hd // 2
    idx = [hh * hd + i for hh in range(nh) for i in range(h2)] + [hh * hd + h2 + i for hh in range(nh) for i in range(h2)]
    return np.ascontiguousarray(W[:, idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--sq", type=int, default=64)
    ap.add_argument("--sk", type=int, default=2048)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    T = a.sk + a.sq
    tok = AutoTokenizer.from_pretrained(a.model_id)
    model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.float32)
    model.eval()
    cfg = model.config
    NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size
    hd = D // NH
    eps = float(cfg.rms_norm_eps)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[:, :T]
    t0 = time.time()
    out = model(ids, output_hidden_states=True)
    print(f"forward pass over {T} tokens: {time.time() - t0:.1f}s")
    hs_in = out.hidden_states[a.layer][0].numpy().astype(np.float64)
    hs_out = out.hidden_states[a.layer + 1][0].numpy().astype(np.float64)
    lay = model.model.layers[a.layer]
    W = {"q": lay.self_attn.q_proj.weight.T, "k": lay.self_attn.k_proj.weight.T, "v": lay.self_attn.v_proj.weight.T,
         "o": lay.self_attn.o_proj.weight.T, "g": lay.mlp.gate_proj.weight.T, "u": lay.mlp.up_proj.weight.T,
         "d": lay.mlp.down_proj.weight.T}
    W = {k: v.numpy().astype(np.float64) for k, v in W.items()}
    ln1 = lay.input_layernorm.weight.numpy().astype(np.float64)
    ln2 = lay.post_attention_layernorm.weight.numpy().astype(np.float64)
    pos = torch.arange(T)[None]
    cos, sin = model.model.rotary_emb(torch.zeros(1, 1, T, hd), pos)
    cos, sin = cos[0].numpy().astype(np.float64), sin[0].numpy().astype(np.float64)   # [T][hd]
    # the cache from the model's own layer input: K after RoPE, V
    hn = rms(hs_in, ln1, eps)
    Kall = rope((hn @ W["k"]).reshape(T, NKV, hd).transpose(1, 0, 2), cos[None], sin[None])
    Vall = (hn @ W["v"]).reshape(T, NKV, hd).transpose(1, 0, 2)

    # check the reference against the model: the fp32 h_pre, cache 0..sk-1, the chunk's own keys appended
    q0 = a.sk
    Kc, Vc = Kall[:, :a.sk], Vall[:, :a.sk]
    chk = layer_ref(hs_in[q0:], W, ln1, ln2, eps, cos[q0:], sin[q0:], Kc, Vc, NH, NKV, hd)
    err = np.linalg.norm(chk["hout"] - hs_out[q0:]) / np.linalg.norm(hs_out[q0:])
    print(f"reference check (causal, all {T} keys) vs model hidden_states[{a.layer + 1}]: rel_fro {err:.2e}")
    assert err < 1e-4, "fp64 reference does not reproduce the model"

    # the device's input is h_pre in BF16; the reference starts from that same input
    h_bits = G.bf16_bits(hs_in[q0:].astype(np.float32))
    h = (h_bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    R = layer_ref(h, W, ln1, ln2, eps, cos[q0:], sin[q0:], Kc, Vc, NH, NKV, hd)
    upd = hs_out[q0:] - hs_in[q0:]
    d_in = np.linalg.norm((R["hout"] - h) - upd) / np.linalg.norm(upd)
    print(f"BF16 h_pre alone: layer update differs {100 * d_in:.3f}% from the model's")

    b = GA.Blob()
    b.add("H_PRE", h_bits, np.uint16)
    b.add("W_IN_LN", G.bf16_bits(ln1.astype(np.float32)), np.uint16)
    b.add("W_POST_LN", G.bf16_bits(ln2.astype(np.float32)), np.uint16)
    wq = {"q": perm_cols(W["q"], NH, hd), "k": perm_cols(W["k"], NKV, hd), "v": W["v"], "o": W["o"],
          "g": W["g"], "u": W["u"], "d": W["d"]}
    for k in "qkvogud":
        t0 = time.time()
        c, s, _ = G.quantize(np.ascontiguousarray(wq[k].astype(np.float32)), axis="col", f=FMT)
        b.add(f"W{k.upper()}_CODES", c, np.uint8)
        b.add(f"W{k.upper()}_SCALES", s, np.uint8)
        print(f"  W{k} {wq[k].shape} quantized ({time.time() - t0:.1f}s)")
    h2 = hd // 2
    cq, sq_ = cos[q0:, :h2], sin[q0:, :h2]
    b.add("ROPE_CQ", G.bf16_bits(np.tile(cq, (1, NH)).astype(np.float32)), np.uint16)
    b.add("ROPE_SQ", G.bf16_bits(np.tile(sq_, (1, NH)).astype(np.float32)), np.uint16)
    b.add("ROPE_CK", G.bf16_bits(np.tile(cq, (1, NKV)).astype(np.float32)), np.uint16)
    b.add("ROPE_SK", G.bf16_bits(np.tile(sq_, (1, NKV)).astype(np.float32)), np.uint16)
    ktc, kts, vc, vs = [], [], [], []
    nan8 = lambda shape: np.full(shape, 0x7F, np.uint8)   # E4M3 NaN
    nan_s = lambda shape: np.full(shape, 0xFF, np.uint8)  # E8M0 NaN
    for kv in range(NKV):
        c, s, _ = G.quantize(np.ascontiguousarray(Kc[kv].T.astype(np.float32)), axis="col", f=FMT)
        ktc.append(np.hstack([c, nan8((hd, a.sq))])); kts.append(np.hstack([s, nan_s((hd // 32, a.sq))]))
        c, s, _ = G.quantize(np.ascontiguousarray(Vc[kv].astype(np.float32)), axis="col", f=FMT)
        vc.append(np.vstack([c, nan8((a.sq, hd))])); vs.append(np.vstack([s, nan_s((a.sq // 32, hd))]))
    b.add("KT_CACHE", np.stack(ktc), np.uint8)
    b.add("KT_SCALES", np.stack(kts), np.uint8)
    b.add("V_CACHE", np.stack(vc), np.uint8)
    b.add("V_SCALES", np.stack(vs), np.uint8)
    mask = np.where(np.arange(a.sq)[None, :] > np.arange(a.sq)[:, None], 0xFF80, 0).astype(np.uint16)
    b.add("MASK", np.vstack([mask, mask]), np.uint16)
    b.add("H_MODEL", hs_out[q0:].astype(np.float32), np.float32)
    for k in ("xn1", "yattn", "hmid", "xn2", "hact", "ymlp", "hout"):
        b.add(f"REF_{k.upper()}", R[k].astype(np.float32), np.float32)
    b.add("REF_O", R["o"].astype(np.float32), np.float32)

    bin_path, hdr_path = DATA / "llama_layer_e2e.bin", DATA / "llama_layer_e2e.h"
    bin_path.write_bytes(bytes(b.buf))
    offs = "\n".join(f"#define E2E_OFF_{n:<14s} {o}u" for n, o in b.off.items())
    table = "\n".join(f"//   {n:<14s} @ {o:>9d}  {sz:>9d} B  {sh}" for n, o, sz, sh in b.desc)
    F = W["g"].shape[1]
    hdr_path.write_text(f"""// GENERATED by gen/gen_llama_layer_e2e.py -- do not edit.
// One TinyLlama decoder layer (layer {a.layer}) for src/llama_layer_e2e.c: {a.sq} query tokens ({q0}..{T - 1}) after a
// {a.sk}-token KV cache (0..{a.sk - 1}), causal, the chunk's K/V appended by the device. Wq/Wk columns permuted [x1 all heads | x2 all heads].
// Data in llama_layer_e2e.bin ({len(b.buf) / 1e6:.1f} MB), linked as a binary section.
{table}
#ifndef INCLUDE_LLAMA_LAYER_E2E_H
#define INCLUDE_LLAMA_LAYER_E2E_H

#include <stdint.h>

#define E2E_M   {a.sq}
#define E2E_SK  {a.sk}      // cached keys; attention runs over E2E_SK + E2E_M
#define E2E_D   {D}
#define E2E_F   {F}
#define E2E_HD  {hd}
#define E2E_NH  {NH}
#define E2E_NKV {NKV}
#define E2E_EPS {eps:.10g}f

extern const uint8_t _binary_llama_layer_e2e_bin_start[];
#define E2E_AT(off, type) ((const type *) (_binary_llama_layer_e2e_bin_start + (off)))

{offs}

#endif
""")
    print(f"  wrote {bin_path} ({len(b.buf) / 1e6:.1f} MB), {hdr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
