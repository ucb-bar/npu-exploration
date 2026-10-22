#!/usr/bin/env python3
"""Data for src/llama_layer_mx.c: one decoder layer of a 7-13B model (llama_mx_common.MODELS) for a 64-token prefill
chunk after a KV cache, in one projection format (fp8 / fp4 / fp6), one blob per (model, format).

The weights and inputs are SYNTHETIC -- seeded random tensors at the model's exact shapes (hidden size, FFN size, heads,
GQA groups, head dim, RoPE base / llama3 scaling, Qwen2's Q/K/V biases, RMSNorm eps) -- because the checkpoints are not
in this tree. Everything the device computes is still checked bit for bit (golden_llama_layer_mx.py, rtl_exact), and
every stage is graded against an fp64 forward pass of the same layer on the same unquantized weights.

Same layouts as gen_llama_layer_e2e.py (TinyLlama), generalized to head dim 128:
  Wq / Wk output columns permuted to [x1 of every head | x2 of every head] (x1 / x2 = the head's two rotary halves),
  Wq and bq carry 1/sqrt(head_dim) (attention then needs no scaling pass). Weights: codes [K][N] (E4M3), or packed
  nibbles [K][N/2] (FP4 E2M1 / FP6 LUT indices, low nibble = even column), with E8M0 scales [K/32][N]; FP6 adds one
  16-entry E3M2 table per 2^G columns ([N >> G][3] uint32). Cache (E4M3 whatever the projection format):
  K^T [kv][hd][sk+m], scales [kv][hd/32][sk+m]; V [kv][sk+m][hd], scales [kv][(sk+m)/32][hd]; the last m keys are NaN
  poison until the device appends the chunk's own K/V. MASK [m][m] BF16 (0 / -inf), the causal mask of the last block.
  FP6 activation tables (L_XN1, L_O, L_XN2, L_H): one table per tensor, k-means over the fp64 reference activation's
  MX-E3M2 codes (a static calibration), LUT_COPIES copies for the loop-managed act-LUT load.

    ../../../.venv/bin/python3 gen_llama_layer_mx.py --model llama2_7b --fmt fp4     [--model all --fmt all]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import llama_mx_common as C
import golden_llama_layer_e2e as GL
from gen_mlp_proj_fp6 import mx_e3m2, e3m2_decode, pack_luts

pk_cols = lambda c: (c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8)    # [K][N] -> [K][N/2]


class Blob:
    """Append-only binary image, every tensor 64-byte aligned (as gen_llama_attn_full.Blob)."""

    def __init__(self) -> None:
        self.buf, self.off, self.desc = bytearray(), {}, []

    def add(self, name: str, arr, dtype) -> None:
        a = np.ascontiguousarray(arr, dtype=dtype)
        self.buf.extend(b"\0" * ((-len(self.buf)) % 64))
        self.off[name] = len(self.buf)
        self.desc.append((name, len(self.buf), a.nbytes, str(a.shape)))
        self.buf.extend(a.tobytes())


# ---------------------------------------------------------------- fp64 reference
def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(axis=1, keepdims=True) + eps) * w


def rope(x, cos, sin):   # HF: x * cos + rotate_half(x) * sin over the last axis (one head)
    h = x.shape[-1] // 2
    return x * cos + np.concatenate([-x[..., h:], x[..., :h]], axis=-1) * sin


def rope_tables(m: C.Model, T: int):
    """cos / sin [T][hd] as HF's rotary embedding (default, or rope_type llama3)."""
    hd = m.HD
    inv = 1.0 / (m.theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    if m.llama3_rope:   # transformers _compute_llama3_parameters
        factor, low, high, old = 8.0, 1.0, 4.0, 8192
        wl = 2 * np.pi / inv
        inv_l = np.where(wl > old / low, inv / factor, inv)
        smooth = (old / wl - low) / (high - low)
        smoothed = (1 - smooth) * inv_l / factor + smooth * inv_l
        medium = ~(wl < old / high) & ~(wl > old / low)
        inv = np.where(medium, smoothed, inv_l)
    f = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([f, f], axis=1)
    return np.cos(emb), np.sin(emb)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layer_ref(h, W, B, ln1, ln2, eps, cos, sin, Kc, Vc, m: C.Model):
    """fp64 decoder layer for the chunk h [M][D] after the cache Kc / Vc [NKV][SK][hd] (K after RoPE); the chunk's own
    K / V are appended, attention causal. Unscaled weights (the standard formula). Returns every stage."""
    M, NH, NKV, hd = h.shape[0], m.NH, m.NKV, m.HD
    r = {"xn1": rms(h, ln1, eps)}
    heads = lambda x, n: x.reshape(M, n, hd).transpose(1, 0, 2)
    q = rope(heads(r["xn1"] @ W["q"] + B["q"], NH), cos[None], sin[None])
    kn = rope(heads(r["xn1"] @ W["k"] + B["k"], NKV), cos[None], sin[None])
    vn = heads(r["xn1"] @ W["v"] + B["v"], NKV)
    q0 = Kc.shape[1]
    Kc, Vc = np.concatenate([Kc, kn], axis=1), np.concatenate([Vc, vn], axis=1)
    o = np.zeros((NH, M, hd))
    for hh in range(NH):
        g = hh // (NH // NKV)
        s = q[hh] @ Kc[g].T / np.sqrt(hd)
        s = np.where(np.arange(Kc.shape[1])[None, :] > (q0 + np.arange(M))[:, None], -np.inf, s)
        o[hh] = softmax(s) @ Vc[g]
    r["o"] = o.transpose(1, 0, 2).reshape(M, NH * hd)
    r["yattn"] = r["o"] @ W["o"]
    r["hmid"] = h + r["yattn"]
    r["xn2"] = rms(r["hmid"], ln2, eps)
    g, u = r["xn2"] @ W["g"], r["xn2"] @ W["u"]
    r["hact"] = g / (1 + np.exp(-g)) * u
    r["ymlp"] = r["hact"] @ W["d"]
    r["hout"] = r["hmid"] + r["ymlp"]
    return r


def perm_cols(W, nh, hd):
    """[x1 of every head | x2 of every head]: new column j <- original column (also for a bias vector)."""
    h2 = hd // 2
    idx = [hh * hd + i for hh in range(nh) for i in range(h2)] + [hh * hd + h2 + i for hh in range(nh) for i in range(h2)]
    return np.ascontiguousarray(W[..., idx])


# ---------------------------------------------------------------- quantizers (operands the device consumes as data)
FLT_EPS = np.float32(1.1920929e-07)


def mx_e4m3(X, axis):
    """X fp32, blocks of 32 along `axis` (0: down the rows = K of a B operand; 1: along a row) -> (codes like X,
    E8M0 [R/32][C] or [R][C/32]): scale 2^floor(log2 max(amax, FLT_EPS)) (the shipped convention, pmax 0), codes the
    mx_fp_math E4M3 encoder (RNE, saturating)."""
    X = np.asarray(X, np.float32)
    Xb = X.reshape(-1, 32, X.shape[1]) if axis == 0 else X.reshape(X.shape[0], -1, 32)
    amax = np.abs(Xb).max(axis=1 if axis == 0 else 2)
    _, ex = np.frexp(np.maximum(amax, FLT_EPS))
    se = np.clip(ex - 1, -127, 127).astype(np.int64)
    sc = np.ldexp(np.float32(1.0), se).astype(np.float32)
    v = Xb / (sc[:, None, :] if axis == 0 else sc[:, :, None])
    return GL.e4m3_encode(v.astype(np.float32)).reshape(X.shape), (se + 127).astype(np.uint8)


def quant_weight(Wf, fmt):
    """fp32 W [K][N] -> blob entries {suffix: (array, dtype)} in the device's B-operand form."""
    if fmt == "fp8":
        c, s = mx_e4m3(Wf, 0)
        return {"C": (c, np.uint8), "S": (s, np.uint8)}
    if fmt == "fp4":
        c, s = GL.mxfp4_weights(Wf, 0)
        return {"C": (pk_cols(c), np.uint8), "S": (s, np.uint8)}
    import torch
    from mxq import lut as mxq_lut
    codes, s = mx_e3m2(np.ascontiguousarray(Wf, np.float32), 0)
    P = torch.from_numpy(e3m2_decode(codes))
    T = mxq_lut.tables(P, "MXFP6_E3M2", group=C.LUT_G, max_iters=50)
    idx = mxq_lut.pick(P, T, group=C.LUT_G).numpy().astype(np.uint8)
    tcodes = mxq_lut.encode(T, "MXFP6_E3M2").numpy()
    return {"C": (pk_cols(idx), np.uint8), "S": (s, np.uint8), "L": (pack_luts(tcodes), np.uint32)}


def act_table(X):
    """One E3M2 table for an activation tensor (fp64 reference [M][N]): k-means over its MX-E3M2 codes; LUT_COPIES
    copies, packed."""
    import torch
    from mxq import lut as mxq_lut
    codes, _ = mx_e3m2(np.ascontiguousarray(X, np.float32), 1)
    P = torch.from_numpy(e3m2_decode(codes).reshape(-1, 1))
    T = mxq_lut.tables(P, "MXFP6_E3M2", group=0, max_iters=50)
    tc = mxq_lut.encode(T, "MXFP6_E3M2").numpy()
    return pack_luts(np.repeat(tc, C.LUT_COPIES, axis=0))


# ---------------------------------------------------------------- one blob
def generate(model: str, fmt: str) -> None:
    import torch  # noqa: F401  -- mxq
    import models  # noqa: F401  -- puts mxq on sys.path
    m = C.MODELS[model]
    t0 = time.time()
    rng = np.random.default_rng(sum(map(ord, model)) * 7919)   # deterministic per model
    M, SK, D, F, NH, NKV, hd = m.m, m.sk, m.D, m.F, m.NH, m.NKV, m.HD
    LKV, T = NKV * hd, m.sk + m.m
    # synthetic layer: N(0, 1/fan_in) weights, RMSNorm gains around 1, hidden states with a few loud channels
    W = {"q": rng.standard_normal((D, D)) / np.sqrt(D), "k": rng.standard_normal((D, LKV)) / np.sqrt(D),
         "v": rng.standard_normal((D, LKV)) / np.sqrt(D), "o": rng.standard_normal((D, D)) / np.sqrt(D),
         "g": rng.standard_normal((D, F)) / np.sqrt(D), "u": rng.standard_normal((D, F)) / np.sqrt(D),
         "d": rng.standard_normal((F, D)) / np.sqrt(F)}
    W = {k: v.astype(np.float32).astype(np.float64) for k, v in W.items()}
    B = {"q": np.zeros(D), "k": np.zeros(LKV), "v": np.zeros(LKV)}
    if m.bias:
        B = {"q": rng.standard_normal(D) * 0.5, "k": rng.standard_normal(LKV) * 0.5, "v": rng.standard_normal(LKV) * 0.05}
        B = {k: GL.bf2f(GL.d2bf(v)).astype(np.float64) for k, v in B.items()}   # what the device adds is BF16
    ln1 = np.abs(1 + 0.2 * rng.standard_normal(D)); ln2 = np.abs(1 + 0.2 * rng.standard_normal(D))
    ln1, ln2 = (GL.bf2f(GL.d2bf(x)).astype(np.float64) for x in (ln1, ln2))
    hs = rng.standard_normal((T, D))
    loud = rng.choice(D, 4, replace=False)
    hs[:, loud] *= 20.0
    cos, sin = rope_tables(m, T)
    # the cache: the fp64 layer's own K (after RoPE) / V of the first SK tokens
    hn = rms(hs[:SK], ln1, m.eps)
    Kc = rope((hn @ W["k"] + B["k"]).reshape(SK, NKV, hd).transpose(1, 0, 2), cos[None, :SK], sin[None, :SK])
    Vc = (hn @ W["v"] + B["v"]).reshape(SK, NKV, hd).transpose(1, 0, 2)
    # the device's input is h_pre in BF16; the reference starts from that same input
    h_bits = GL.d2bf(hs[SK:])
    h = GL.bf2f(h_bits).astype(np.float64)
    R = layer_ref(h, W, B, ln1, ln2, m.eps, cos[SK:], sin[SK:], Kc, Vc, m)
    print(f"[{model} {fmt}] reference layer ({time.time() - t0:.0f}s)", flush=True)

    b = Blob()
    b.add("H_PRE", h_bits, np.uint16)
    b.add("W_IN_LN", GL.d2bf(ln1), np.uint16)
    b.add("W_POST_LN", GL.d2bf(ln2), np.uint16)
    sq = 1.0 / np.sqrt(hd)   # carried by Wq / bq
    dev = {"q": perm_cols(W["q"] * sq, NH, hd), "k": perm_cols(W["k"], NKV, hd), "v": W["v"], "o": W["o"],
           "g": W["g"], "u": W["u"], "d": W["d"]}
    for k in "qkvogud":
        for suf, (arr, dt) in quant_weight(dev[k].astype(np.float32), fmt).items():
            b.add(f"W{k.upper()}_{suf}", arr, dt)
        print(f"  W{k} {dev[k].shape} quantized ({time.time() - t0:.0f}s)", flush=True)
    if m.bias:
        b.add("BQ", GL.d2bf(perm_cols(B["q"] * sq, NH, hd)), np.uint16)
        b.add("BK", GL.d2bf(perm_cols(B["k"], NKV, hd)), np.uint16)
        b.add("BV", GL.d2bf(B["v"]), np.uint16)
    h2 = hd // 2
    cq, sq_ = cos[SK:, :h2], sin[SK:, :h2]
    b.add("ROPE_CQ", GL.d2bf(np.tile(cq, (1, NH))), np.uint16)
    b.add("ROPE_SQ", GL.d2bf(np.tile(sq_, (1, NH))), np.uint16)
    b.add("ROPE_CK", GL.d2bf(np.tile(cq, (1, NKV))), np.uint16)
    b.add("ROPE_SK", GL.d2bf(np.tile(sq_, (1, NKV))), np.uint16)
    ktc, kts, vc, vs = [], [], [], []
    for g in range(NKV):
        c, s = mx_e4m3(Kc[g].T, 0)
        ktc.append(np.hstack([c, np.full((hd, M), 0x7F, np.uint8)]))
        kts.append(np.hstack([s, np.full((hd // 32, M), 0xFF, np.uint8)]))
        c, s = mx_e4m3(Vc[g], 0)
        vc.append(np.vstack([c, np.full((M, hd), 0x7F, np.uint8)]))
        vs.append(np.vstack([s, np.full((M // 32, hd), 0xFF, np.uint8)]))
    b.add("KT_CACHE", np.stack(ktc), np.uint8)
    b.add("KT_SCALES", np.stack(kts), np.uint8)
    b.add("V_CACHE", np.stack(vc), np.uint8)
    b.add("V_SCALES", np.stack(vs), np.uint8)
    b.add("MASK", np.where(np.arange(M)[None, :] > np.arange(M)[:, None], 0xFF80, 0).astype(np.uint16), np.uint16)
    if fmt == "fp6":
        for n, x in (("XN1", R["xn1"]), ("O", R["o"]), ("XN2", R["xn2"]), ("H", R["hact"])):
            b.add(f"L_{n}", act_table(x), np.uint32)
    for k in ("xn1", "o", "yattn", "hmid", "ymlp", "hout"):
        b.add(f"REF_{k.upper()}", R[k].astype(np.float32), np.float32)

    name = C.data_name(model, fmt)
    (C.DATA / f"{name}.bin").write_bytes(bytes(b.buf))
    offs = "\n".join(f"#define LMX_OFF_{n:<10s} {o}u" for n, o in b.off.items())
    table = "\n".join(f"//   {n:<10s} @ {o:>10d}  {sz:>10d} B  {sh}" for n, o, sz, sh in b.desc)
    (C.DATA / f"{name}.h").write_text(f"""// GENERATED by gen/gen_llama_layer_mx.py -- do not edit.
// One {m.name} decoder layer (synthetic weights at the model's shapes) for src/llama_layer_mx.c, projections in {fmt}:
// {M} query tokens after a {SK}-token KV cache, causal; Wq/Wk columns permuted [x1 all heads | x2 all heads], Wq/bq
// carry 1/sqrt({hd}). Data in {name}.bin ({len(b.buf) / 1e6:.1f} MB), linked as a binary section.
{table}
#ifndef INCLUDE_LLAMA_MX_DATA_H
#define INCLUDE_LLAMA_MX_DATA_H

#include <stdint.h>

#define LMX_MODEL_NAME "{m.name}"
#define LMX_FMT_NAME   "{fmt}"
#define LMX_FMT   {C.FORMATS[fmt]}      // projections: 0 E4M3, 1 FP6 E3M2 LUT, 2 FP4 E2M1 (attention always E4M3)
#define LMX_G     {C.LUT_G}
#define LMX_M     {M}
#define LMX_SK    {SK}      // cached keys; attention runs over LMX_SK + LMX_M
#define LMX_D     {D}
#define LMX_F     {F}
#define LMX_HD    {hd}
#define LMX_NH    {NH}
#define LMX_NKV   {NKV}
#define LMX_BIAS  {int(m.bias)}
#define LMX_EPS   {m.eps:.10g}f

extern const uint8_t _binary_{name}_bin_start[];
#define LMX_AT(off, type) ((const type *) (_binary_{name}_bin_start + (off)))

{offs}

#endif
""")
    print(f"[{model} {fmt}] wrote {name}.bin ({len(b.buf) / 1e6:.1f} MB) ({time.time() - t0:.0f}s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["tiny"])
    ap.add_argument("--fmt", nargs="+", default=["fp8"])
    a = ap.parse_args()
    models = [k for k in C.MODELS if k != "tiny"] if a.model == ["all"] else a.model
    fmts = list(C.FORMATS) if a.fmt == ["all"] else a.fmt
    for mo in models:
        for f in fmts:
            generate(mo, f)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(C.NPU))
    raise SystemExit(main())
