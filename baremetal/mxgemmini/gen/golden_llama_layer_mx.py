#!/usr/bin/env python3
"""Bit-exact Python golden of src/llama_layer_mx.c, independent of Spike: one (model, format, mesh) build.

* matmuls: npu-exploration's rtl_exact datapath (rtl_exact/rtl_datapath.py primitives under the mesh's config:
  mxgemmini_rtl.json at DIM 16, _dim8 / _dim4 / _dim32: window = DIM, that mesh's accumulator lane schedule; truncated
  PE product, both addends quantized to the lane, bf16 cross-tile accumulation), split over output columns;
* VPU ops: golden_llama_layer_e2e's port of include/vpu_ref.h, the RSUM lane tree over DIM/2 lanes;
* SPAD_REQUANT: E8M0 = floor(log2 amax) per 32 values; E4M3 (a -0.0 element keeps its sign), FP4 (BF16 -> E2M1 encoder),
  FP6 LUT (BF16 -> E3M2 encoder, then the hardware nearest-finder against the tensor's table, ties to the lower index);
* mesh outputs to BF16 as the RTL / Spike store them (-0 -> +0).
Follows the kernel's order of operations stage by stage and writes data/<name>_d<DIM>_expect.h: the plan constants that
change numbers (RMSNorm column chunk, attention key blocks) and the 22 stage hashes the kernel checks itself against.

    ../../../.venv/bin/python3 golden_llama_layer_mx.py --model llama2_7b --fmt fp4 --dim 8
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

import llama_mx_common as C  # noqa: E402
import golden_llama_layer_e2e as GL  # noqa: E402

sys.path.insert(0, str(C.NPU))

CFG = {16: "mxgemmini_rtl.json", 8: "mxgemmini_rtl_dim8.json", 4: "mxgemmini_rtl_dim4.json", 32: "mxgemmini_rtl_dim32.json"}
_DIM = 16          # set in main(); read by the forked workers
_VL = 8


# ---------------------------------------------------------------- VPU: RSUM over DIM/2 lanes
def v_rsum(x):
    """RSUM over the last axis (a logical row of VL-lane spad rows): per spad row the RTL's pairwise lane tree in fp32,
    the rows accumulated in order in fp32, then BF16 (vpu_ref.h)."""
    t = GL.bf2f(x).reshape(x.shape[:-1] + (-1, _VL))
    while t.shape[-1] > 1:
        t = (t[..., 0::2] + t[..., 1::2]).astype(np.float32)
    rs = t[..., 0]
    acc = rs[..., 0].copy()
    for j in range(1, rs.shape[-1]):
        acc = (acc + rs[..., j]).astype(np.float32)
    return GL.d2bf(acc.astype(np.float64))


# ---------------------------------------------------------------- FP6 E3M2 LUT requant (mx_fp_math.h)
def e3m2_from_bf16(bits):
    """bf16_bits_to_fp6_e3m2_code: RNE to E3M2 (subnormal quantum 2^-4), saturating at 28; zero / non-finite -> +-0."""
    b = np.asarray(bits, np.int64)
    s5, E8, Mn = ((b >> 15) & 1) << 5, (b >> 7) & 0xFF, b & 0x7F
    e = E8 - 127
    av = (1.0 + Mn / 128.0) * np.exp2(np.clip(e, -140, 140).astype(np.float64))
    k = np.rint(av / 0.0625).astype(np.int64)
    sub = np.where(k <= 0, s5, np.where(k >= 4, s5 | 4, s5 | k))
    eu = np.minimum(e, 4)
    base = np.exp2(eu.astype(np.float64))
    mm = np.rint((av - base) / (base / 4)).astype(np.int64)
    carry = mm >= 4
    eu2 = np.where(carry, eu + 1, eu)
    mm = np.where(carry, 0, np.clip(mm, 0, 3))
    sat = eu2 > 4
    eu2, mm = np.where(sat, 4, eu2), np.where(sat, 3, mm)
    norm = s5 | (((eu2 + 3) & 7) << 2) | mm
    code = np.where(e < -2, sub, norm)
    return np.where((E8 == 0) | (E8 == 0xFF), s5, code).astype(np.int64)


def fp6_fixed(c):
    """fp6_to_fixed_point (FP6E3M2NearestFinder.scala): 9-bit signed fixed point, magnitude masked to 8 bits."""
    c = np.asarray(c, np.int64) & 0x3F
    sign, ex, mn = (c >> 5) & 1, (c >> 2) & 7, c & 3
    s_exp = np.where(ex == 0, -2, ex - 3)
    sig = (np.where(ex == 0, 0, 1) << 2) | mn
    mag = ((sig << ((s_exp + 2) & 7)) & 0x1FF) & 0xFF
    return np.where((ex == 0) & (mn == 0), 0, np.where(sign == 1, -mag, mag))


def finder(codes, table):
    """fp6e3m2_nearest_finder: |fixed(in) - fixed(entry)| & 0x1FF, the first minimum (lower index on ties)."""
    d = np.abs(fp6_fixed(codes)[..., None] - fp6_fixed(table)[None, :]) & 0x1FF
    return np.argmin(d, axis=-1).astype(np.uint8)


def spad_requant_lut(bits, table):
    """[M][N] BF16 -> (indices [M][N], E8M0 [M][N/32]): the SPAD_REQUANT block scale (as E4M3), each element bf16(v /
    scale) -> E3M2 code -> the finder's index in the tensor's table (every row group shares it)."""
    M, N = bits.shape
    v = GL.bf2f(bits).reshape(M, N // 32, 32)
    fin = np.isfinite(v)
    max_abs = np.max(np.where(fin, np.abs(v), 0), axis=-1)
    bad = np.isnan(v).any(-1) | np.isinf(v).any(-1)
    _, ex = np.frexp(np.maximum(max_abs, GL.FLT_EPS))
    sc = np.clip(ex - 1 + 127, 0, 254)
    scale = np.ldexp(np.float32(1.0), (sc - 127).astype(np.int64)).astype(np.float32)
    scale = np.where(bad, np.float32(np.nan), scale).astype(np.float32)
    sc = np.where(bad, 0xFF, sc).astype(np.uint8)
    with np.errstate(invalid="ignore"):
        codes = e3m2_from_bf16(GL.f2bf_mesh(v / scale[..., None]))
    return finder(codes, table).reshape(M, N), sc


def unpack_luts(words):
    """[n][3] uint32 -> [n][16] E3M2 codes."""
    out = np.zeros((words.shape[0], 16), np.int64)
    for i, w in enumerate(words):
        x = int(w[0]) | (int(w[1]) << 32) | (int(w[2]) << 64)
        out[i] = [(x >> (6 * e)) & 0x3F for e in range(16)]
    return out


# ---------------------------------------------------------------- the mesh: rtl_exact at this DIM
def _mesh_cols(args):
    import torch
    from rtl_exact import rtl_datapath as RD
    torch.set_num_threads(1)
    n0, PA, XA, PB, XB = args
    cfg = RD.load_config(C.RTL_EXACT / CFG[_DIM])
    FM = RD._golden(cfg)
    PA, XA, PB, XB = (torch.from_numpy(np.ascontiguousarray(t)) for t in (PA, XA, PB, XB))
    K, M = PA.shape
    lanes = [tuple(x) for x in cfg.acc_schedule][:cfg.window]
    C_ = torch.zeros((M, PB.shape[1]), dtype=torch.float32)
    for w in range(K // cfg.window):
        S = torch.zeros_like(C_)
        for l in range(cfg.window):
            kk = w * cfg.window + l
            outer = PA[kk].unsqueeze(1) * PB[kk].unsqueeze(0)
            S = RD.accumulate(S, RD.product_quantize(outer, cfg.product[0], cfg.product[1], FM), *lanes[l], FM)
        g = (w * cfg.window) // RD.BLOCK
        C_ = RD.cross_tile_accumulate(C_, S * (XA[g].unsqueeze(1) * XB[g].unsqueeze(0)), FM)
    return n0, C_.numpy()


_POOL = None


def mesh(a_val, a_sc, b_val, b_sc, pool=True):
    """C [M][N] BF16 bits = A [M][K] x B [K][N], operands as decoded element VALUES; a_sc E8M0 [M][K/32], b_sc
    [K/32][N]."""
    global _POOL
    PA = np.ascontiguousarray(np.asarray(a_val, np.float32).T)
    XA = np.exp2(np.asarray(a_sc, np.float64).T - 127).astype(np.float32)
    PB = np.asarray(b_val, np.float32)
    XB = np.exp2(np.asarray(b_sc, np.float64) - 127).astype(np.float32)
    N = PB.shape[1]
    ncores = len(os.sched_getaffinity(0))
    chunk = max(64, min(512, (N // ncores) // 64 * 64)) if pool else N
    items = [(n0, PA, XA, PB[:, n0:n0 + chunk], XB[:, n0:n0 + chunk]) for n0 in range(0, N, chunk)]
    if len(items) == 1 or not pool:
        Cf = np.empty((PA.shape[1], N), np.float32)
        for it in items:
            n0, c = _mesh_cols(it)
            Cf[:, n0:n0 + c.shape[1]] = c
        return GL.f2bf_mesh(Cf)
    if _POOL is None:
        import multiprocessing as mp
        _POOL = mp.get_context("fork").Pool(min(ncores, 250))
    Cf = np.empty((PA.shape[1], N), np.float32)
    for n0, c in _POOL.imap_unordered(_mesh_cols, items, chunksize=1):
        Cf[:, n0:n0 + c.shape[1]] = c
    return GL.f2bf_mesh(Cf)


# ---------------------------------------------------------------- data
def load(name):
    hdr = (C.DATA / f"{name}.h").read_text()
    off = {k: int(v) for k, v in re.findall(r"#define LMX_OFF_(\w+)\s+(\d+)u", hdr)}
    blob = np.fromfile(C.DATA / f"{name}.bin", dtype=np.uint8)

    def get(n, dtype, shape):
        k = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return blob[off[n]:off[n] + k].view(dtype).reshape(shape).copy()
    return get, off


def fnv(a):
    w = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
    assert len(w) % 8 == 0
    h = 1469598103934665603
    for x in w.view(np.uint64).tolist():
        h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


pk_rows = lambda c: (c[0::2] | (c[1::2] << 4)).astype(np.uint8)          # [M][N] -> [M/2][N]
unpk_cols = lambda p: np.stack([p & 15, p >> 4], axis=-1).reshape(p.shape[0], -1).astype(np.int64)   # [K][N/2] -> [K][N]

HASHES = ["xn1", "xn1_s", "q_bf", "k_bf", "v_bf", "q1", "q2", "k1", "k2", "v", "o", "o_s", "yattn", "hmid",
          "xn2", "xn2_s", "g", "u", "h", "h_s", "ymlp", "hout"]


# ---------------------------------------------------------------- attention, one query head per worker
_ATT = {}


def _head(hh):
    a = _ATT
    M, hd, H2, bl_list = a["M"], a["hd"], a["hd"] // 2, a["BL"]
    kv = hh // (a["NH"] // a["NKV"])
    q1_c, q2_c, q1_s, q2_s = a["q1_c"], a["q2_c"], a["q1_s"], a["q2_s"]
    qc = np.hstack([q1_c[:, hh * H2:(hh + 1) * H2], q2_c[:, hh * H2:(hh + 1) * H2]])
    qs = np.hstack([q1_s[:, hh * H2 // 32:(hh + 1) * H2 // 32], q2_s[:, hh * H2 // 32:(hh + 1) * H2 // 32]])
    KT, KTS, VC, VS = a["KT"][kv], a["KTS"][kv], a["VC"][kv], a["VS"][kv]
    qv = GL.e4m3_decode(qc)
    off, m_, L, O, alpha = 0, None, None, None, None
    for j, bl in enumerate(bl_list):
        S = mesh(qv, qs, GL.e4m3_decode(KT[:, off:off + bl]), KTS[:, off:off + bl], pool=False)
        if j == len(bl_list) - 1:
            S = GL.v_add(S, a["MASK"])
        mt = GL.v_rmax(S)
        if j == 0:
            m_ = mt
        else:
            mn = GL.v_max(m_, mt); alpha = GL.v_exp(GL.v_sub(m_, mn)); m_ = mn
        P = GL.v_exp(GL.v_sub(S, m_[:, None]))
        lt = v_rsum(P)
        L = lt if j == 0 else GL.v_add(GL.v_mul(L, alpha), lt)
        pc, ps = GL.spad_requant(P)
        Oj = mesh(GL.e4m3_decode(pc), ps, GL.e4m3_decode(VC[off:off + bl]), VS[off // 32:(off + bl) // 32], pool=False)
        O = Oj if j == 0 else GL.v_add(GL.v_mul(O, alpha[:, None]), Oj)
        off += bl
    return hh, GL.v_mul(O, GL.v_rcp(L)[:, None])


# ---------------------------------------------------------------- the layer
def run(model: str, fmt: str, dim: int) -> int:
    global _DIM, _VL
    _DIM, _VL = dim, C.vlanes(dim)
    m = C.MODELS[model]
    name = C.data_name(model, fmt)
    get, off = load(name)
    M, SK, D, F, NH, NKV, hd = m.m, m.sk, m.D, m.F, m.NH, m.NKV, m.HD
    SKT, LKV, QH, KH, H2 = SK + M, NKV * hd, D // 2, NKV * hd // 2, hd // 2
    nc = C.rms_nc(D, dim, fmt)
    bk, bkf, bkl = C.attn_blocks(m, dim)
    t0 = time.time()
    print(f"golden {name} DIM {dim}: RMSNorm chunk {nc}, key blocks {bkf} + {bk} x {(SKT - bkf - bkl) // bk} + {bkl}",
          flush=True)
    got = {}

    def check(n, arr):
        got[n] = fnv(arr)
        print(f"  {n:6s} {got[n]:016x}  ({time.time() - t0:.0f}s)", flush=True)

    # format plumbing: requant of a projection input, decode of A / B operands, the hashed code layout
    tabs = {}
    if fmt == "fp6":
        for n in ("XN1", "O", "XN2", "H"):
            tabs[n] = unpack_luts(get(f"L_{n}", np.uint32, (C.LUT_COPIES, 3)))[0]
        fp6_val = lambda c: GL.bf2f(GL.d2bf(e3m2_val(c)))   # exact

    def rq(x, tname):
        if fmt == "fp8":
            return GL.spad_requant(x)
        if fmt == "fp4":
            return GL.spad_requant_fp4(x, 0)
        return spad_requant_lut(x, tabs[tname])

    def a_val(c, tname):
        if fmt == "fp8":
            return GL.e4m3_decode(c)
        if fmt == "fp4":
            return GL.fp4_decode(c)
        return e3m2_val(tabs[tname][c])

    def w_val(n, K, N):
        s = get(f"W{n}_S", np.uint8, (K // 32, N))
        if fmt == "fp8":
            return GL.e4m3_decode(get(f"W{n}_C", np.uint8, (K, N))), s
        idx = unpk_cols(get(f"W{n}_C", np.uint8, (K, N // 2)))
        if fmt == "fp4":
            return GL.fp4_decode(idx), s
        T = unpack_luts(get(f"W{n}_L", np.uint32, (N >> C.LUT_G, 3)))
        return e3m2_val(T[np.arange(N)[None, :] >> C.LUT_G, idx]), s

    hashed = (lambda c: c) if fmt == "fp8" else pk_rows
    proj = lambda ac, as_, tn, n, K, N: mesh(a_val(ac, tn), as_, *w_val(n, K, N))
    inv_d, eps = GL.vpu_bf(1.0 / D), GL.vpu_bf(np.float32(m.eps))

    def rms(x, res, w, tname):
        if res is not None:
            x = GL.v_add(x, res)
        parts = [v_rsum(GL.v_mul(x[:, c:c + nc], x[:, c:c + nc])) for c in range(0, D, nc)]
        S = parts[0]
        for p in parts[1:]:
            S = GL.v_add(S, p)
        S = GL.v_rsqrt(GL.v_add(GL.v_mul(S, inv_d), eps))
        return rq(GL.v_mul(GL.v_mul(x, S[:, None]), w[None, :]), tname)

    H_PRE = get("H_PRE", np.uint16, (M, D))
    xn1_c, xn1_s = rms(H_PRE, None, get("W_IN_LN", np.uint16, (D,)), "XN1")
    check("xn1", hashed(xn1_c)); check("xn1_s", np.ascontiguousarray(xn1_s.T))
    Qb = proj(xn1_c, xn1_s, "XN1", "Q", D, D); check("q_bf", Qb)
    Kb = proj(xn1_c, xn1_s, "XN1", "K", D, LKV); check("k_bf", Kb)
    Vb = proj(xn1_c, xn1_s, "XN1", "V", D, LKV); check("v_bf", Vb)
    if m.bias:
        Qb = GL.v_add(Qb, get("BQ", np.uint16, (D,))[None, :])
        Kb = GL.v_add(Kb, get("BK", np.uint16, (LKV,))[None, :])
        Vb = GL.v_add(Vb, get("BV", np.uint16, (LKV,))[None, :])
    x1, x2 = GL.rope(Qb, get("ROPE_CQ", np.uint16, (M, QH)), get("ROPE_SQ", np.uint16, (M, QH)))
    q1_c, q1_s = GL.spad_requant(x1); q2_c, q2_s = GL.spad_requant(x2)
    check("q1", q1_c); check("q2", q2_c)
    k1, k2 = GL.rope(Kb, get("ROPE_CK", np.uint16, (M, KH)), get("ROPE_SK", np.uint16, (M, KH)))
    k1_c, k1_s = GL.spad_requant(k1); k2_c, k2_s = GL.spad_requant(k2)
    check("k1", k1_c); check("k2", k2_c)
    vt_c, vt_s = GL.spad_requant(np.ascontiguousarray(Vb.T))
    check("v", vt_c)
    KT = get("KT_CACHE", np.uint8, (NKV, hd, SKT)); KTS = get("KT_SCALES", np.uint8, (NKV, hd // 32, SKT))
    VC = get("V_CACHE", np.uint8, (NKV, SKT, hd)); VS = get("V_SCALES", np.uint8, (NKV, SKT // 32, hd))
    for g in range(NKV):
        KT[g, :H2, SK:] = k1_c[:, g * H2:(g + 1) * H2].T
        KT[g, H2:, SK:] = k2_c[:, g * H2:(g + 1) * H2].T
        KTS[g, :H2 // 32, SK:] = k1_s[:, g * H2 // 32:(g + 1) * H2 // 32].T
        KTS[g, H2 // 32:, SK:] = k2_s[:, g * H2 // 32:(g + 1) * H2 // 32].T
        VC[g, SK:, :] = vt_c[g * hd:(g + 1) * hd, :].T
        VS[g, SK // 32:, :] = vt_s[g * hd:(g + 1) * hd, :].T
    _ATT.update(M=M, hd=hd, NH=NH, NKV=NKV, KT=KT, KTS=KTS, VC=VC, VS=VS, MASK=get("MASK", np.uint16, (M, M)),
                q1_c=q1_c, q2_c=q2_c, q1_s=q1_s, q2_s=q2_s, BL=[bkf] + [bk] * ((SKT - bkf - bkl) // bk) + [bkl])
    import multiprocessing as mp
    Ob = np.zeros((NH, M, hd), np.uint16)
    with mp.get_context("fork").Pool(min(NH, len(os.sched_getaffinity(0)))) as pool:
        for hh, o in pool.imap_unordered(_head, range(NH)):
            Ob[hh] = o
    print(f"  attention, {NH} heads ({time.time() - t0:.0f}s)", flush=True)
    o_c = np.zeros((M, D), np.int64); o_s = np.zeros((M, D // 32), np.uint8)
    for hh in range(NH):
        c, s = rq(Ob[hh], "O")
        o_c[:, hh * hd:(hh + 1) * hd] = c
        o_s[:, hh * hd // 32:(hh + 1) * hd // 32] = s
    check("o", hashed(o_c.astype(np.uint8))); check("o_s", np.ascontiguousarray(o_s.T))
    Ya = proj(o_c, o_s, "O", "O", D, D); check("yattn", Ya)
    Hmid = GL.v_add(H_PRE, Ya); check("hmid", Hmid)
    xn2_c, xn2_s = rms(H_PRE, Ya, get("W_POST_LN", np.uint16, (D,)), "XN2")
    check("xn2", hashed(xn2_c)); check("xn2_s", np.ascontiguousarray(xn2_s.T))
    G = proj(xn2_c, xn2_s, "XN2", "G", D, F); check("g", G)
    U = proj(xn2_c, xn2_s, "XN2", "U", D, F); check("u", U)
    U = GL.v_mul(U, G)
    G = GL.v_rcp(GL.v_add(GL.v_exp(GL.v_mul(G, np.uint16(0xBF80))), np.uint16(0x3F80)))
    h_c, h_s = rq(GL.v_mul(U, G), "H")
    check("h", hashed(h_c)); check("h_s", np.ascontiguousarray(h_s.T))
    Ym = proj(h_c, h_s, "H", "D", F, D); check("ymlp", Ym)
    Hout = GL.v_add(Hmid, Ym); check("hout", Hout)

    # accuracy vs the fp64 layer (same BF16 input, unquantized weights)
    rel = lambda x, r: float(np.linalg.norm(x - r) / np.linalg.norm(r))
    f = lambda b: GL.bf2f(b).astype(np.float64)
    h = f(H_PRE)
    acc = {"yattn": rel(f(Ya), get("REF_YATTN", np.float32, (M, D))), "ymlp": rel(f(Ym), get("REF_YMLP", np.float32, (M, D))),
           "hout": rel(f(Hout), get("REF_HOUT", np.float32, (M, D))),
           "update": rel(f(Hout) - h, get("REF_HOUT", np.float32, (M, D)) - h)}
    print("  accuracy vs fp64: " + ", ".join(f"{k} {v * 100:.2f}%" for k, v in acc.items()), flush=True)

    ename = C.expect_name(model, fmt, dim)
    (C.DATA / f"{ename}.h").write_text(
        f"// GENERATED by gen/golden_llama_layer_mx.py -- do not edit. {m.name} layer, {fmt} projections, DIM {dim}:\n"
        f"// the rtl_exact golden ({CFG[dim]}). Accuracy vs the fp64 layer: "
        + ", ".join(f"{k} {v * 100:.2f}%" for k, v in acc.items()) + ".\n"
        f"#ifndef INCLUDE_LLAMA_MX_EXPECT_H\n#define INCLUDE_LLAMA_MX_EXPECT_H\n#include <stdint.h>\n"
        f"#define LMX_EXPECT_DIM {dim}\n#define LMX_RMS_NC {nc}     // RMSNorm column chunk (partial sums added in BF16)\n"
        f"#define LMX_BK {bk}\n#define LMX_BK_FIRST {bkf}\n#define LMX_BK_LAST {bkl}\n"
        f"#define LMX_NHASH {len(HASHES)}\n"
        f"static const char *const LMX_HASH_NAME[{len(HASHES)}] = {{" + ", ".join(f'"{n}"' for n in HASHES) + "};\n"
        f"static const uint64_t LMX_EXPECT[{len(HASHES)}] = {{"
        + ", ".join(f"0x{got[n]:016x}ULL" for n in HASHES) + "};\n#endif\n")
    print(f"golden {name} DIM {dim}: wrote {ename}.h ({time.time() - t0:.0f}s)", flush=True)
    return 0


def e3m2_val(c):
    from gen_mlp_proj_fp6 import e3m2_decode
    return e3m2_decode(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fmt", required=True, choices=list(C.FORMATS))
    ap.add_argument("--dim", type=int, required=True, choices=[4, 8, 16, 32])
    a = ap.parse_args()
    return run(a.model, a.fmt, a.dim)


if __name__ == "__main__":
    raise SystemExit(main())
