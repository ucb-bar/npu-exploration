#!/usr/bin/env python3
"""Bit-exact Python golden of src/llama_layer_e2e.c, independent of Spike.

* matmuls: npu-exploration's rtl_exact datapath (rtl_exact/rtl_datapath.py primitives under mxgemmini_rtl.json: truncated
  PE product, both addends quantized to the lane, bf16 cross-tile accumulation), split over output columns;
* VPU ops: a vectorized port of include/vpu_ref.h (its exp / rcp / sqrt tables are read from the header);
* SPAD_REQUANT: E8M0 = floor(log2 amax) per 32 values, E4M3 RNE codes, a -0.0 element keeps its sign (as the RTL);
* mesh outputs to BF16 as the RTL / Spike store them (-0 -> +0).
Follows the kernel's order of operations and checks each stage against the kernel's 12 hashes
(data/llama_layer_e2e_expect.h, recorded from Spike).

--fp4: W4A4 projections (Q/K/V, o_proj, gate/up, down), attention unchanged in E4M3. Weights MXFP4 from the model's
fp32 weights (E8M0 = floor(log2 amax) - w_pmax, RNE, saturate at 6); the four projection inputs (xn1, O, xn2, h) by an
FP4 SPAD_REQUANT (E8M0 = floor(log2 amax) - a_pmax, then the requantizer's BF16 -> E3M1 -> E2M1 encoder); the same
rtl_exact mesh on the decoded values (Spike's FP4 x FP4 path: same product / lane schedule / bf16 cross-tile).

    ../../../.venv/bin/python3 golden_llama_layer_e2e.py [--fp4 [--w-pmax 0|1|2] [--a-pmax 0|1|2]]
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

HERE = Path(__file__).resolve().parent
KDIR = HERE.parent
NPU = HERE.parents[2]
ROCC = NPU.parent / "software" / "gemmini-rocc-tests"
sys.path.insert(0, str(NPU))

# ---------------------------------------------------------------- BF16 / VPU (port of vpu_ref.h)
_hdr = (ROCC / "include" / "vpu_ref.h").read_text()
def _tab(name):
    body = re.search(name + r"\[\d+\] = \{([^}]*)\}", _hdr).group(1)
    return np.array([int(x) for x in body.replace("\n", " ").split(",")], dtype=np.int64)
RCP_TAB, SQRT_TAB, EXP_TAB = _tab("vpu_rcp_tab"), _tab("vpu_sqrt_tab"), _tab("vpu_exp_tab")


def bf2f(b):
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def d2bf(x):
    """vpu_d_to_bf16: RNE from double once, subnormals kept, NaN -> 0x7fc0, overflow -> inf."""
    x = np.asarray(x, dtype=np.float64)
    u = x.view(np.uint64)
    sign = ((u >> np.uint64(48)) & np.uint64(0x8000)).astype(np.uint32)
    de = ((u >> np.uint64(52)) & np.uint64(0x7ff)).astype(np.int64)
    frac = u & np.uint64((1 << 52) - 1)
    e = de - 1023
    mant = frac | np.uint64(1 << 52)
    s = np.where(e >= -126, 45, -81 - e)
    s_c = np.clip(s, 1, 63).astype(np.uint64)
    n = mant >> s_c
    rem = mant & ((np.uint64(1) << s_c) - np.uint64(1))
    half = np.uint64(1) << (s_c - np.uint64(1))
    up = (rem > half) | ((rem == half) & ((n & np.uint64(1)) == np.uint64(1)))
    n = np.where(s >= 54, np.uint64(0), n + up.astype(np.uint64)).astype(np.int64)
    bits = np.where(e >= -126, ((e + 127) << 7) + (n - 128), n)
    bits = np.minimum(bits, 0x7f80).astype(np.uint32)
    out = sign | bits
    out = np.where(de == 0, sign, out)
    out = np.where(de == 0x7ff, np.where(frac != np.uint64(0), np.uint32(0x7fc0), sign | np.uint32(0x7f80)), out)
    return out.astype(np.uint16)


def f2bf_mesh(x):
    """mx_fp_math f32_to_bf16_rne (how the mesh output reaches BF16): -0 / tiny -> +0."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32).astype(np.uint64)
    nan = ((bits >> np.uint64(23)) & np.uint64(0xFF)) == np.uint64(0xFF)
    lsb = (bits >> np.uint64(16)) & np.uint64(1)
    out = ((bits + np.uint64(0x7FFF) + lsb) >> np.uint64(16)) & np.uint64(0xFFFF)
    out = np.where(nan, bits >> np.uint64(16), out)
    out = np.where((out & np.uint64(0x7FFF)) == np.uint64(0), np.uint64(0), out)
    return out.astype(np.uint16)


def v_add(a, b): return d2bf(bf2f(a).astype(np.float64) + bf2f(b).astype(np.float64))
def v_sub(a, b): return d2bf(bf2f(a).astype(np.float64) - bf2f(b).astype(np.float64))
def v_mul(a, b): return d2bf(bf2f(a).astype(np.float64) * bf2f(b).astype(np.float64))


def _ord(x):
    x = np.asarray(x, dtype=np.uint16)
    return np.where(x & 0x8000, ~x, x ^ 0x8000).astype(np.uint16)


def v_max(a, b):
    return np.where(_ord(a) > _ord(b), a, b).astype(np.uint16)


def _log2_17(y):
    return np.floor(np.log2(np.maximum(y, 1))).astype(np.int64)


def v_rcp(x):
    x = np.asarray(x, dtype=np.int64)
    neg, e, f = x >> 15, (x >> 7) & 0xff, x & 0x7f
    y = RCP_TAB[f]
    hi = _log2_17(y)
    bfe = 254 + (hi - 16) - e
    frac = (y >> np.where(hi > 7, hi - 7, 0)) & 0x7f
    sg = neg << 15
    out = sg | ((bfe & 0xff) << 7) | frac
    out = np.where(bfe <= 0, sg, out)
    out = np.where((e == 0) | (bfe >= 255), sg | 0x7f80, out)
    out = np.where(e == 255, sg, out)
    return out.astype(np.uint16)


def v_sqrt(x):
    x = np.asarray(x, dtype=np.int64)
    e, f = (x >> 7) & 0xff, x & 0x7f
    t = SQRT_TAB[f]
    base = np.where((e & 1) == 0, ((t * 92682) >> 16) & 0x1ffff, t)
    hi = _log2_17(base)
    sm = hi - 16 + e - 127
    bfe = 127 + (sm >> 1)
    frac = (base >> np.where(hi > 7, hi - 7, 0)) & 0x7f
    out = ((bfe & 0xff) << 7) | frac
    out = np.where((e == 255) & (f == 0) | (bfe >= 255), 0x7f80, out)
    out = np.where((e == 0) | ((e == 255) & (f != 0)), 0, out)
    return out.astype(np.uint16)


def v_rsqrt(x): return v_rcp(v_sqrt(x))


def v_exp(x):
    x = np.asarray(x, dtype=np.int64)
    sign, e, f = x >> 15, (x >> 7) & 0xff, x & 0x7f
    shift = e - 127 + 5
    m8 = 0x80 | f
    mag = np.where(shift < 0, m8 >> np.minimum(np.maximum(-shift, 0), 20), m8 << np.clip(shift, 0, 40))
    q = np.where(sign == 1, -mag, mag)
    prod = (q * 5909) >> 12
    k = prod >> 12
    r = prod & 0xfff
    addr, rl = r >> 7, r & 0x7f
    y0 = EXP_TAB[addr]
    y1 = np.where(addr == 31, 131071, EXP_TAB[np.minimum(addr + 1, 31)])
    interp = (y0 + (((y1 - y0) * rl) >> 7)) & 0x1ffff
    m, sticky = interp >> 7, (interp & 0x7f) != 0
    v = np.ldexp((2 * m + sticky).astype(np.float64), (k - 10).astype(np.int64))
    out = d2bf(v).astype(np.int64)
    out = np.where(sign & (e >= 0x86), 0, out)
    out = np.where((sign == 0) & ((e > 0x85) | ((e == 0x85) & (f > 0x31))), 0x7f80, out)
    out = np.where(e == 255, np.where(sign == 1, 0, 0x7f80), out)
    out = np.where(e == 0, 0x3f80, out)
    nan = (e == 255) & (f != 0)
    out = np.where(nan, (sign << 15) | 0x7f80 | ((((f & 0x40) == 0).astype(np.int64)) << 6) | 0x3f, out)
    return out.astype(np.uint16)


def v_rsum(x):
    """RSUM over the last axis (a logical row of 8-lane spad rows): per row ((s0+s1)+(s2+s3))+((s4+s5)+(s6+s7)) in
    fp32, rows accumulated in order in fp32, then BF16."""
    s = bf2f(x).reshape(x.shape[:-1] + (-1, 8))
    rs = ((s[..., 0] + s[..., 1]) + (s[..., 2] + s[..., 3])) + ((s[..., 4] + s[..., 5]) + (s[..., 6] + s[..., 7]))
    acc = rs[..., 0].copy()
    for j in range(1, rs.shape[-1]):
        acc = (acc + rs[..., j]).astype(np.float32)
    return d2bf(acc.astype(np.float64))


def v_rmax(x):
    o = _ord(x)
    idx = np.argmax(o, axis=-1)
    return np.take_along_axis(x, idx[..., None], axis=-1)[..., 0]


def vpu_bf(f):
    return d2bf(np.float64(np.float32(f)))


# ---------------------------------------------------------------- SPAD_REQUANT and E4M3
FLT_EPS = np.float32(1.1920929e-07)


def e4m3_decode(c):
    c = np.asarray(c, dtype=np.int64)
    s, e, m = (c >> 7) & 1, (c >> 3) & 0xf, c & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * np.exp2(e - 7.0))
    v = np.where(s == 1, -v, v)
    return np.where((c & 0x7f) == 0x7f, np.nan, v).astype(np.float32)


def e4m3_encode(v):
    """mx_fp_math fp8_e4m3_to_code (float32 math), with a -0.0 input -> 0x80 (SPAD_REQUANT, as the RTL)."""
    v = np.asarray(v, dtype=np.float32)
    s = np.signbit(v).astype(np.int64)
    av = np.abs(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        mfr, ex = np.frexp(av)
    E = (ex - 1).astype(np.int64)
    # subnormal range
    k_sub = np.rint(av / np.float32(2.0 ** -9)).astype(np.int64)
    sub = np.where(k_sub <= 0, s << 7, np.where(k_sub >= 8, (s << 7) | 8, (s << 7) | k_sub))
    base = np.ldexp(np.float32(1.0), np.clip(E, -126, 127)).astype(np.float32)
    k = np.rint((av - base) / (base / np.float32(8.0))).astype(np.int64)
    Eu = E.copy()
    up = k >= 8
    Eu = np.where(up, E + 1, Eu)
    k = np.where(up, 0, k)
    k = np.where(up & (Eu > 8), 6, k)
    Eu = np.where(up & (Eu > 8), 8, Eu)
    hi = np.where(Eu == 8, 6, 7)
    k = np.where(~up, np.clip(k, 0, hi), k)
    norm = (s << 7) | (((Eu + 7) & 0xf) << 3) | (k & 7)
    sat = (s << 7) | (((8 + 7) & 0xf) << 3) | 6
    code = np.where(E > 8, sat, np.where(E < -6, sub, norm))
    code = np.where(v == 0, np.where(s == 1, 0x80, 0), code)
    code = np.where(np.isinf(v), (s << 7) | 0x7f, code)
    code = np.where(np.isnan(v), 0x7f, code)
    return code.astype(np.uint8)


def spad_requant(bits):
    """[M][N] BF16 bits -> (E4M3 codes [M][N], E8M0 scales [M][N/32]) as SPAD_REQUANT computes them."""
    M, N = bits.shape
    v = bf2f(bits).reshape(M, N // 32, 32)
    fin = np.isfinite(v)
    max_abs = np.max(np.where(fin, np.abs(v), 0), axis=-1)
    has_nan, has_inf = np.isnan(v).any(-1), np.isinf(v).any(-1)
    amax = np.maximum(max_abs, FLT_EPS)
    _, ex = np.frexp(amax)
    sc = np.clip(ex - 1 + 127, 0, 254)
    scale = np.ldexp(np.float32(1.0), (sc - 127).astype(np.int64)).astype(np.float32)
    scale = np.where(has_nan, np.float32(np.nan), np.where(has_inf, np.float32(np.inf), scale)).astype(np.float32)
    sc = np.where(has_nan | has_inf, 0xFF, sc).astype(np.uint8)
    with np.errstate(invalid="ignore"):
        codes = e4m3_encode((v / scale[..., None]).astype(np.float32))
    return codes.reshape(M, N), sc


# ---------------------------------------------------------------- FP4 (E2M1)
FP4_MAG = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)


def fp4_decode(c):
    c = np.asarray(c, np.int64)
    v = FP4_MAG[c & 7]
    return np.where(c & 8, -v, v).astype(np.float32)


def fp4_encode_bf16(bits):
    """The requantizer's BF16 -> E2M1 code (fp4_matmul_model.hw_bf16_to_e2m1 / mx_fp_math bf16_bits_to_fp4_e2m1_code):
    RNE to E3M1, then the E3M1 -> E2M1 map. Zeros and underflow -> +0, overflow / Inf -> +-6, NaN -> s|7."""
    b = np.asarray(bits, np.int64)
    s, E, M = (b >> 15) & 1, (b >> 7) & 0xFF, b & 0x7F
    e = E - 127
    S = 128 + M
    q, r, st = (S >> 6) & 1, (S >> 5) & 1, (S & 0x1F) != 0
    sig2 = q + (r & (st | q))
    eo = np.where(sig2 >= 2, e + 1, e)
    mo = np.where(sig2 >= 2, 0, sig2)
    mag = (1 + 0.5 * mo) * np.exp2(np.clip(eo, -10, 10).astype(np.float64))
    mag = np.where(eo > 3, 8.0, mag)
    mag = np.where(e == -3, np.where(M >= 64, 0.25, 0.125), mag)
    mag = np.where(e == -4, np.where(M > 0, 0.125, 0.0), mag)
    mag = np.where(e <= -5, 0.0, mag)
    mc = np.select([mag <= 0.25, mag <= 0.5, mag <= 1, mag == 1.5, mag == 2, mag == 3, mag == 4], [0, 1, 2, 3, 4, 5, 6], 7)
    code = np.where(mc == 0, 0, (s << 3) | mc)
    code = np.where(E == 0, 0, code)
    code = np.where(E == 255, (s << 3) | 7, code)
    return code.astype(np.uint8)


def spad_requant_fp4(bits, pmax):
    """[M][N] BF16 bits -> (FP4 codes [M][N], E8M0 [M][N/32]): the SPAD_REQUANT block scale with the shift pmax, the
    element as the matmul FP4 requant post-pass (bf16(v / scale), then the E2M1 encoder)."""
    M, N = bits.shape
    v = bf2f(bits).reshape(M, N // 32, 32)
    fin = np.isfinite(v)
    max_abs = np.max(np.where(fin, np.abs(v), 0), axis=-1)
    bad = np.isnan(v).any(-1) | np.isinf(v).any(-1)
    _, ex = np.frexp(np.maximum(max_abs, FLT_EPS))
    sc = np.clip(ex - 1 - pmax + 127, 0, 254)
    scale = np.ldexp(np.float32(1.0), (sc - 127).astype(np.int64)).astype(np.float32)
    scale = np.where(bad, np.float32(np.nan), scale).astype(np.float32)
    sc = np.where(bad, 0xFF, sc).astype(np.uint8)
    with np.errstate(invalid="ignore"):
        codes = fp4_encode_bf16(f2bf_mesh(v / scale[..., None]))
    return codes.reshape(M, N), sc


def mxfp4_weights(W, pmax):
    """fp32 W [K][N] -> MXFP4 (codes [K][N], E8M0 [K/32][N]), blocks of 32 along K: scale 2^(floor(log2 amax) - pmax),
    amax floored at 2^-23, elements RNE (ties to the even code) saturating at 6. pmax=0 is the shipped operand
    convention (models/mxquant/block.py MXFP4)."""
    K, N = W.shape
    v = W.astype(np.float32).reshape(K // 32, 32, N)
    _, ex = np.frexp(np.maximum(np.abs(v).max(1), np.float32(2.0 ** -23)))
    se = (ex - 1 - pmax).astype(np.int64)
    a = np.abs(v / np.ldexp(np.float32(1.0), se)[:, None, :])
    hi = np.clip(np.searchsorted(FP4_MAG, a), 1, 7)
    lo = hi - 1
    dl, dh = a - FP4_MAG[lo], FP4_MAG[hi] - a
    mc = np.where(dl < dh, lo, np.where(dh < dl, hi, np.where(lo % 2 == 0, lo, hi)))
    mc = np.where(a >= 6, 7, mc)
    code = np.where(mc == 0, 0, (np.signbit(v).astype(np.int64) << 3) | mc)
    return code.reshape(K, N).astype(np.uint8), (se + 127).astype(np.uint8)


def fp4_model_weights():
    """The model's projection weights [K][N] fp32, Wq / Wk columns permuted as the E4M3 blob's."""
    import torch
    from transformers import AutoModelForCausalLM
    from gen_llama_layer_e2e import perm_cols
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    m = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float32)
    lay, c = m.model.layers[5], m.config
    hd = c.hidden_size // c.num_attention_heads
    t = lambda x: x.weight.T.detach().numpy().astype(np.float32)
    a, p = lay.self_attn, lay.mlp
    return {"Q": perm_cols(t(a.q_proj), c.num_attention_heads, hd), "K": perm_cols(t(a.k_proj), c.num_key_value_heads, hd),
            "V": t(a.v_proj), "O": t(a.o_proj), "G": t(p.gate_proj), "U": t(p.up_proj), "D": t(p.down_proj)}


# ---------------------------------------------------------------- the mesh: npu-exploration rtl_exact
_CFG = None


def _mesh_cols(args):
    """rtl_exact datapath (the batched loop of rtl_datapath.install, standalone) for one column chunk."""
    import torch
    from rtl_exact import rtl_datapath as RD
    torch.set_num_threads(1)
    n0, PA, XA, PB, XB = args
    cfg = RD.load_config()
    FM = RD._golden(cfg)
    PA, XA, PB, XB = (torch.from_numpy(np.ascontiguousarray(t)) for t in (PA, XA, PB, XB))
    K, M = PA.shape
    N = PB.shape[1]
    window = cfg.window
    lanes = [tuple(x) for x in cfg.acc_schedule][:window]
    C = torch.zeros((M, N), dtype=torch.float32)
    n_win = K // window
    for w in range(n_win):
        S_red = torch.zeros((M, N), dtype=torch.float32)
        for l in range(window):
            kk = w * window + l
            outer = PA[kk].unsqueeze(1) * PB[kk].unsqueeze(0)
            S_red = RD.accumulate(S_red, RD.product_quantize(outer, cfg.product[0], cfg.product[1], FM), *lanes[l], FM)
        g = (w * window) // RD.BLOCK
        C = RD.cross_tile_accumulate(C, S_red * (XA[g].unsqueeze(1) * XB[g].unsqueeze(0)), FM)
    return n0, C.numpy()


_POOL = None


def mesh(a_codes, a_sc, b_codes, b_sc, decode=e4m3_decode, chunk=32):
    """C [M][N] BF16 bits = A [M][K] x B [K][N]; a_sc E8M0 [M][K/32], b_sc E8M0 [K/32][N]."""
    global _POOL
    PA = decode(a_codes).T.astype(np.float32)                         # [K][M]
    XA = np.exp2(np.asarray(a_sc, np.float64).T - 127).astype(np.float32)   # [K/32][M]
    PB = decode(b_codes).astype(np.float32)
    XB = np.exp2(np.asarray(b_sc, np.float64) - 127).astype(np.float32)
    N = PB.shape[1]
    assert PA.shape[0] % 16 == 0
    items = [(n0, PA, XA, PB[:, n0:n0 + chunk], XB[:, n0:n0 + chunk]) for n0 in range(0, N, chunk)]
    if len(items) == 1:
        return f2bf_mesh(_mesh_cols(items[0])[1])
    if _POOL is None:
        import multiprocessing as mp
        _POOL = mp.get_context("fork").Pool(min(len(os.sched_getaffinity(0)), 250))
    C = np.empty((PA.shape[1], N), np.float32)
    for n0, c in _POOL.imap_unordered(_mesh_cols, items, chunksize=1):
        C[:, n0:n0 + c.shape[1]] = c
    return f2bf_mesh(C)


# ---------------------------------------------------------------- data
def load():
    hdr = (KDIR / "data" / "llama_layer_e2e.h").read_text()
    off = {k: int(v) for k, v in re.findall(r"#define E2E_OFF_(\w+)\s+(\d+)u", hdr)}
    dims = {k: v for k, v in re.findall(r"#define E2E_(M|SK|D|F|HD|NH|NKV)\s+(\d+)", hdr)}
    dims = {k: int(v) for k, v in dims.items()}
    blob = np.fromfile(KDIR / "data" / "llama_layer_e2e.bin", dtype=np.uint8)
    order = sorted(off.items(), key=lambda kv: kv[1])
    ends = {k: (order[i + 1][1] if i + 1 < len(order) else len(blob)) for i, (k, _) in enumerate(order)}
    def get(name, dtype, shape):
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return blob[off[name]:off[name] + n].view(dtype).reshape(shape).copy()
    return get, dims


def fnv(a):
    w = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
    w = w[: len(w) // 8 * 8].view(np.uint64)
    h = 1469598103934665603
    for x in w.tolist():
        h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def expected():
    t = (KDIR / "data" / "llama_layer_e2e_expect.h").read_text()
    return [int(x, 16) for x in re.findall(r"0x([0-9a-f]+)ULL", t)]


# ---------------------------------------------------------------- the layer
def rms(x, w, inv_d, eps, rq=spad_requant):
    y = v_mul(x, x)
    ss = v_rsum(y)
    ss = v_rsqrt(v_add(v_mul(ss, inv_d), eps))
    y = v_mul(x, ss[:, None])
    y = v_mul(y, w[None, :])
    return rq(y)


def rope(x, cs, sn):
    h = x.shape[1] // 2
    x1, x2 = x[:, :h], x[:, h:]
    b = v_mul(x2, sn); t2 = v_mul(x1, sn)
    x1 = v_sub(v_mul(x1, cs), b)
    x2 = v_add(v_mul(x2, cs), t2)
    return x1, x2


def accuracy(get, M, D, st, H_PRE):
    """rel_fro of each BF16 stage vs the fp64 reference (same BF16 input), and of the layer update vs the model."""
    rel = lambda x, r: float(np.linalg.norm(x - r) / np.linalg.norm(r))
    f = lambda b: bf2f(b).astype(np.float64)
    for k, v in st.items():
        print(f"  acc {k:6s} vs fp64 ref: {100 * rel(f(v), get('REF_' + k.upper(), np.float32, (M, D))):7.3f}%")
    h = f(H_PRE)
    upd = f(st["hout"]) - h
    print(f"  acc layer update vs fp64 ref: {100 * rel(upd, get('REF_HOUT', np.float32, (M, D)) - h):7.3f}%   "
          f"vs model hidden_states[6]: {100 * rel(upd, get('H_MODEL', np.float32, (M, D)) - h):7.3f}%   "
          f"(non-finite hout: {int((~np.isfinite(f(st['hout']))).sum())})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp4", action="store_true")
    ap.add_argument("--w-pmax", type=int, default=0)
    ap.add_argument("--a-pmax", type=int, default=0)
    a = ap.parse_args()
    get, d = load()
    M, D, F, HD, NH, NKV, SKC = d["M"], d["D"], d["F"], d["HD"], d["NH"], d["NKV"], d["SK"]
    SK = SKC + M
    KH, QH, LKV = NKV * HD // 2, D // 2, NKV * HD
    exp = expected()
    names = ["xn1", "q", "q2", "k", "v", "o", "yattn", "hmid", "xn2", "h", "ymlp", "hout"]
    got = {}
    def check(name, arr):
        h = fnv(arr); got[name] = h
        e = exp[names.index(name)]
        tag = "" if a.fp4 else ("== Spike" if h == e else "DIFFERS from Spike " + format(e, "016x"))
        print(f"  {name:6s} {h:016x} {tag}", flush=True)
    t0 = time.time()
    rq, dec = spad_requant, e4m3_decode
    if a.fp4:
        WF = {k: mxfp4_weights(v, a.w_pmax) for k, v in fp4_model_weights().items()}
        rq, dec = (lambda x: spad_requant_fp4(x, a.a_pmax)), fp4_decode
        print(f"  FP4 projections: w_pmax {a.w_pmax}, a_pmax {a.a_pmax} ({time.time() - t0:.0f}s)", flush=True)
    H_PRE = get("H_PRE", np.uint16, (M, D))
    inv_d, eps = vpu_bf(1.0 / D), vpu_bf(np.float32(1e-5))
    W = lambda n, k, nn: WF[n] if a.fp4 else (get(f"W{n}_CODES", np.uint8, (k, nn)), get(f"W{n}_SCALES", np.uint8, (k // 32, nn)))
    proj = lambda ac, as_, n, k, nn: mesh(ac, as_, *W(n, k, nn), decode=dec)

    xn1_c, xn1_s = rms(H_PRE, get("W_IN_LN", np.uint16, (D,)), inv_d, eps, rq); check("xn1", xn1_c)
    Q = proj(xn1_c, xn1_s, "Q", D, D)
    Kb = proj(xn1_c, xn1_s, "K", D, LKV)
    Vb = proj(xn1_c, xn1_s, "V", D, LKV)
    print(f"  Q/K/V ({time.time() - t0:.0f}s)", flush=True)
    x1, x2 = rope(Q, get("ROPE_CQ", np.uint16, (M, QH)), get("ROPE_SQ", np.uint16, (M, QH)))
    q1_c, q1_s = spad_requant(x1); q2_c, q2_s = spad_requant(x2)
    check("q", q1_c); check("q2", q2_c)
    k1, k2 = rope(Kb, get("ROPE_CK", np.uint16, (M, KH)), get("ROPE_SK", np.uint16, (M, KH)))
    k1_c, k1_s = spad_requant(k1); k2_c, k2_s = spad_requant(k2)
    check("k", k1_c)
    vt_c, vt_s = spad_requant(np.ascontiguousarray(Vb.T))
    check("v", vt_c)
    KT = get("KT_CACHE", np.uint8, (NKV, HD, SK)); KTS = get("KT_SCALES", np.uint8, (NKV, HD // 32, SK))
    VC = get("V_CACHE", np.uint8, (NKV, SK, HD)); VS = get("V_SCALES", np.uint8, (NKV, SK // 32, HD))
    for g in range(NKV):
        KT[g, :32, SKC:] = k1_c[:, g * 32:(g + 1) * 32].T
        KT[g, 32:, SKC:] = k2_c[:, g * 32:(g + 1) * 32].T
        KTS[g, 0, SKC:] = k1_s[:, g]; KTS[g, 1, SKC:] = k2_s[:, g]
        VC[g, SKC:, :] = vt_c[g * HD:(g + 1) * HD, :].T
        VS[g, SKC // 32:, :] = vt_s[g * HD:(g + 1) * HD, :].T
    MASK = get("MASK", np.uint16, (2 * M, M))
    # attention: 16 passes of 2 GQA heads, key blocks 128 x 16 + 64 (the last = the chunk, causal mask)
    BL = [128] * ((SK - M) // 128) + [M]
    o_c = np.zeros((NH, M, HD), np.uint8); o_s = np.zeros((NH // 2, 2 * M, 2), np.uint8)
    for p in range(NH // 2):
        kv, hs = p // (NH // 2 // NKV), (2 * p, 2 * p + 1)
        qc = np.vstack([np.hstack([q1_c[:, h * 32:(h + 1) * 32], q2_c[:, h * 32:(h + 1) * 32]]) for h in hs])
        qs = np.vstack([np.stack([q1_s[:, h], q2_s[:, h]], 1) for h in hs]).astype(np.int64) - 3   # 1/sqrt(64) fold
        off = 0; m_ = None; L = None; O = None; alphas = []
        for j, bl in enumerate(BL):
            S = mesh(qc, qs, KT[kv][:, off:off + bl], KTS[kv][:, off:off + bl])
            if j == len(BL) - 1:
                S = v_add(S, MASK)
            mt = v_rmax(S)
            if j == 0:
                m_ = mt
            else:
                mn = v_max(m_, mt); alphas.append(v_exp(v_sub(m_, mn))); m_ = mn
            P = v_exp(v_sub(S, m_[:, None]))
            lt = v_rsum(P)
            L = lt if j == 0 else v_add(v_mul(L, alphas[-1]), lt)
            pc, ps = spad_requant(P)
            Oj = mesh(pc, ps, VC[kv][off:off + bl], VS[kv][off // 32:(off + bl) // 32])
            O = Oj if j == 0 else v_add(v_mul(O, alphas[-1][:, None]), Oj)
            off += bl
        O = v_mul(O, v_rcp(L)[:, None])
        oc, osc = rq(O)
        o_c[hs[0]], o_c[hs[1]] = oc[:M], oc[M:]
        o_s[p] = osc
        print(f"  pass {p:2d} ({time.time() - t0:.0f}s)", flush=True)
    check("o", o_c)
    oa = np.concatenate([o_c[h] for h in range(NH)], axis=1)
    oa_s = np.zeros((M, D // 32), np.uint8)
    for h in range(NH):
        for g in range(2):
            oa_s[:, 2 * h + g] = o_s[h // 2][(h % 2) * M:(h % 2 + 1) * M, g]
    Ya = proj(oa, oa_s, "O", D, D); check("yattn", Ya)
    Hmid = v_add(H_PRE, Ya); check("hmid", Hmid)
    xn2_c, xn2_s = rms(Hmid, get("W_POST_LN", np.uint16, (D,)), inv_d, eps, rq); check("xn2", xn2_c)
    G = proj(xn2_c, xn2_s, "G", D, F)
    U = proj(xn2_c, xn2_s, "U", D, F)
    print(f"  G/U ({time.time() - t0:.0f}s)", flush=True)
    U = v_mul(U, G)
    G = v_rcp(v_add(v_exp(v_mul(G, np.uint16(0xBF80))), np.uint16(0x3F80)))
    U = v_mul(U, G)
    h_c, h_s = rq(U); check("h", h_c)
    Ym = proj(h_c, h_s, "D", F, D); check("ymlp", Ym)
    Hout = v_add(Hmid, Ym); check("hout", Hout)
    accuracy(get, M, D, {"yattn": Ya, "hmid": Hmid, "ymlp": Ym, "hout": Hout}, H_PRE)
    if a.fp4:
        print(f"FP4 golden done ({time.time() - t0:.0f}s)")
        return 0
    ok = all(got[n] == exp[names.index(n)] for n in names)
    print(f"golden vs Spike: {sum(got[n] == exp[names.index(n)] for n in names)}/{len(names)} stages identical "
          f"({time.time() - t0:.0f}s) -> {'MATCH' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
