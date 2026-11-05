#!/usr/bin/env python3
"""Bit-exact Python golden of src/moe_layer_mx.c, independent of Spike: one (input, format, mesh) build.

The same primitives as golden_llama_layer_mx.py: matmuls through rtl_exact under the mesh's accumulator config, the VPU
through golden_llama_layer_e2e's port of vpu_ref.h (RSUM's lane tree over DIM/2 lanes), SPAD_REQUANT for E4M3 / FP4.
Follows the kernel's order of operations -- the top-4 pick included, from the golden's own logits, so the routing is
the device's, not the generator's target -- and writes data/moe_mx_<cfg>_<fmt>_d<DIM>_expect.h: the chunk constants
that change numbers (RMSNorm's column chunk) or layouts, and the 13 stage hashes the kernel checks itself against.

    ../../../.venv/bin/python3 golden_moe_mx.py --cfg gptoss20b_t8 --fmt fp4 --dim 16
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

import moe_mx_common as C  # noqa: E402
import golden_llama_layer_e2e as GL  # noqa: E402
import golden_llama_layer_mx as GM  # noqa: E402

M_ = C.GPTOSS
HASHES = ["xr", "xr_s", "xe", "xe_s", "lg", "pw", "g", "u", "h", "h_s", "y", "moe", "hout"]


def fnv_cont(h, a):
    w = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
    assert len(w) % 8 == 0
    for x in w.view(np.uint64).tolist():
        h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


FNV0 = 1469598103934665603


def blob(name: str, prefix: str):
    hdr = (C.DATA / f"{name}.h").read_text()
    off = {k: int(v) for k, v in re.findall(rf"#define {prefix}(\w+)\s+(\d+)u", hdr)}
    return np.memmap(C.DATA / f"{name}.bin", dtype=np.uint8, mode="r"), off, hdr


_POOL = None


def _job(args):
    j, item = args
    n0, c = GM._mesh_cols(item)
    return j, n0, c


def mesh_many(jobs):
    """Several rtl_exact matmuls at once, [(a_val, a_sc, b_val, b_sc)] -> [C BF16], spread over every core in items of
    W columns (an item's cost is mostly per-op overhead: 576 columns take ~2x the time of 64)."""
    global _POOL
    items, outs = [], []
    for j, (a_val, a_sc, b_val, b_sc) in enumerate(jobs):
        PA = np.ascontiguousarray(np.asarray(a_val, np.float32).T)
        XA = np.exp2(np.asarray(a_sc, np.float64).T - 127).astype(np.float32)
        PB = np.asarray(b_val, np.float32)
        XB = np.exp2(np.asarray(b_sc, np.float64) - 127).astype(np.float32)
        outs.append(np.empty((PA.shape[1], PB.shape[1]), np.float32))
        W = 576 if PB.shape[1] % 576 == 0 else 64
        items += [(j, (n0, PA, XA, PB[:, n0:n0 + W], XB[:, n0:n0 + W])) for n0 in range(0, PB.shape[1], W)]
    if _POOL is None:
        import multiprocessing as mp
        _POOL = mp.get_context("fork").Pool(min(len(os.sched_getaffinity(0)), 250))
    for j, n0, c in _POOL.imap_unordered(_job, items, chunksize=1):
        outs[j][:, n0:n0 + c.shape[1]] = c
    return [GL.f2bf_mesh(o) for o in outs]


def pad_rows(c, rows):
    out = np.zeros((rows, c.shape[1]), np.uint16)
    out[:c.shape[0]] = c
    return out


def ord16(x):
    x = np.asarray(x, np.uint16).astype(np.int64)
    return np.where(x & 0x8000, (~x) & 0xFFFF, x ^ 0x8000)


def run(cfg: str, fmt: str, dim: int) -> int:
    GM._DIM, GM._VL = dim, C.vlanes(dim)
    T, D, F, E, K = C.CFGS[cfg].T, M_.D, M_.F, M_.E, M_.K
    MT, nc, fc, sw = C.mt(T, dim, fmt), C.chunk(D, dim), C.chunk(F, dim), C.softmax_width(dim)
    q = dim * (2 if fmt == "fp4" else 1)
    MX, MR = (T + q - 1) // q * q, (T + dim - 1) // dim * dim   # matmul rows (experts, router); the rest stay 0
    t0 = time.time()
    print(f"golden moe {cfg} {fmt} DIM {dim}: {T} tokens in {MT} rows (matmuls {MX}, router {MR}), chunks {nc} / {fc}, "
          f"softmax width {sw}", flush=True)
    ib, ioff, _ = blob(C.cfg_name(cfg), "MOE_OFF_")
    xb, xoff, xhdr = blob(C.experts_name(fmt), "MOE_XO_")
    stride = int(re.search(r"#define MOE_X_STRIDE (\d+)u", xhdr).group(1))

    def iget(n, dtype, shape):
        k = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return np.array(ib[ioff[n]:ioff[n] + k]).view(dtype).reshape(shape)

    def xget(e, n, dtype, shape):
        o = e * stride + xoff[n]
        k = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return np.array(xb[o:o + k]).view(dtype).reshape(shape)

    got = {}

    def check(n, v):
        got[n] = v
        print(f"  {n:5s} {v:016x}  ({time.time() - t0:.0f}s)", flush=True)

    pk = (lambda c: c.astype(np.uint8)) if fmt == "fp8" else GM.pk_rows
    rq = GL.spad_requant if fmt == "fp8" else (lambda x: GL.spad_requant_fp4(x, C.ACT_PMAX_FP4))
    a_val = GL.e4m3_decode if fmt == "fp8" else GL.fp4_decode

    def w_val(e, n, Kd, Nd):
        s = xget(e, f"{n}S", np.uint8, (Kd // 32, Nd))
        if fmt == "fp8":
            return GL.e4m3_decode(xget(e, f"{n}C", np.uint8, (Kd, Nd))), s
        return GL.fp4_decode(GM.unpk_cols(xget(e, f"{n}C", np.uint8, (Kd, Nd // 2)))), s

    # ---- RMSNorm -> E4M3 (router), Q(.) (experts)
    Hp = np.zeros((MT, D), np.uint16)
    Hp[:T] = iget("H", np.uint16, (T, D))
    w = iget("W_LN", np.uint16, (D,))
    inv_d, eps = GL.vpu_bf(1.0 / D), GL.vpu_bf(np.float32(M_.eps))
    parts = [GM.v_rsum(GL.v_mul(Hp[:, c:c + nc], Hp[:, c:c + nc])) for c in range(0, D, nc)]
    S = parts[0]
    for p in parts[1:]:
        S = GL.v_add(S, p)
    S = GL.v_rsqrt(GL.v_add(GL.v_mul(S, inv_d), eps))
    y = GL.v_mul(GL.v_mul(Hp, S[:, None]), w[None, :])
    xr_c, xr_s = GL.spad_requant(y)
    check("xr", GM.fnv(xr_c)); check("xr_s", GM.fnv(np.ascontiguousarray(xr_s.T)))
    xe_c, xe_s = (xr_c, xr_s) if fmt == "fp8" else rq(y)
    check("xe", GM.fnv(pk(xe_c))); check("xe_s", GM.fnv(np.ascontiguousarray(xe_s.T)))

    # ---- router + bias, top-4, softmax
    Lg, = mesh_many([(GL.e4m3_decode(xr_c[:MR]), xr_s[:MR], GL.e4m3_decode(iget("WR_C", np.uint8, (D, C.ROUTER_N))),
                      iget("WR_S", np.uint8, (D // 32, C.ROUTER_N)))])
    Lg = pad_rows(Lg, MT)
    Lg = GL.v_add(Lg, iget("BR", np.uint16, (C.ROUTER_N,))[None, :])
    check("lg", GM.fnv(Lg))
    Sm = np.zeros((MT, sw), np.uint16)
    pick = np.zeros((T, K), np.int64)
    for t in range(T):
        o = ord16(Lg[t, :E])
        pick[t] = sorted(range(E), key=lambda e: (-o[e], e))[:K]
        Sm[t, :K] = Lg[t, pick[t]]
        Sm[t, K:] = 0xFF80
    m = GL.v_rmax(Sm)
    P = GL.v_exp(GL.v_sub(Sm, m[:, None]))
    Pw = GL.v_mul(P, GL.v_rcp(GM.v_rsum(P))[:, None])
    check("pw", GM.fnv(Pw))
    touched = sorted(set(pick.reshape(-1).tolist()))
    tgt = iget("TARGET", np.uint8, (T, K))
    same = all(set(pick[t]) == set(tgt[t]) for t in range(T))
    print(f"  routing: {len(touched)} distinct experts ({'as targeted' if same else 'DIFFERS from the target'})", flush=True)

    # ---- the experts, ascending
    m1, p1, c7 = np.uint16(0xBF80), np.uint16(0x3F80), np.uint16(0xC0E0)
    ma = GL.vpu_bf(np.float32(-M_.alpha))
    hs = {k: FNV0 for k in ("g", "u", "h", "h_s", "y")}
    Acc = None
    GU = [pad_rows(c, MT) for c in mesh_many([(a_val(xe_c[:MX]), xe_s[:MX], *w_val(e, n, D, F))
                                               for e in touched for n in "GU"])]
    print(f"  gate/up of {len(touched)} experts ({time.time() - t0:.0f}s)", flush=True)
    HC = []
    for i, e in enumerate(touched):
        G, U = GU[2 * i], GU[2 * i + 1]
        g = GL.v_add(G, xget(e, "BG", np.uint16, (F,))[None, :])
        u = GL.v_add(U, xget(e, "BU", np.uint16, (F,))[None, :])
        g = GL.v_mul(GL.v_max(GL.v_mul(g, m1), c7), m1)
        u = GL.v_mul(GL.v_max(GL.v_mul(GL.v_max(u, c7), m1), c7), m1)
        u = GL.v_mul(GL.v_add(u, p1), g)
        g = GL.v_rcp(GL.v_add(GL.v_exp(GL.v_mul(g, ma)), p1))
        HC.append(rq(GL.v_mul(u, g)))
    YS = [pad_rows(c, MT) for c in mesh_many([(a_val(h_c[:MX]), h_s[:MX], *w_val(e, "D", F, D))
                                               for e, (h_c, h_s) in zip(touched, HC)])]
    print(f"  down of {len(touched)} experts ({time.time() - t0:.0f}s)", flush=True)
    for i, e in enumerate(touched):
        G, U, (h_c, h_s), Y = GU[2 * i], GU[2 * i + 1], HC[i], YS[i]
        hs["g"], hs["u"] = fnv_cont(hs["g"], G), fnv_cont(hs["u"], U)
        hs["h"], hs["h_s"] = fnv_cont(hs["h"], pk(h_c)), fnv_cont(hs["h_s"], np.ascontiguousarray(h_s.T))
        hs["y"] = fnv_cont(hs["y"], Y)
        wr = np.zeros(MT, np.uint16)
        for t in range(T):
            for k in range(K):
                if pick[t, k] == e:
                    wr[t] = Pw[t, k]
        y2 = GL.v_mul(GL.v_add(Y, xget(e, "BD", np.uint16, (D,))[None, :]), wr[:, None])
        Acc = y2 if i == 0 else GL.v_add(Acc, y2)
    for k in ("g", "u", "h", "h_s", "y"):
        check(k, hs[k])
    check("moe", GM.fnv(Acc))
    Hout = GL.v_add(Hp, Acc)
    check("hout", GM.fnv(Hout))

    f = lambda b: GL.bf2f(b[:T]).astype(np.float64)
    rm, rh = iget("REF_MOE", np.float32, (T, D)), iget("REF_HOUT", np.float32, (T, D))
    rel = lambda x, r: float(np.linalg.norm(x - r) / np.linalg.norm(r))
    acc = {"moe": rel(f(Acc), rm), "hout": rel(f(Hout), rh), "update": rel(f(Hout) - f(Hp), rh - f(Hp))}
    print("  accuracy vs fp64: " + ", ".join(f"{k} {v * 100:.2f}%" for k, v in acc.items()), flush=True)

    ename = C.expect_name(cfg, fmt, dim)
    (C.DATA / f"{ename}.h").write_text(
        f"// GENERATED by gen/golden_moe_mx.py -- do not edit. {M_.name} MoE block, {cfg}, {fmt} experts, DIM {dim}:\n"
        f"// the rtl_exact golden ({GM.CFG[dim]}); routing: {len(touched)} distinct experts. Accuracy vs the fp64 block: "
        + ", ".join(f"{k} {v * 100:.2f}%" for k, v in acc.items()) + ".\n"
        f"#ifndef INCLUDE_MOE_EXPECT_H\n#define INCLUDE_MOE_EXPECT_H\n#include <stdint.h>\n"
        f"#define MOE_EXPECT_DIM {dim}\n#define MOE_NC {nc}     // RMSNorm / combine / residual column chunk\n"
        f"#define MOE_FC {fc}     // activation column chunk\n#define MOE_SW {sw}     // softmax lanes per token\n"
        f"#define MOE_NHASH {len(HASHES)}\n"
        f"static const char *const MOE_HASH_NAME[{len(HASHES)}] = {{" + ", ".join(f'"{n}"' for n in HASHES) + "};\n"
        f"static const uint64_t MOE_EXPECT[{len(HASHES)}] = {{"
        + ", ".join(f"0x{got[n]:016x}ULL" for n in HASHES) + "};\n#endif\n")
    print(f"golden moe {cfg} {fmt} DIM {dim}: wrote {ename}.h ({time.time() - t0:.0f}s)", flush=True)
    if _POOL is not None:
        _POOL.close(); _POOL.join()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, choices=list(C.CFGS))
    ap.add_argument("--fmt", required=True, choices=list(C.FORMATS))
    ap.add_argument("--dim", type=int, required=True, choices=[4, 8, 16, 32])
    a = ap.parse_args()
    return run(a.cfg, a.fmt, a.dim)


if __name__ == "__main__":
    sys.path.insert(0, str(C.NPU))
    raise SystemExit(main())
