#!/usr/bin/env python3
"""Data + bit-exact golden for src/mlp_proj_fp6.c: the TinyLlama MLP's gate and down projections in FP6 E3M2 LUT form
(4-bit codebook indices + E8M0 block scales), one blob per LUT update granularity G (2^G rows of A / columns of B share
a 16-entry E3M2 codebook):

  gate: G[:, :NG] = xn @ Wg[:, :NG]              K 2048
  down: Y[:, :ND] = h[:, :KD] @ Wd[:KD, :ND]     K KD

xn = the layer's RMSNorm output (BF16, golden_llama_layer_e2e.rms); h = silu(xn Wg) * (xn Wu) in fp32, rounded to BF16.
Each operand: MX E3M2 (E8M0 = floor(log2 amax), the shipped MxGemmini / MXQuant convention as FP8 / FP4 pmax 0:
values below 2, so the E4M3-wide PE product and the narrow accumulator lanes cannot overflow; RNE), then per group a 16-code codebook by k-means
on the group's E3M2 code histogram, each element -> its nearest codebook entry. The golden runs the deprojected E3M2
values through npu-exploration's rtl_exact mesh (bit-exact with the RTL-verified LUT headers and Spike).

    ../../../.venv/bin/python3 gen_mlp_proj_fp6.py [--g 0 1 2 3 4 5] [--ng 256 --nd 512 --kd 2048]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import golden_llama_layer_e2e as GL
from gen_llama_mlp_fp4 import pk_cols, pk_rows, DATA

# E3M2: s eee mm, bias 3, subnormals at e = 0, max 28
_C = np.arange(64)
_S, _E, _M = (_C >> 5) & 1, (_C >> 2) & 7, _C & 3
E3M2_VAL = np.where(_E == 0, _M / 4.0 * 0.25, (1 + _M / 4.0) * np.exp2(_E - 3.0)) * np.where(_S == 1, -1.0, 1.0)
POS = E3M2_VAL[:32]   # codes 0..31, increasing


def e3m2_decode(c):
    return E3M2_VAL[np.asarray(c, np.int64)].astype(np.float32)


def e3m2_encode(v):
    """float -> E3M2 code, RNE (ties to the even code), saturating at 28."""
    a = np.minimum(np.abs(v), 28.0)
    hi = np.clip(np.searchsorted(POS, a), 1, 31)
    lo = hi - 1
    dl, dh = a - POS[lo], POS[hi] - a
    m = np.where(dl < dh, lo, np.where(dh < dl, hi, np.where(lo % 2 == 0, lo, hi)))
    return np.where((m == 0) | ~np.signbit(v), m, m | 32).astype(np.int64)


def mx_e3m2(X, axis):
    """X [R][C] fp32, blocks of 32 along `axis` (1: along columns = K for A; 0: along rows = K for B) ->
    (codes like X, E8M0 [R][C/32] or [R/32][C])."""
    Xb = X.reshape(X.shape[0], -1, 32) if axis == 1 else X.reshape(-1, 32, X.shape[1])
    amax = np.abs(Xb).max(axis=2 if axis == 1 else 1)
    _, ex = np.frexp(np.maximum(amax, np.float32(2.0 ** -23)))
    se = (ex - 1).astype(np.int64)
    sc = np.ldexp(np.float32(1.0), se)
    v = Xb / (sc[:, :, None] if axis == 1 else sc[:, None, :])
    return e3m2_encode(v).reshape(X.shape), (se + 127).astype(np.uint8)


def kmeans_codebook(hist):
    """16 distinct E3M2 codes for a 64-bin code histogram: weighted 1-D k-means on the values, centers snapped to codes."""
    used = np.nonzero(hist)[0]
    if len(used) <= 16:
        cb = list(used)
    else:
        vals, w = E3M2_VAL[used], hist[used].astype(np.float64)
        order = np.argsort(vals); vals, w = vals[order], w[order]
        cw = np.cumsum(w) / w.sum()
        cent = vals[np.searchsorted(cw, (np.arange(16) + 0.5) / 16)].astype(np.float64)
        for _ in range(50):
            lab = np.argmin(np.abs(vals[:, None] - cent[None, :]), axis=1)
            new = np.array([np.average(vals[lab == j], weights=w[lab == j]) if (lab == j).any() else cent[j] for j in range(16)])
            if np.allclose(new, cent): break
            cent = new
        cb = []
        for c in cent:   # snap to the nearest unused code
            for code in np.argsort(np.abs(E3M2_VAL - c), kind="stable"):
                if code not in cb: cb.append(int(code)); break
    for code in np.argsort(np.abs(E3M2_VAL), kind="stable"):   # pad (small groups) with small values
        if len(cb) >= 16: break
        if code not in cb: cb.append(int(code))
    return np.array(sorted(cb, key=lambda c: (E3M2_VAL[c], c)), np.int64)


def lut_quantize(codes, axis, G):
    """codes [R][C] E3M2; groups of 2^G rows (axis 0, A) or columns (axis 1, B) -> (indices, codebooks [n][16])."""
    n = (codes.shape[0] if axis == 0 else codes.shape[1]) >> G
    cbs = np.zeros((n, 16), np.int64)
    idx = np.zeros_like(codes)
    for g in range(n):
        sl = (slice(g << G, (g + 1) << G), slice(None)) if axis == 0 else (slice(None), slice(g << G, (g + 1) << G))
        part = codes[sl]
        cb = kmeans_codebook(np.bincount(part.ravel(), minlength=64))
        cbs[g] = cb
        nearest = np.argmin(np.abs(E3M2_VAL[:, None] - E3M2_VAL[cb][None, :]), axis=1)   # code -> index (lowest on ties)
        idx[sl] = nearest[part]
    return idx, cbs


def pack_luts(cbs):
    """[n][16] 6-bit codes -> [n][3] uint32 (entry e at bits 6e of the 96-bit codebook)."""
    out = np.zeros((cbs.shape[0], 3), np.uint32)
    for i, cb in enumerate(cbs):
        x = sum(int(c) << (6 * e) for e, c in enumerate(cb))
        out[i] = [(x >> (32 * w)) & 0xFFFFFFFF for w in range(3)]
    return out


def rel_err(C, ref):
    return float(np.linalg.norm(GL.bf2f(C).astype(np.float64) - ref) / np.linalg.norm(ref))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--ng", type=int, default=256)
    ap.add_argument("--nd", type=int, default=512)
    ap.add_argument("--kd", type=int, default=2048)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    get, d = GL.load()
    M, D = d["M"], d["D"]
    eps = float(next(l for l in (DATA / "llama_layer_e2e.h").read_text().splitlines() if "E2E_EPS" in l).split()[2].rstrip("f"))
    H, w = get("H_PRE", np.uint16, (M, D)), get("W_POST_LN", np.uint16, (D,))
    Wf = GL.fp4_model_weights()
    xn_b = GL.rms(H, w, GL.vpu_bf(1.0 / D), GL.vpu_bf(np.float32(eps)), rq=lambda y: y)   # BF16 bits
    xn = GL.bf2f(xn_b)
    KD = a.kd
    g32 = xn @ Wf["G"][:, :KD]; u32 = xn @ Wf["U"][:, :KD]
    h = GL.bf2f(GL.d2bf((g32 / (1 + np.exp(-g32.astype(np.float64)))) * u32))
    Wg, Wd = Wf["G"][:, :a.ng].astype(np.float32), Wf["D"][:KD, :a.nd].astype(np.float32)
    ref_g, ref_y = xn.astype(np.float64) @ Wg, h.astype(np.float64) @ Wd
    xc, xs = mx_e3m2(xn, 1); hc, hs = mx_e3m2(h, 1)
    wgc, wgs = mx_e3m2(Wg, 0); wdc, wds = mx_e3m2(Wd, 0)
    # MXFP6 without a LUT (every E3M2 code allowed): the accuracy the codebooks are measured against
    e_mx = (rel_err(GL.mesh(xc, xs, wgc, wgs, decode=e3m2_decode, chunk=8), ref_g),
            rel_err(GL.mesh(hc, hs, wdc, wds, decode=e3m2_decode, chunk=8), ref_y))
    print(f"MXFP6 E3M2 (no LUT): gate rel err {e_mx[0]:.4f}, down {e_mx[1]:.4f}")
    for G in a.g:
        assert (M >> G) <= 64, "act-in codebooks exceed the 64-entry table"
        xi, xl = lut_quantize(xc, 0, G); hi, hl = lut_quantize(hc, 0, G)
        gi, gl = lut_quantize(wgc, 1, G); di, dl = lut_quantize(wdc, 1, G)
        dec = lambda idx, cbs, axis: cbs[(np.arange(idx.shape[0])[:, None] >> G) if axis == 0 else
                                         (np.arange(idx.shape[1])[None, :] >> G), idx]
        Gs = GL.mesh(dec(xi, xl, 0), xs, dec(gi, gl, 1), wgs, decode=e3m2_decode, chunk=8)
        Ys = GL.mesh(dec(hi, hl, 0), hs, dec(di, dl, 1), wds, decode=e3m2_decode, chunk=8)
        parts = [("XN4", pk_rows(xi)), ("XN_S", np.ascontiguousarray(xs.T)), ("XN_L", pack_luts(xl)),
                 ("H4", pk_rows(hi)), ("H_S", np.ascontiguousarray(hs.T)), ("H_L", pack_luts(hl)),
                 ("WG4", pk_cols(gi)), ("WG_S", wgs), ("WG_L", pack_luts(gl)),
                 ("WD4", pk_cols(di)), ("WD_S", wds), ("WD_L", pack_luts(dl))]
        buf, offs = bytearray(), []
        for n, x in parts:
            buf += bytes((-len(buf)) % 64)
            offs.append((n, len(buf), x.shape))
            buf += np.ascontiguousarray(x).tobytes()
        name = f"mlp_proj_fp6{a.tag}_g{G}"
        eg, ey = rel_err(Gs, ref_g), rel_err(Ys, ref_y)
        (DATA / f"{name}.bin").write_bytes(bytes(buf))
        (DATA / f"{name}.h").write_text(
            "// GENERATED by gen/gen_mlp_proj_fp6.py -- do not edit. FP6 E3M2 LUT projections of the TinyLlama layer-5 MLP\n"
            f"// for src/mlp_proj_fp6.c, LUT update granularity G = {G}. Rel. error vs fp64: gate {eg:.4f}, down {ey:.4f}\n"
            f"// (MXFP6 without a LUT: {e_mx[0]:.4f}, {e_mx[1]:.4f}).\n"
            f"#ifndef INCLUDE_{name.upper()}_H\n#define INCLUDE_{name.upper()}_H\n#include <stdint.h>\n"
            f"#define MP_M {M}\n#define MP_D {D}\n#define MP_KD {KD}\n#define MP_NG {a.ng}\n#define MP_ND {a.nd}\n#define MP_G {G}\n"
            f"extern const uint8_t _binary_{name}_bin_start[];\n"
            f"#define MP_AT(off, type) ((const type *) (_binary_{name}_bin_start + (off)))\n"
            + "".join(f"#define MP_OFF_{n:<5s} {o}u   // {s}\n" for n, o, s in offs)
            + f"#define MP_HASH_G 0x{GL.fnv(Gs):016x}ULL\n#define MP_HASH_Y 0x{GL.fnv(Ys):016x}ULL\n#endif\n")
        print(f"  G={G}: wrote {name}.bin ({len(buf) / 1e6:.2f} MB); codebooks act {M >> G}, gate {a.ng >> G}, "
              f"down {a.nd >> G}; rel err gate {eg:.4f} down {ey:.4f}; hash G {GL.fnv(Gs):016x} Y {GL.fnv(Ys):016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
