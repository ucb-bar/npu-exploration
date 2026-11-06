#!/usr/bin/env python3
"""Data + goldens for the dual-Gemmini A8W4 kernels (src/dual_mlp.c, src/dual_attn.c).

Random fp32 tensors -> MX operands -> every mesh matmul through fp8_matmul_model.tiled_matmul_hwlike (DIM-aware) and the
fp32 host glue emulated the way include/mx_host.h does it. Weights are FP4 E2M1 (W4), activations E4M3 single (A8):
  mlp : G = X@Wg, U = X@Wu (A8W4) -> H = silu(G)*U (host) -> Y = H@Wd (A8W4)
  attn: Q,K,V = X@W{q,k,v} (A8W4) -> S = Q@K^T (E4M3 x E4M3) -> P = softmax_causal(S/sqrt(hd)) (host)
        -> O = P@V (E4M3 x E4M3) -> Out = O@Wo (A8W4)

    gen_dual_a8w4.py --dim 16 --out data/     # writes dual_mlp_d16.h, dual_attn_d16.h
"""
import argparse
import os
import sys

import numpy as np

ROCC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../software/gemmini-rocc-tests"))
sys.path.insert(0, ROCC)
import torch  # noqa: E402
import fp8_matmul_model  # noqa: E402
from fp8_matmul_model import tiled_matmul_hwlike  # noqa: E402

BLOCK = 32
F32 = np.float32


# ---- formats ---------------------------------------------------------------------------------
def e4m3_decode(c):
    s, e, m = (c >> 7) & 1, (c >> 3) & 0xF, c & 7
    v = m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
    return -v if s else v


E4M3_MAG = np.array([e4m3_decode(c) for c in range(0x7F)], dtype=F32)  # codes 0..0x7E, increasing
FP4_MAG = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=F32)


def fp4_decode(c):
    v = FP4_MAG[c & 7]
    return -v if c & 8 else v


def e4m3_encode(v):
    """mx_host.h mx_e4m3_encode: nearest, ties to even, saturating at 0x7E."""
    v = F32(v)
    sign = 0x80 if (v < 0 or (v == 0 and np.signbit(v))) else 0
    a = abs(v)
    if a >= E4M3_MAG[-1]:
        return sign | 0x7E
    lo = int(np.searchsorted(E4M3_MAG, a, side="right")) - 1
    d_lo, d_hi = F32(a - E4M3_MAG[lo]), F32(E4M3_MAG[lo + 1] - a)
    up = d_hi < d_lo or (d_hi == d_lo and (lo & 1))
    return sign | (lo + 1 if up else lo)


def floor_log2(x):
    return int(np.frexp(F32(x))[1]) - 1


def e8m0_of(amax):
    amax = F32(amax)
    if not amax >= F32(1.1920929e-7):
        amax = F32(1.1920929e-7)
    return min(max(floor_log2(amax) + 127, 0), 254)


def quant_rows(V):
    """mx_quantize_rows: V[M][K] f32 -> codes [M][K], scales_a [K/32][M]."""
    M, K = V.shape
    codes = np.zeros((M, K), np.uint8)
    sc = np.zeros((K // BLOCK, M), np.uint8)
    for m in range(M):
        for g in range(K // BLOCK):
            blk = V[m, g * BLOCK:(g + 1) * BLOCK]
            s = e8m0_of(np.max(np.abs(blk)))
            sc[g, m] = s
            inv = F32(2.0 ** -(s - 127))
            codes[m, g * BLOCK:(g + 1) * BLOCK] = [e4m3_encode(F32(x) * inv) for x in blk]
    return codes, sc


def quant_cols(V):
    """mx_quantize_cols: V[R][C] f32 -> codes [R][C], scales_b [R/32][C]."""
    c, s = quant_rows(np.ascontiguousarray(V.T))
    return np.ascontiguousarray(c.T), s


def quant_fp4_weight(W):
    """FP4 weight [K][N]: per (32-row group, col) E8M0 = floor(log2 amax) - 2 (fp4 emax), nearest code."""
    K, N = W.shape
    codes = np.zeros((K, N), np.uint8)
    sc = np.zeros((K // BLOCK, N), np.uint8)
    grid = np.array([fp4_decode(c) for c in range(16)], dtype=F32)
    for g in range(K // BLOCK):
        for n in range(N):
            col = W[g * BLOCK:(g + 1) * BLOCK, n]
            s = min(max(floor_log2(max(np.max(np.abs(col)), 1e-30)) - 2 + 127, 0), 254)
            sc[g, n] = s
            x = col / F32(2.0 ** (s - 127))
            codes[g * BLOCK:(g + 1) * BLOCK, n] = np.abs(x[:, None] - grid[None, :]).argmin(1)
    return codes, sc


def nibble_pack(codes):
    return (codes[:, 1::2].astype(np.uint8) << 4) | codes[:, 0::2]


def bf16_to_f32(b):
    return (b.astype(np.uint32) << 16).view(F32)


# ---- mesh golden -------------------------------------------------------------------------------
def precision_for_dim(dim):
    acc_prec = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)] * 1
    prod = [(4, 3)] * dim
    if dim == 8:
        acc = [(8, 7)] * 8
    elif dim <= 16:
        acc = acc_prec[:dim]
    else:
        acc = acc_prec + [(8, 7)] * (dim - 16)
    return prod, acc


def mesh(dim, a_fp, a_sc, b_fp, b_sc):
    """a_fp [M][K] decoded elements, a_sc [K/32][M] e8m0; b_fp [K][N], b_sc [K/32][N] -> BF16 bits [M][N]."""
    fp8_matmul_model.INPUT_SPEC = "fp8:e4m3"
    fp8_matmul_model.TILE = dim
    for attr, v in (("TILE_K", dim), ("TILE_M", 2 * dim), ("TILE_N", 2 * dim)):
        if hasattr(fp8_matmul_model, attr):
            setattr(fp8_matmul_model, attr, v)
    prod, acc = precision_for_dim(dim)
    pw = lambda s: torch.tensor(2.0 ** (s.astype(np.float64) - 127), dtype=torch.float32)
    C = tiled_matmul_hwlike(torch.tensor(a_fp), torch.tensor(b_fp), pw(a_sc).T.contiguous(), pw(b_sc),
                            verbose=False, prod_precision_list=prod, acc_precision_list=acc)
    return C.to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)


def dec_e4m3(c):
    return np.vectorize(e4m3_decode, otypes=[F32])(c)


def dec_fp4(c):
    return np.vectorize(fp4_decode, otypes=[F32])(c)


# ---- host glue, as mx_host.h computes it ------------------------------------------------------
def expf(x):
    """Correctly rounded f32 exp (newlib expf is, in practice)."""
    return F32(np.exp(np.float64(F32(x))))


def silu(x):
    x = F32(x)
    return F32(x / F32(F32(1) + expf(-x)))


def swiglu(g_bf, u_bf):
    g, u = bf16_to_f32(g_bf), bf16_to_f32(u_bf)
    out = np.zeros(g.shape, F32)
    for i, (a, b) in enumerate(zip(g.ravel(), u.ravel())):
        out.flat[i] = F32(silu(a) * b)
    return out


def softmax_causal(s_bf, scale):
    M = s_bf.shape[0]
    s = bf16_to_f32(s_bf)
    out = np.zeros((M, M), F32)
    for m in range(M):
        v = [F32(s[m, j] * F32(scale)) for j in range(m + 1)]
        mx = max(v)
        e = [F32(expf(F32(x - mx))) for x in v]
        tot = F32(0)
        for x in e:
            tot = F32(tot + x)
        inv = F32(F32(1) / tot)
        out[m, :m + 1] = [F32(x * inv) for x in e]
    return out


# ---- emit --------------------------------------------------------------------------------------
def arr(name, a, ctype):
    dims = "".join(f"[{d}]" for d in a.shape)
    w = 2 if ctype == "uint8_t" else 4
    body = ", ".join(f"0x{int(v):0{w}x}" for v in a.ravel())
    return f"static const {ctype} {name}{dims} __attribute__((aligned(64))) = {{ {body} }};\n"


def header(path, defs, arrays):
    g = os.path.basename(path).upper().replace(".", "_")
    out = [f"// GENERATED by gen/gen_dual_a8w4.py -- do not edit.\n#ifndef {g}\n#define {g}\n#include <stdint.h>\n"]
    out += [f"#define {k} {v}\n" for k, v in defs.items()]
    out += [arr(n, a, t) for n, a, t in arrays]
    out.append(f"#endif // {g}\n")
    with open(path, "w") as f:
        f.write("".join(out))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True, choices=(8, 16, 32))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../data"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    d = args.dim
    rng = np.random.default_rng(args.seed)

    # ---------------- MLP ----------------
    M, D, F = 32, 64, 128
    X = rng.standard_normal((M, D)).astype(F32)
    x_c, x_s = quant_rows(X)
    wg_c, wg_s = quant_fp4_weight((rng.standard_normal((D, F)) * 0.2).astype(F32))
    wu_c, wu_s = quant_fp4_weight((rng.standard_normal((D, F)) * 0.2).astype(F32))
    wd_c, wd_s = quant_fp4_weight((rng.standard_normal((F, D)) * 0.1).astype(F32))
    x_fp = dec_e4m3(x_c)
    G = mesh(d, x_fp, x_s, dec_fp4(wg_c), wg_s)
    U = mesh(d, x_fp, x_s, dec_fp4(wu_c), wu_s)
    h_c, h_s = quant_rows(swiglu(G, U))
    Y = mesh(d, dec_e4m3(h_c), h_s, dec_fp4(wd_c), wd_s)
    header(os.path.join(args.out, f"dual_mlp_d{d}.h"),
           {"MLP_M": M, "MLP_D": D, "MLP_F": F},
           [("mlp_x", x_c, "uint8_t"), ("mlp_x_s", x_s, "uint8_t"),
            ("mlp_wg", nibble_pack(wg_c), "uint8_t"), ("mlp_wg_s", wg_s, "uint8_t"),
            ("mlp_wu", nibble_pack(wu_c), "uint8_t"), ("mlp_wu_s", wu_s, "uint8_t"),
            ("mlp_wd", nibble_pack(wd_c), "uint8_t"), ("mlp_wd_s", wd_s, "uint8_t"),
            ("mlp_g_ref", G, "uint16_t"), ("mlp_u_ref", U, "uint16_t"),
            ("mlp_h_ref", h_c, "uint8_t"), ("mlp_h_s_ref", h_s, "uint8_t"),
            ("mlp_y_ref", Y, "uint16_t")])

    # ---------------- attention, one head ----------------
    T, Dm, H = 32, 64, 32
    scale = F32(1.0 / np.sqrt(H))
    X = rng.standard_normal((T, Dm)).astype(F32)
    x_c, x_s = quant_rows(X)
    x_fp = dec_e4m3(x_c)
    W = {n: quant_fp4_weight((rng.standard_normal(s) * 0.25).astype(F32))
         for n, s in (("q", (Dm, H)), ("k", (Dm, H)), ("v", (Dm, H)), ("o", (H, Dm)))}
    Q, K, V = (mesh(d, x_fp, x_s, dec_fp4(W[n][0]), W[n][1]) for n in "qkv")
    q_c, q_s = quant_rows(bf16_to_f32(Q))
    kt_c, kt_s = quant_cols(np.ascontiguousarray(bf16_to_f32(K).T))      # B = K^T [H][T]
    S = mesh(d, dec_e4m3(q_c), q_s, dec_e4m3(kt_c), kt_s)
    p_c, p_s = quant_rows(softmax_causal(S, scale))
    v_c, v_s = quant_cols(bf16_to_f32(V))                                # B = V [T][H]
    O = mesh(d, dec_e4m3(p_c), p_s, dec_e4m3(v_c), v_s)
    o_c, o_s = quant_rows(bf16_to_f32(O))
    Out = mesh(d, dec_e4m3(o_c), o_s, dec_fp4(W["o"][0]), W["o"][1])
    arrays = [("att_x", x_c, "uint8_t"), ("att_x_s", x_s, "uint8_t")]
    for n in "qkvo":
        arrays += [(f"att_w{n}", nibble_pack(W[n][0]), "uint8_t"), (f"att_w{n}_s", W[n][1], "uint8_t")]
    arrays += [("att_q_ref", Q, "uint16_t"), ("att_k_ref", K, "uint16_t"), ("att_v_ref", V, "uint16_t"),
               ("att_s_ref", S, "uint16_t"), ("att_p_ref", p_c, "uint8_t"), ("att_o_ref", O, "uint16_t"),
               ("att_out_ref", Out, "uint16_t")]
    header(os.path.join(args.out, f"dual_attn_d{d}.h"),
           {"ATT_T": T, "ATT_DM": Dm, "ATT_H": H, "ATT_SCALE": f"{float(scale):.9g}f"}, arrays)


if __name__ == "__main__":
    main()
