#!/usr/bin/env python3
"""Data + goldens for src/dual_mlp_llama.c: one llama-shaped SwiGLU MLP (M=32 tokens, D=2048, F=1024 slice of
TinyLlama's 5632) in A8W4 (E4M3-single activations x FP4 E2M1 weights) split across the dual-Gemmini SoC:

  phase 1 (F split F16 | F8):  G_g = X @ Wg[:, F_g], U_g = X @ Wu[:, F_g]      on Gemmini g (DIM 16 / DIM 8)
                               H_g = silu(G_g) * U_g on that Gemmini's VPU (MULS -1, EXP, ADDS 1, RCP, MUL G, MUL U)
                               SPAD_REQUANT(H_g) -> E4M3 codes + E8M0 [F/32][M] scales into H_full
  phase 2 (N split N16 | N8):  Y[:, N_g] = H_full @ Wd[:, N_g]                 on Gemmini g

Everything goes to one blob, data/dual_mlp_llama.bin (64 B-aligned sections), with offsets in data/dual_mlp_llama.h.
Mesh goldens: fp8_matmul_model.tiled_matmul_hwlike at each Gemmini's DIM; VPU / requant: golden_llama_layer_e2e ports.
"""
import argparse
import os
import sys
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_dual_a8w4 as A  # noqa: E402  (formats, quantizers, mesh())
import golden_llama_layer_e2e as G  # noqa: E402  (VPU + SPAD_REQUANT ports)

F32 = np.float32
M, D, F = 32, 2048, 1024
F16, N16 = 768, 1536          # 16x16's slices (3:1, the measured A8W4 rate ratio at M = 32); the 8x8 takes the rest


def quant_rows_fast(V):
    """mx_quantize_rows, vectorized: codes [M][K], scales [K/32][M] (MXQuant e2e: X = 2^floor(log2 amax))."""
    Mr, K = V.shape
    blk = V.reshape(Mr, K // 32, 32)
    amax = np.maximum(np.abs(blk).max(-1), F32(1.1920929e-7))
    s = np.clip(np.frexp(amax)[1] - 1 + 127, 0, 254)
    x = (blk / np.ldexp(F32(1), (s - 127))[..., None]).astype(F32)
    codes = G.e4m3_encode(x).reshape(Mr, K).astype(np.uint8)
    return codes, s.T.astype(np.uint8).copy()


def quant_fp4_fast(W):
    """FP4 weight [K][N]: E8M0 per (32-row group, col) = floor(log2 amax) - 2, nearest E2M1 code."""
    K, N = W.shape
    blk = W.reshape(K // 32, 32, N)
    amax = np.maximum(np.abs(blk).max(1), 1e-30)
    s = np.clip(np.frexp(amax)[1] - 1 - 2 + 127, 0, 254)
    x = blk / np.ldexp(1.0, s - 127)[:, None, :]
    grid = np.array([A.fp4_decode(c) for c in range(16)], F32)
    codes = np.abs(x[..., None] - grid).argmin(-1).astype(np.uint8)
    return codes.reshape(K, N), s.astype(np.uint8)


def _mesh_job(args):
    dim, a_fp, a_sc, b_fp, b_sc, n0 = args
    return n0, A.mesh(dim, a_fp, a_sc, b_fp, b_sc)


def mesh_cols(dim, a_fp, a_sc, b_fp, b_sc, chunk=64, pool=None):
    """A.mesh over column chunks in parallel (columns are independent)."""
    N = b_fp.shape[1]
    jobs = [(dim, a_fp, a_sc, b_fp[:, n:n + chunk], b_sc[:, n:n + chunk], n) for n in range(0, N, chunk)]
    out = np.zeros((a_fp.shape[0], N), np.uint16)
    for n0, c in pool.imap_unordered(_mesh_job, jobs):
        out[:, n0:n0 + c.shape[1]] = c
    return out


def swiglu_vpu(g, u):
    one, neg = np.uint16(0x3F80), np.uint16(0xBF80)
    t = G.v_mul(g, np.full_like(g, neg))
    t = G.v_exp(t)
    t = G.v_add(t, np.full_like(t, one))
    t = G.v_rcp(t)
    t = G.v_mul(t, g)
    return G.v_mul(t, u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "../data"))
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pool = get_context("fork").Pool(min(len(os.sched_getaffinity(0)), 64))

    x_c, x_s = quant_rows_fast(rng.standard_normal((M, D)).astype(F32))
    wg_c, wg_s = quant_fp4_fast((rng.standard_normal((D, F)) / np.sqrt(D)).astype(F32))
    wu_c, wu_s = quant_fp4_fast((rng.standard_normal((D, F)) / np.sqrt(D)).astype(F32))
    wd_c, wd_s = quant_fp4_fast((rng.standard_normal((F, D)) / np.sqrt(F)).astype(F32))
    x_fp = A.dec_e4m3(x_c)
    dec4 = lambda c: A.dec_fp4(c)

    H = np.zeros((M, F), np.uint16)
    for dim, f0, f1 in ((16, 0, F16), (8, F16, F)):
        g = mesh_cols(dim, x_fp, x_s, dec4(wg_c[:, f0:f1]), wg_s[:, f0:f1], pool=pool)
        u = mesh_cols(dim, x_fp, x_s, dec4(wu_c[:, f0:f1]), wu_s[:, f0:f1], pool=pool)
        H[:, f0:f1] = swiglu_vpu(g, u)
        print(f"phase 1 dim{dim}: G/U/H [{f0}:{f1}] done", flush=True)
    h_c, h_s_mn = G.spad_requant(H)                 # codes [M][F], scales [M][F/32]
    h_s = h_s_mn.T.copy()                           # block-major [F/32][M], as SPAD_REQUANT writes DRAM
    Y = np.zeros((M, D), np.uint16)
    h_fp = A.dec_e4m3(h_c)
    for dim, n0, n1 in ((16, 0, N16), (8, N16, D)):
        Y[:, n0:n1] = mesh_cols(dim, h_fp, h_s, dec4(wd_c[:, n0:n1]), wd_s[:, n0:n1], pool=pool)
        print(f"phase 2 dim{dim}: Y [{n0}:{n1}] done", flush=True)

    secs = [("X", x_c), ("X_SC", x_s), ("WG", A.nibble_pack(wg_c)), ("WG_SC", wg_s), ("WU", A.nibble_pack(wu_c)),
            ("WU_SC", wu_s), ("WD", A.nibble_pack(wd_c)), ("WD_SC", wd_s), ("H_REF", h_c), ("H_SC_REF", h_s),
            ("Y_REF", Y)]
    blob, offs = bytearray(), {}
    for name, a in secs:
        blob += b"\0" * (-len(blob) % 64)
        offs[name] = len(blob)
        blob += np.ascontiguousarray(a).tobytes()
    with open(os.path.join(args.out, "dual_mlp_llama.bin"), "wb") as f:
        f.write(blob)
    h = ["// GENERATED by gen/gen_dual_mlp_llama.py -- do not edit.", "#ifndef DUAL_MLP_LLAMA_H", "#define DUAL_MLP_LLAMA_H",
         "#include <stdint.h>", f"#define DML_M {M}", f"#define DML_D {D}", f"#define DML_F {F}",
         f"#define DML_F16 {F16}", f"#define DML_N16 {N16}",
         "extern const uint8_t _binary_dual_mlp_llama_bin_start[];",
         "#define DML_BLOB(off) ((const uint8_t *) (_binary_dual_mlp_llama_bin_start + (off)))"]
    h += [f"#define DML_OFF_{k} {v}u" for k, v in offs.items()]
    h += ["#endif"]
    with open(os.path.join(args.out, "dual_mlp_llama.h"), "w") as f:
        f.write("\n".join(h) + "\n")
    print(f"wrote dual_mlp_llama.bin ({len(blob)} B) + .h")


if __name__ == "__main__":
    main()
