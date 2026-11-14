#!/usr/bin/env python3
"""Real TinyLlama data + goldens for npu-exploration's src/lutk_proj.c: the exact operands of the MXQuant
per-K-block LUT study (weights through the LUT, 32xN usage, per-K-block codebooks).

Same blob / header layout as gen/gen_lutk_proj.py (same LK_OFF_* names, same goldens), so lutk_proj.c runs on it
unchanged at G = 1. Instead of random operands it takes, from the MXQuant run behind the perplexity numbers:
  A      real layer inputs (WikiText-2 calibration sequence 0, tokens tok0 .. tok0+M-1), MXFP8 E4M3 codes with the
         gemmini scale rule (block max in [1, 2)), one E8M0 scale per 32 along K           -> A, AS
  W      the layer's weights, exact MXFP8 E4M3 codes + E8M0 scales                          -> B8, BS
  LUT    per-K-block codebooks fit as the study does (abhiKmeans, g^2-weighted k-means, one table per 32 K-rows x
         2^G output columns) and the 4-bit indices                                          -> BL, BLUT, BLUTC<i>
  masks  the study's LUT K-blocks of this layer at budgets 5 / 6 / 7 bits (from the run's jsonl), plus "all"
plus data/lutk_<tag>_ref.npz: the study's own fp32 output of every variant (quant.FAST, the arithmetic the
perplexities were computed with), the all-MXFP8 and unquantized outputs, codebooks, indices, masks.

B4 (MXFP4 ablation) is the nearest E2M1 of the MXFP8 element codes on the SAME scales, only so the blob keeps every
offset lutk_proj.c references. It is NOT the study's FP4 ablation (that one used OCP scales).

Two environments: the export needs the MXQuant env (transformers, mxq, the framework), the goldens need
npu-exploration's .venv (qtorch, fp8_matmul_model). Running it with the MXQuant python does the export and then
re-runs itself with --build under the .venv python. CPU only.

    cd npu-exploration/baremetal/mxgemmini
    /scratch/nicorakela/envs/mxquant/bin/python gen/gen_lutk_tinyllama.py \
        --layer model.layers.10.mlp.down_proj --m 64 --k 2048 --n 256 --g 1 --tag tl_down_g1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

NPU = Path("/tools/scratch/nicorakela/radiance-cy-dev/generators/gemmini/npu-exploration")
GEN = NPU / "baremetal" / "mxgemmini" / "gen"
DATA = GEN.parent / "data"
VENV_PY = NPU / ".venv" / "bin" / "python"
MXQ = "/scratch/nicorakela/experiments/MXQuant/mixed-precision-framework"
GROUP = 32


# ---------------------------------------------------------------------------------------------------- codes
def e4m3_table():
    """{value: code} of every finite E4M3 code (0x7F/0xFF excluded; 0.0 -> 0x00)."""
    out = {}
    for c in range(256):
        if (c & 0x7F) == 0x7F:
            continue
        s, e, m = (c >> 7) & 1, (c >> 3) & 0xF, c & 7
        v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
        v = -v if s else v
        if v == 0.0 and s:
            continue
        out[v] = c
    return out


def e4m3_codes(vals) -> np.ndarray:
    """E4M3 values (exact) -> uint8 codes; refuses values off the grid and mxgen's NaN codes (exp 1111, man != 0)."""
    lut = e4m3_table()
    arr = np.asarray(vals, np.float32)
    try:
        out = np.array([lut[float(v)] for v in arr.reshape(-1)], np.uint8)
    except KeyError as e:
        raise SystemExit(f"value {e} is not an E4M3 code")
    bad = ((out & 0x78) == 0x78) & ((out & 0x07) != 0)
    assert not bad.any(), f"{int(bad.sum())} codes in mxgen's NaN range (|v| >= 288)"
    return out.reshape(arr.shape)


def e8m0(X) -> np.ndarray:
    e = np.log2(np.asarray(X, np.float64))
    assert np.array_equal(e, np.round(e)), "scales are not powers of two"
    e = np.round(e).astype(np.int64) + 127
    assert e.min() >= 0 and e.max() <= 254
    return e.astype(np.uint8)


def codebooks(P_lut: torch.Tensor, G: int):
    """K x N LUT values -> (tables [NKB][N>>G][16], idx [K][N]): a table = its region's distinct values, sorted,
    padded with its last value (16 centroids; fewer distinct codes only when centroids collide on the E4M3 grid)."""
    K, N = P_lut.shape
    w = 1 << G
    tabs = torch.zeros(K // GROUP, N // w, 16)
    idx = torch.zeros(K, N, dtype=torch.int64)
    for kb in range(K // GROUP):
        for t in range(N // w):
            reg = P_lut[kb * GROUP:(kb + 1) * GROUP, t * w:(t + 1) * w].contiguous()
            u = torch.unique(reg)
            assert u.numel() <= 16, f"K-block {kb} table {t}: {u.numel()} distinct values"
            tabs[kb, t] = torch.cat([u, u[-1:].expand(16 - u.numel())])
            idx[kb * GROUP:(kb + 1) * GROUP, t * w:(t + 1) * w] = torch.searchsorted(u, reg)
    back = torch.gather(tabs[torch.arange(K) // GROUP][:, torch.arange(N) // w], 2, idx.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(back, P_lut), "codebook + index does not reproduce the LUT values"
    return tabs, idx


# ---------------------------------------------------------------------------------------------------- export
def run_file(mxq, G, seed):
    return Path(mxq) / "runs" / (f"tinyllama_var_u32x0_tkblock_w_s{seed}.jsonl" if G == 1
                                 else f"tinyllama_var_u32x0_tkblock_w_g{G}_s{seed}.jsonl")


def study_masks(path, layer, G, budgets):
    """{"b5": [0/1 per K-block of the whole layer], ...} from the run's rows (label 'tile 32xN w wT32x<2^G> <b>')."""
    want = {f"tile 32xN w wT32x{1 << G} {b:.1f}": f"b{b:g}" for b in budgets}
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["label"] in want:
            m = r["masks"][layer]
            assert all(len(row) == 1 for row in m), "expected 32xN usage masks ([K/32][1])"
            out[want[r["label"]]] = [int(row[0]) for row in m]
    miss = set(want.values()) - set(out)
    assert not miss, f"{path}: no rows for {sorted(miss)}"
    return {k: out[k] for k in want.values()}


@torch.no_grad()
def layer_input(model_path, layer, seqlen):
    """The layer's input on WikiText-2 calibration sequence 0 (the study's loader, seed 0) [seqlen, K], its weight."""
    from transformers import AutoModelForCausalLM
    from utils.loader import get_loaders
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, use_safetensors=True).eval()
    calib, _ = get_loaders("wikitext2", 1, 0, seqlen, model_path)
    seen = {}

    class Stop(Exception):
        pass

    def hook(_m, args):
        seen["x"] = args[0].detach().reshape(-1, args[0].shape[-1]).float()
        raise Stop
    h = model.get_submodule(layer).register_forward_pre_hook(hook)
    try:
        model(input_ids=calib[0][0], use_cache=False)
    except Stop:
        pass
    finally:
        h.remove()
    return seen["x"], model.get_submodule(layer).weight.detach().float().clone()


def export(a) -> Path:
    os.environ.setdefault("MXQ_TILE_BASE", "MXFP8_E4M3")
    sys.path.insert(0, a.mxquant)
    os.chdir(a.mxquant)                                          # the loader's dataset cache paths are relative
    from mxq import block
    from quant import FAST, configure_lut_backend, loss_weight, quantize_px
    from tile_lut import KB, TileConfig, lut_weight_codes
    assert KB == GROUP
    configure_lut_backend("abhiKmeans")                          # main.py --lut_backend default
    quantizer = block.mxgemmini                                  # main.py --quantizer gemmini (the study's runs)
    BASE = "MXFP8_E4M3"
    G, w = a.g, 1 << a.g
    budgets = [float(b) for b in a.budgets.split(",") if b]
    masks_full = study_masks(a.run or run_file(a.mxquant, G, a.seed), a.layer, G, budgets)

    x, W = layer_input(a.model, a.layer, a.seqlen)
    Kf, Nf = W.shape[1], W.shape[0]
    K, M, N = a.k or Kf, a.m, a.n
    assert a.k0 % GROUP == 0 and K % GROUP == 0 and a.k0 + K <= Kf, "K slice: whole 32-channel K-blocks"
    assert a.n0 % w == 0 and N % w == 0 and N % 64 == 0 and a.n0 + N <= Nf, "N slice: whole codebook groups, N % 64"
    assert M % 32 == 0 and a.tok0 + M <= x.shape[0]
    NKB, kb0 = K // GROUP, a.k0 // GROUP
    ks, ns = slice(a.k0, a.k0 + K), slice(a.n0, a.n0 + N)

    xs = x[a.tok0:a.tok0 + M, ks].contiguous()
    P_A, X_A = quantize_px(xs.t().contiguous(), BASE, axis=0, q=quantizer)          # [K][M], [NKB][M]
    # B = W^T (K x N, blocks down K). A table covers 32 K-rows x 2^G columns inside the slice, so fitting on the
    # slice gives the full-layer fit (up to the k-means RNG stream).
    B = W.t()[ks, ns].contiguous()
    g2 = torch.load(a.grads, weights_only=False)[a.layer]["w_g2"].float().t()[ks, ns].contiguous()
    P, X = quantize_px(B, BASE, axis=0, q=quantizer)
    cfg = TileConfig(tile_k=32, tile_n=0, operands="w", w_table_k=32, w_table_n=w)
    torch.manual_seed(a.seed)
    P_lut, X_lut = lut_weight_codes(B, cfg, quantizer, loss_weight(g2))
    assert torch.equal(X, X_lut), "LUT fit changed the scales"
    tabs, idx = codebooks(P_lut, G)

    masks = {k: v[kb0:kb0 + NKB] for k, v in masks_full.items()}
    masks["all"] = [1] * NKB
    ref = {"A_fp": P_A.t().numpy(), "As": X_A.t().numpy(), "B8_fp": P.numpy(), "Bs": X.numpy(),
           "BL_fp": P_lut.numpy(), "tables": tabs.numpy(), "idx": idx.numpy().astype(np.uint8),
           "x_real": xs.numpy(), "w_real": B.numpy(), "y_real": (xs.double() @ B.double()).float().numpy(),
           "y_fp8": FAST(P_A, X_A, P, X).float().numpy()}
    for vi, m in enumerate(masks.values()):
        rows = torch.tensor(m, dtype=torch.bool).repeat_interleave(GROUP)
        ref[f"y_study_{vi}"] = FAST(P_A, X_A, torch.where(rows[:, None], P_lut, P), X).float().numpy()
    meta = dict(layer=a.layer, tok0=a.tok0, m=M, k0=a.k0, k=K, n0=a.n0, n=N, g=G, seed=a.seed,
                variants=list(masks), masks=masks, scale_rule="gemmini (block max in [1,2))",
                lut_backend="abhiKmeans", run=str(a.run or run_file(a.mxquant, G, a.seed)), argv=sys.argv[1:])
    DATA.mkdir(exist_ok=True)
    dest = DATA / f"lutk_{a.tag}_ref.npz"
    np.savez_compressed(dest, meta=json.dumps(meta), **ref)
    print(f"exported {a.layer} slice M={M} K={K} N={N} G={G} -> {dest}")
    return dest


# ---------------------------------------------------------------------------------------------------- blob
def build(a):
    sys.path.insert(0, str(GEN))
    import gen_lutk_proj as LK                                   # golden, nib, bf16_add, fp4_grid, nearest
    path = DATA / f"lutk_{a.tag}_ref.npz"
    z = dict(np.load(path))
    meta = json.loads(str(z.pop("meta")))
    M, K, N, G = meta["m"], meta["k"], meta["n"], meta["g"]
    NKB = K // GROUP
    masks = meta["masks"]
    A_fp, As = torch.from_numpy(z["A_fp"]), torch.from_numpy(z["As"])
    B8_fp, Bs, BL_fp = torch.from_numpy(z["B8_fp"]), torch.from_numpy(z["Bs"]), torch.from_numpy(z["BL_fp"])
    g4 = LK.fp4_grid()
    w4 = LK.nearest(B8_fp, g4)
    B4_fp, B4_codes = g4[w4], w4.numpy()
    lut_codes = e4m3_codes(z["tables"])                          # [NKB][N>>G][16]

    parts = [("A", e4m3_codes(z["A_fp"])), ("AS", e8m0(z["As"].T)), ("B8", e4m3_codes(z["B8_fp"])),
             ("BL", LK.nib(z["idx"])), ("BLUT", lut_codes), ("B4", LK.nib(B4_codes)), ("BS", e8m0(z["Bs"]))]
    for vi, m in enumerate(masks.values()):
        parts.append((f"BLUTC{vi}", lut_codes[[i for i in range(NKB) if m[i]]]))
    dims = [int(d) for d in a.dims.split(",")]
    summary = {}
    for d in dims:
        parts.append((f"FP8_D{d}", LK.golden(A_fp, B8_fp, As, Bs, list(range(NKB)), d)))
    for vi, (name, m) in enumerate(masks.items()):
        lo = [i for i in range(NKB) if m[i]]
        hi = [i for i in range(NKB) if not m[i]]
        ks_lo = np.concatenate([np.arange(kb * GROUP, (kb + 1) * GROUP) for kb in lo]) if lo else np.zeros(0, int)
        for d in dims:
            p_hi = LK.golden(A_fp, B8_fp, As, Bs, hi, d)
            p_l = LK.golden(A_fp, BL_fp, As, Bs, lo, d)
            p_4 = LK.golden(A_fp, B4_fp, As, Bs, lo, d)
            bm_l, bm_4 = B8_fp.clone(), B8_fp.clone()
            bm_l[ks_lo], bm_4[ks_lo] = BL_fp[ks_lo], B4_fp[ks_lo]
            c_ml = LK.golden(A_fp, bm_l, As, Bs, list(range(NKB)), d)
            for t, arr in (("PHI", p_hi), ("PLUT", p_l), ("PFP4", p_4),
                           ("CLUT", LK.bf16_add(p_l, p_hi)), ("CFP4", LK.bf16_add(p_4, p_hi)),
                           ("CSL", LK.golden(A_fp, bm_l, As, Bs, lo + hi, d)),
                           ("CS4", LK.golden(A_fp, bm_4, As, Bs, lo + hi, d)),
                           ("CML", c_ml), ("CM4", LK.golden(A_fp, bm_4, As, Bs, list(range(NKB)), d))):
                parts.append((f"{t}{vi}_D{d}", arr))
            cml = (c_ml.astype(np.uint32) << 16).view(np.float32)
            ys = z[f"y_study_{vi}"]
            rel = lambda u, v: float(np.linalg.norm(u - v) / np.linalg.norm(v))
            summary[f"{name} d{d}"] = dict(lut_kblocks=f"{len(lo)}/{NKB}", golden_vs_study=rel(cml, ys),
                                           study_vs_fp8=rel(ys, z["y_fp8"]), study_vs_real=rel(ys, z["y_real"]))

    buf, offs = bytearray(), {}
    for n, arr in parts:
        buf += bytes((-len(buf)) % 64)
        offs[n] = len(buf)
        buf += np.ascontiguousarray(arr).tobytes()
    base = f"lutk_{a.tag}"
    (DATA / f"{base}.bin").write_bytes(bytes(buf))
    sym = f"_binary_{base}_bin_start"
    h = [f"// generated by gen/gen_lutk_tinyllama.py {' '.join(meta['argv'])}",
         f"// TinyLlama {meta['layer']}: tokens {meta['tok0']}..{meta['tok0'] + M - 1}, K {meta['k0']}..{meta['k0'] + K - 1}, "
         f"N {meta['n0']}..{meta['n0'] + N - 1}; one codebook per 32 K x {1 << G} N (G={G}); study seed {meta['seed']}",
         f"#ifndef LUTK_{a.tag.upper()}_H", f"#define LUTK_{a.tag.upper()}_H", "#include <stdint.h>",
         f"#define LK_M {M}", f"#define LK_K {K}", f"#define LK_N {N}", f"#define LK_NKB {NKB}",
         f"#define LK_DATA_G {G}   /* BLUT is [LK_NKB][LK_N >> LK_DATA_G][16]; lutk_proj.c hard-codes LK_G 1 */",
         f"#define LK_NVAR {len(masks)}", f"extern const uint8_t {sym}[];", f"#define LK_BLOB {sym}"]
    h += [f"#define LK_OFF_{n} {o}" for n, o in offs.items()]
    h.append(f"static const char *const lk_var_name[LK_NVAR] = {{{', '.join(json.dumps(k) for k in masks)}}};")
    h.append("static const uint8_t lk_mask[LK_NVAR][LK_NKB] = {")
    h += ["  {" + ", ".join(str(v) for v in m) + "}," for m in masks.values()]
    h += ["};", "#endif", ""]
    (DATA / f"{base}.h").write_text("\n".join(h))
    meta["golden_summary"] = summary
    meta["dims"] = dims
    np.savez_compressed(path, meta=json.dumps(meta), **z)
    print(f"wrote data/{base}.bin ({len(buf)} B), data/{base}.h; variants {list(masks)}")
    for k, v in summary.items():
        print(f"  {k:8s} LUT K-blocks {v['lut_kblocks']:>7s}   rel err: hw golden vs study {v['golden_vs_study']:.2e}"
              f" | study vs all-MXFP8 {v['study_vs_fp8']:.2e} | study vs unquantized {v['study_vs_real']:.2e}")


def venv_env():
    """the .venv's bin first on PATH: qtorch JIT-builds its CPU kernels with the venv's ninja"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PATH"] = f"{VENV_PY.parent}:{env.get('PATH', '')}"
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", default="model.layers.10.mlp.down_proj")
    ap.add_argument("--m", type=int, default=64, help="tokens (multiple of 32)")
    ap.add_argument("--tok0", type=int, default=16, help="first token of the sequence used (16: skips BOS)")
    ap.add_argument("--k", type=int, default=0, help="input channels (multiple of 32; 0 = all of the layer's K)")
    ap.add_argument("--k0", type=int, default=0, help="first input channel (multiple of 32)")
    ap.add_argument("--n", type=int, default=256, help="output channels (multiple of 64 and of 2^G)")
    ap.add_argument("--n0", type=int, default=0, help="first output channel (multiple of 2^G)")
    ap.add_argument("--g", type=int, default=1, help="2^G output columns per codebook (study ran 1, 3, 5, 7, 8)")
    ap.add_argument("--budgets", default="5,6,7", help="study budgets whose masks become variants (+ 'all')")
    ap.add_argument("--seed", type=int, default=0, help="study seed: LUT fit RNG and the run file's masks")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--mxquant", default=MXQ)
    ap.add_argument("--model", default="/scratch/nicorakela/models/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--grads", default="/scratch/nicorakela/mxquant_grads/grads_TinyLlama-1.1B-Chat-v1.0_wikitext216x2048s0.pt")
    ap.add_argument("--run", default=None, help="study jsonl with the masks (default runs/tinyllama_var_u32x0_tkblock_w[_gG]_s<seed>.jsonl)")
    ap.add_argument("--dims", default="16,32", help="mesh DIMs to write goldens for")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--build", action="store_true", help="only build blob + goldens from data/lutk_<tag>_ref.npz (.venv python)")
    a = ap.parse_args()
    if not a.build:
        export(a)
        return subprocess.call([str(VENV_PY), os.path.abspath(__file__), "--build", "--tag", a.tag, "--dims", a.dims],
                               env=venv_env())
    build(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
