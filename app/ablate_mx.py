"""Is the 4-point perplexity drop the MX FORMAT, or this datapath's ARITHMETIC?

The stacked-model kernel lands at ppl 32.86 against the bf16 model's 28.51 on its 32-token window.
That is ~15%, and it is far worse than published MXFP8 inference results, which are typically
near-lossless. Those results assume **fp32 accumulation**; this hardware does not do that:

    PROD_PRECISION = [(4, 3)] * 16                                       every product -> e4m3
    ACC_PRECISION  = [(4, 4)]*8 + [(4, 5)]*2 + [(4, 6)]*5 + [(8, 7)]*1   (exp, frac) per lane

EIGHT of sixteen accumulator lanes carry 4 exponent and 4 mantissa bits; exactly one is bf16. So
"MXFP8" here means the format AND a very narrow accumulator, and the question is which one costs
the perplexity. `llama_layer_hw_plan.md` 8.2 splits it for ONE sub-layer as a tensor error (format
6.57%, full datapath 11.59% on the MLP) -- this converts that into perplexity, end to end, which is
the number that actually decides whether the format or the hardware needs changing.

Three arithmetic models over the SAME 22-layer chain, the same captures, the same fp32 host glue:

    fp32      no quantization at all -- sanity, should sit on top of the bf16 reference
    mx-exact  operands quantized to MXFP8 (E4M3 codes + E8M0 per-32 block scales), then an EXACT
              fp32 matmul. This is MXFP8 as the literature means it: the format, perfect accumulate.
    mx-rtl    the full datapath -- what the kernel computes. Already measured at 32.863 by
              gen_llama_model.py, so it is NOT recomputed here (it costs ~36 min); the value is
              quoted for comparison.

`mx-exact` is cheap: a dequantize plus a numpy matmul, no per-lane simulation, so the whole 22-layer
chain runs in seconds rather than the datapath model's half hour.

    .venv/bin/python3 -m app.ablate_mx
    .venv/bin/python3 -m app.ablate_mx --layers 4        # quick check on a prefix
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPU = HERE.parent
ROCC = NPU.parent / "software" / "gemmini-rocc-tests"
GEN = NPU / "baremetal" / "mxgemmini" / "gen"
for p in (str(NPU), str(ROCC), str(GEN)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gen_matmul_llama as G          # noqa: E402
import gen_llama_layer as GL          # noqa: E402
from kernels.captures.llama_layer import rmsnorm, rope, softmax_causal, silu  # noqa: E402
from compiler.wire import e8m0_decode    # noqa: E402

FMT = GL.FMT
BLOCK = 32
CAPTURE = NPU / "out" / "model_capture"

#: The measured datapath result, from gen_llama_model.py / the spike kernel. Not recomputed.
MX_RTL_PPL, MX_RTL_NLL = 32.8657, 3.492431


def _deq_row(P: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Operand A: `quantize(axis='row')` gives scales [M][K/32]; value = code_value * 2^(s-127)."""
    return (P * np.repeat(e8m0_decode(scales).astype(np.float32), BLOCK, axis=1)).astype(np.float32)


def _deq_col(P: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Operand B: `quantize(axis='col')` gives scales [K/32][N]."""
    return (P * np.repeat(e8m0_decode(scales).astype(np.float32), BLOCK, axis=0)).astype(np.float32)


def mm_fp32(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """No quantization anywhere."""
    return (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)


def mm_mx_exact(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """MXFP8 operands, EXACT accumulation -- the format's cost with a perfect datapath."""
    _, a_s, a_P = G.quantize(A, axis="row", f=FMT)
    _, b_s, b_P = G.quantize(B, axis="col", f=FMT)
    return mm_fp32(_deq_row(a_P, a_s), _deq_col(b_P, b_s))


def layer(cap: dict, x: np.ndarray, mm) -> np.ndarray:
    """One decoder layer with `mm` as the matmul. Host glue is fp32 and identical across models."""
    eps = float(cap["meta_rms_eps"])
    H = int(cap["meta_head_dim"])
    NH = int(cap["meta_n_heads"])
    NKV = int(cap["meta_n_kv_heads"])
    PER = NH // NKV
    cos, sin = cap["rope_cos"], cap["rope_sin"]

    xn = rmsnorm(x, cap["w_in_ln"], eps)
    Q, K, V = mm(xn, cap["Wq"]), mm(xn, cap["Wk"]), mm(xn, cap["Wv"])
    O = np.empty_like(Q)
    for h in range(NH):
        kv = h // PER
        qh = rope(Q[:, h * H:(h + 1) * H], cos, sin)
        kh = rope(K[:, kv * H:(kv + 1) * H], cos, sin)
        vh = V[:, kv * H:(kv + 1) * H]
        S = mm(qh, np.ascontiguousarray(kh.T)) / np.sqrt(np.float32(H))
        O[:, h * H:(h + 1) * H] = mm(softmax_causal(S), np.ascontiguousarray(vh))
    x = x + mm(O, cap["Wo"])

    xn = rmsnorm(x, cap["w_post_ln"], eps)
    h_act = silu(mm(xn, cap["Wg"])) * mm(xn, cap["Wu"])
    return (x + mm(h_act, cap["Wd"])).astype(np.float32)


def run(nl: int, mm, mdl: dict, label: str) -> tuple[float, float, int, float]:
    x = mdl["embed_out"].astype(np.float32)
    for n in range(nl):
        with np.load(CAPTURE / f"layer{n}.npz") as z:
            cap = {k: z[k] for k in z.files}
        x = layer(cap, x, mm)
        print(f"  [{label}] layer {n:2d}  |x| = {np.abs(x).max():.4g}", flush=True)
    eps = float(cap["meta_rms_eps"])
    logits = mm(rmsnorm(x, mdl["w_final_ln"], eps), mdl["lm_head"])

    lab = mdl["labels"].astype(np.int64)
    z_ = logits - logits.max(axis=-1, keepdims=True)
    nll = float(np.mean(np.log(np.exp(z_).sum(-1)) - z_[np.arange(len(lab)), lab]))
    agree = int((logits.argmax(-1) == mdl["logits"].argmax(-1)).sum())
    rel = float(np.linalg.norm(logits - mdl["logits"]) / np.linalg.norm(mdl["logits"]))
    return nll, math.exp(nll), agree, rel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=0, help="0 = every captured layer")
    args = ap.parse_args()

    with np.load(CAPTURE / "model.npz") as z:
        mdl = {k: z[k] for k in z.files}
    have = sorted(int(p.stem[5:]) for p in CAPTURE.glob("layer*.npz"))
    nl = args.layers or len(have)
    lab = mdl["labels"].astype(np.int64)
    lg = mdl["logits"]
    zt = lg - lg.max(axis=-1, keepdims=True)
    nll_t = float(np.mean(np.log(np.exp(zt).sum(-1)) - zt[np.arange(len(lab)), lab]))
    M = lg.shape[0]
    print(f"ablate_mx: {nl} layers, {M} tokens\n"
          f"  reference (torch bf16 forward): nll {nll_t:.6f}  ppl {math.exp(nll_t):.4f}\n")

    rows = []
    for name, mm in (("fp32", mm_fp32), ("mx-exact", mm_mx_exact)):
        nll, ppl, agree, rel = run(nl, mm, mdl, name)
        rows.append((name, nll, ppl, agree, rel))
        print(f"  => {name:9s} nll {nll:.6f}  ppl {ppl:8.4f}  argmax {agree}/{M}  "
              f"logits rel_fro {100 * rel:.4f}%\n", flush=True)

    print(f"{'model':<12}{'nll':>10}{'ppl':>10}{'d(ppl) vs bf16':>16}")
    print(f"{'torch bf16':<12}{nll_t:>10.4f}{math.exp(nll_t):>10.4f}{'--':>16}")
    for name, nll, ppl, _, _ in rows:
        print(f"{name:<12}{nll:>10.4f}{ppl:>10.4f}{ppl - math.exp(nll_t):>+16.4f}")
    print(f"{'mx-rtl':<12}{MX_RTL_NLL:>10.4f}{MX_RTL_PPL:>10.4f}"
          f"{MX_RTL_PPL - math.exp(nll_t):>+16.4f}   <- the kernel (measured, not recomputed)")

    fmt_cost = rows[1][2] - rows[0][2]
    dp_cost = MX_RTL_PPL - rows[1][2]
    print(f"\nSPLIT  format (MXFP8 operands, exact accumulate) : {fmt_cost:+.4f} ppl")
    print(f"       datapath (prod e4m3 trunc, acc e4m4..bf16): {dp_cost:+.4f} ppl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
