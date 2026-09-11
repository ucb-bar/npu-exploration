"""Capture ONE real TinyLlama decoder layer -- activations, weights, RoPE tables, references.

`capture_llama_tiles.py` logs a 512x512 (A, W) pair per projection, each at a *random* and
unrecorded (token, in-feature, out-feature) offset (`log_pairs_from_eval.py:183-195`). That is
enough for a standalone matmul and useless for a layer: `gate_proj`'s out-features and
`down_proj`'s in-features are different random windows, so `down(silu(gate) * up)` cannot be
composed from them, and `k_proj`/`v_proj` are skipped entirely (out_features 256 < 512).

This captures a real decoder layer instead, from one real forward pass, everything index-consistent:

    h_pre      [S][D]     the residual stream entering the layer      -> attention's input
    h_mid      [S][D]     the residual stream after attention         -> the MLP's input
    h_out      [S][D]     the layer's true output (context, not a grading target)
    w_in_ln    [D]        input_layernorm weight       (RMSNorm is over the FULL D -> exact)
    w_post_ln  [D]        post_attention_layernorm weight
    Wq,Wk,Wv   [D][H]     one query head and its GQA kv head, as [in][out]
    Wo         [H][D]     o_proj columns for that head
    Wg,Wu      [D][F]     gate_proj/up_proj rows for neurons [n0, n0+F)
    Wd         [F][D]     down_proj columns for those same neurons
    rope_cos   [S][H]     the model's own rotary tables for these positions
    rope_sin   [S][H]

D (hidden size) is kept FULL, so RMSNorm, the residual and every projection *input* are exact; the
slicing is on the OUTPUT side only. Consequences, stated where they are made rather than discovered
later: `Wq/Wk/Wv/Wg/Wu` produce exact real llama values, while `Wo`/`Wd` are honest PARTIAL SUMS --
over one of `heads` heads, and over F of `intermediate` neurons. The fp32 references saved here are
truncated the same way, so they grade what the device actually computes:

    ref_mlp    [S][D]     (silu(xn@Wg) * (xn@Wu)) @ Wd,   xn = rmsnorm(h_mid, w_post_ln)
    ref_attn   [S][D]     that head's attention output through its slice of o_proj

References are computed in fp32 numpy from the captured tensors -- the same values the device is
given -- not by re-running torch in a different precision.

Run it with the npu-exploration venv:

    .venv/bin/python3 -m app.capture_llama_layer                     # layer 5, head 0, neurons 0..63
    .venv/bin/python3 -m app.capture_llama_layer --layer 10 --nf 128
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

MXQ_ROOT = Path(__file__).resolve().parent.parent / "MXQuant"
#: Captures land in THIS repo, not in the MXQuant checkout -- MXQuant is upstream and stays clean.
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "out" / "layer_capture"

#: `mxquant/datautils.py:10` asks for the bare dataset id `wikitext`, which datasets>=4 rejects.
#: Same corpus, canonical id -- identical tokenization to capture_llama_tiles.py.
WIKITEXT2 = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- fp32 model of the layer, over the captured slice ---------------------------------------------

def _rope_theta(cfg) -> float:
    """RoPE base. transformers 5.x moved it into `rope_parameters`; older configs have the field."""
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    return float(getattr(cfg, "rope_theta", 10000.0))


def rmsnorm(h: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Llama RMSNorm, fp32: h / sqrt(mean(h^2) + eps) * w. Exact -- the norm is over the full D."""
    h = h.astype(np.float32)
    inv = 1.0 / np.sqrt((h * h).mean(axis=-1, keepdims=True) + eps)
    return (h * inv * w.astype(np.float32)).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """HF llama rotary: x*cos + rotate_half(x)*sin, halves of the head dim."""
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[:, half:], x[:, :half]], axis=-1)
    return (x * cos + rot * sin).astype(np.float32)


def softmax_causal(s: np.ndarray) -> np.ndarray:
    """Row softmax with a causal mask over a square score matrix, fp32."""
    S = s.shape[0]
    mask = np.triu(np.ones((S, S), dtype=bool), k=1)
    s = np.where(mask, -np.inf, s.astype(np.float32))
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def reference_mlp(t: dict, eps: float) -> np.ndarray:
    """The sliced MLP, fp32: exactly what llama_mlp.c computes, in full precision."""
    xn = rmsnorm(t["h_mid"], t["w_post_ln"], eps)
    h = silu(xn @ t["Wg"]) * (xn @ t["Wu"])
    return (h @ t["Wd"]).astype(np.float32)


def reference_attn(t: dict, eps: float) -> np.ndarray:
    """The sliced single-head attention, fp32: exactly what llama_attention.c computes."""
    xn = rmsnorm(t["h_pre"], t["w_in_ln"], eps)
    q = rope(xn @ t["Wq"], t["rope_cos"], t["rope_sin"])
    k = rope(xn @ t["Wk"], t["rope_cos"], t["rope_sin"])
    v = (xn @ t["Wv"]).astype(np.float32)
    s = (q @ k.T) / np.sqrt(np.float32(q.shape[-1]))
    return (softmax_causal(s) @ v @ t["Wo"]).astype(np.float32)


# --- capture --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--layer", type=int, default=5, help="decoder layer index")
    ap.add_argument("--seq", type=int, default=32, help="tokens (also the attention context)")
    ap.add_argument("--tok0", type=int, default=0, help="first token of the wikitext2 test split")
    ap.add_argument("--head", type=int, default=0, help="query head; its GQA kv head is derived")
    ap.add_argument("--neuron0", type=int, default=0, help="first FFN neuron of the slice")
    ap.add_argument("--nf", type=int, default=64, help="FFN neurons (multiple of 32)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.nf % 32:
        raise SystemExit(f"--nf {args.nf} must be a multiple of 32 (the E8M0 block)")
    if args.seq % 16:
        raise SystemExit(f"--seq {args.seq} must be a multiple of 16 (the PE tile)")

    import torch
    if str(MXQ_ROOT) not in sys.path:
        sys.path.insert(0, str(MXQ_ROOT))
    esq = _load("eval_simquant", MXQ_ROOT / "eval_simquant.py")

    print(f"[load] {args.model_id} via eval_simquant.get_model (bf16, as MXQuant runs it)")
    model = esq.get_model(args.model_id, args.seq, args.seq, gpu=0)
    model.eval()
    cfg = model.config
    D, H = cfg.hidden_size, cfg.hidden_size // cfg.num_attention_heads
    kv_head = args.head // (cfg.num_attention_heads // cfg.num_key_value_heads)
    print(f"[model] D={D} heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
          f"head_dim={H} intermediate={cfg.intermediate_size} layers={cfg.num_hidden_layers}")
    print(f"[slice] layer={args.layer} head={args.head} (kv head {kv_head})  "
          f"neurons [{args.neuron0}, {args.neuron0 + args.nf})  seq={args.seq} from token {args.tok0}")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    testdata = load_dataset(*WIKITEXT2, split="test")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    enc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    ids = enc.input_ids[:, args.tok0:args.tok0 + args.seq]
    if ids.shape[1] < args.seq:
        raise SystemExit(f"only {ids.shape[1]} tokens available at --tok0 {args.tok0}")

    layer = model.model.layers[args.layer]
    grab: dict[str, torch.Tensor] = {}

    def pre(name):
        def hook(_mod, a):
            grab.setdefault(name, a[0].detach()[0].float().cpu())
        return hook

    def pre_kw(name):
        def hook(_mod, a, kw):
            pe = kw.get("position_embeddings")
            if pe is None and len(a) >= 2 and isinstance(a[1], tuple):
                pe = a[1]
            if pe is not None:
                grab.setdefault(name + "_cos", pe[0].detach()[0].float().cpu())
                grab.setdefault(name + "_sin", pe[1].detach()[0].float().cpu())
        return hook

    def post(name):
        def hook(_mod, _a, out):
            o = out[0] if isinstance(out, tuple) else out
            grab.setdefault(name, o.detach()[0].float().cpu())
        return hook

    handles = [
        layer.register_forward_pre_hook(pre("h_pre")),
        layer.post_attention_layernorm.register_forward_pre_hook(pre("h_mid")),
        layer.register_forward_hook(post("h_out")),
        layer.self_attn.register_forward_pre_hook(pre_kw("rope"), with_kwargs=True),
        layer.input_layernorm.register_forward_hook(post("xn_attn_torch")),
        layer.post_attention_layernorm.register_forward_hook(post("xn_mlp_torch")),
        layer.mlp.gate_proj.register_forward_hook(post("gate_torch")),
    ]
    print(f"[run] one forward pass, input_ids {tuple(ids.shape)}")
    with torch.no_grad():
        model(input_ids=ids.to(model.device), use_cache=False, return_dict=True)
    for h in handles:
        h.remove()

    missing = [k for k in ("h_pre", "h_mid", "h_out") if k not in grab]
    if missing:
        raise SystemExit(f"hooks did not fire for {missing}; transformers "
                         f"{__import__('transformers').__version__} layer API may have moved")

    def w(mod) -> np.ndarray:
        return mod.weight.detach().float().cpu().numpy()

    q0, n0, nf = args.head * H, args.neuron0, args.nf
    k0 = kv_head * H
    sa, mlp = layer.self_attn, layer.mlp
    t = {
        "h_pre": grab["h_pre"].numpy(),
        "h_mid": grab["h_mid"].numpy(),
        "h_out": grab["h_out"].numpy(),
        "w_in_ln": w(layer.input_layernorm),
        "w_post_ln": w(layer.post_attention_layernorm),
        "Wq": np.ascontiguousarray(w(sa.q_proj)[q0:q0 + H, :].T),      # [D][H]
        "Wk": np.ascontiguousarray(w(sa.k_proj)[k0:k0 + H, :].T),      # [D][H]
        "Wv": np.ascontiguousarray(w(sa.v_proj)[k0:k0 + H, :].T),      # [D][H]
        "Wo": np.ascontiguousarray(w(sa.o_proj)[:, q0:q0 + H].T),      # [H][D]
        "Wg": np.ascontiguousarray(w(mlp.gate_proj)[n0:n0 + nf, :].T),  # [D][F]
        "Wu": np.ascontiguousarray(w(mlp.up_proj)[n0:n0 + nf, :].T),    # [D][F]
        "Wd": np.ascontiguousarray(w(mlp.down_proj)[:, n0:n0 + nf].T),  # [F][D]
    }
    if "rope_cos" in grab:
        t["rope_cos"], t["rope_sin"] = grab["rope_cos"].numpy(), grab["rope_sin"].numpy()
        src = "model's own position_embeddings"
    else:                                    # older layer API: recompute HF's tables ourselves
        inv = 1.0 / (_rope_theta(cfg) ** (np.arange(0, H, 2, dtype=np.float64) / H))
        f = np.outer(np.arange(args.seq, dtype=np.float64), inv)
        emb = np.concatenate([f, f], axis=-1)
        t["rope_cos"], t["rope_sin"] = np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)
        src = "recomputed (hook saw no position_embeddings)"
    print(f"[rope] cos/sin {t['rope_cos'].shape} from the {src}")

    eps = float(cfg.rms_norm_eps)
    t["ref_mlp"] = reference_mlp(t, eps)
    t["ref_attn"] = reference_attn(t, eps)

    # ---- gates: the slicing and the transposes, which fail silently with plausible numbers ----
    xn = rmsnorm(t["h_mid"], t["w_post_ln"], eps)
    xn_torch = grab["xn_mlp_torch"].numpy()
    d_norm = float(np.abs(xn - xn_torch).max() / max(np.abs(xn_torch).max(), 1e-30))
    g_ours = xn @ t["Wg"]
    g_torch = grab["gate_torch"].numpy()[:, n0:n0 + nf]
    d_gate = float(np.abs(g_ours - g_torch).max() / max(np.abs(g_torch).max(), 1e-30))
    print(f"[check] rmsnorm vs torch post_attention_layernorm: rel max|d| = {d_norm:.3e}")
    print(f"[check] xn @ Wg vs torch gate_proj[:, slice]:       rel max|d| = {d_gate:.3e}  "
          f"(bf16 forward vs fp32 here, so ~1e-2 is expected and ~1e-6 is not required)")
    if d_gate > 0.05:
        raise SystemExit("gate_proj slice does not reproduce the model's own output -- the weight "
                         "slicing or the [out][in] -> [in][out] transpose is wrong")

    meta = dict(model_id=args.model_id, layer=args.layer, seq=args.seq, tok0=args.tok0,
                head=args.head, kv_head=kv_head, neuron0=n0, nf=nf, d_model=D, head_dim=H,
                n_heads=cfg.num_attention_heads, n_kv_heads=cfg.num_key_value_heads,
                intermediate=cfg.intermediate_size, rms_eps=eps, rope_theta=_rope_theta(cfg),
                token_ids=ids[0].cpu().numpy())

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / (f"layer{args.layer}_h{args.head}_n{n0}-{n0 + nf}"
                       f"_s{args.seq}t{args.tok0}.npz")
    np.savez(path, **{k: v.astype(np.float32) for k, v in t.items()},
             **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    print(f"\n[done] {path}")
    for k in ("h_pre", "h_mid", "Wq", "Wo", "Wg", "Wd", "ref_mlp", "ref_attn"):
        print(f"  {k:9s} {str(t[k].shape):12s} |max| = {np.abs(t[k]).max():.6g}")
    print(f"  text: {tokenizer.decode(ids[0][:16])!r} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
