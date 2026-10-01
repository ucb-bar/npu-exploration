"""Capture the WHOLE TinyLlama model -- every decoder layer, the head, and the true logits.

`kernels/captures/llama_layer.py` captures ONE layer, which is what the sub-layer and single-layer kernels
needed. A stacked-model kernel needs every layer's weights plus the pieces around them:

    out/model_capture/layer<N>.npz   per layer: the same tensors kernels/captures/llama_layer.py emits with
                                     --all-heads --all-neurons (nothing sliced), plus that layer's
                                     own h_pre / h_out from the real forward pass
    out/model_capture/model.npz      embed_out (layer 0's input), the final RMSNorm weight, the
                                     lm_head weight, the true logits, and the token ids

ONE FILE PER LAYER, deliberately. The weights are ~176 MB per layer as fp32, so a single npz would
be ~4 GB and could not be loaded incrementally; the generator processes one layer at a time and the
per-layer files are what make that possible. They are also what lets a 2-layer kernel be built and
tested before committing to 22.

WHY THE PER-LAYER h_pre/h_out ARE STILL CAPTURED even though a stacked kernel chains its own: they
are the only way to localize a divergence. The kernel feeds layer N the value the DEVICE produced,
so a late-layer mismatch says nothing about where it started; comparing each layer's output against
the model's own says exactly which layer first departs, and by how much it had already drifted.

    .venv/bin/python3 -m kernels.captures.llama_model                 # all 22 layers
    .venv/bin/python3 -m kernels.captures.llama_model --layers 0-1    # just the first two
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

MXQ_ROOT = Path(__file__).resolve().parents[2] / "MXQuant"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "out" / "model_capture"
WIKITEXT2 = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_layers(s: str, n: int) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise SystemExit(f"--layers names {bad}, outside 0..{n - 1}")
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--seq", type=int, default=32, help="tokens (also the attention context)")
    ap.add_argument("--tok0", type=int, default=0, help="first token of the wikitext2 test split")
    ap.add_argument("--layers", default=None,
                    help="layers to capture, e.g. '0-3' or '0,5,21'. Default: all of them.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.seq % 16:
        raise SystemExit(f"--seq {args.seq} must be a multiple of 16 (the PE tile)")

    import torch
    if str(MXQ_ROOT) not in sys.path:
        sys.path.insert(0, str(MXQ_ROOT))
    cll = _load("llama_layer", Path(__file__).resolve().parent / "llama_layer.py")

    if (MXQ_ROOT / "eval_simquant.py").exists():
        esq = _load("eval_simquant", MXQ_ROOT / "eval_simquant.py")
        print(f"[load] {args.model_id} via eval_simquant.get_model (bf16, as MXQuant runs it)")
        model = esq.get_model(args.model_id, args.seq, args.seq, gpu=0)
    else:   # bf16 from_pretrained, equivalent for seq <= 2048
        from transformers import AutoModelForCausalLM
        print(f"[load] {args.model_id} via transformers.from_pretrained (bf16; MXQuant not present)")
        model = AutoModelForCausalLM.from_pretrained(args.model_id, use_safetensors=True,
                                                     torch_dtype=torch.bfloat16)
    model.eval()
    cfg = model.config
    D = cfg.hidden_size
    H = D // cfg.num_attention_heads
    NL = cfg.num_hidden_layers
    layers = _parse_layers(args.layers, NL) if args.layers else list(range(NL))
    print(f"[model] D={D} heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
          f"head_dim={H} intermediate={cfg.intermediate_size} layers={NL} vocab={cfg.vocab_size}")
    print(f"[capture] layers {layers[0]}..{layers[-1]} ({len(layers)} of {NL}), seq={args.seq}")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    testdata = load_dataset(*WIKITEXT2, split="test")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    enc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    # One extra token: the last position's label is the token AFTER the window, which is what a
    # perplexity over all `seq` positions needs.
    ids_all = enc.input_ids[:, args.tok0:args.tok0 + args.seq + 1]
    if ids_all.shape[1] < args.seq + 1:
        raise SystemExit(f"only {ids_all.shape[1]} tokens available at --tok0 {args.tok0}")
    ids = ids_all[:, :args.seq]

    grab: dict[str, "torch.Tensor"] = {}

    def pre(name):
        def hook(_mod, a):
            grab.setdefault(name, a[0].detach()[0].float().cpu())
        return hook

    def post(name):
        def hook(_mod, _a, out):
            o = out[0] if isinstance(out, tuple) else out
            grab.setdefault(name, o.detach()[0].float().cpu())
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

    handles = []
    for i in layers:
        L = model.model.layers[i]
        handles += [L.register_forward_pre_hook(pre(f"h_pre{i}")),
                    L.register_forward_hook(post(f"h_out{i}")),
                    L.self_attn.register_forward_hook(post(f"attn{i}")),
                    L.mlp.register_forward_hook(post(f"mlp{i}"))]
    handles.append(model.model.layers[layers[0]].self_attn.register_forward_pre_hook(
        pre_kw("rope"), with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(pre("h_final")))

    print(f"[run] one forward pass, input_ids {tuple(ids.shape)}")
    with torch.no_grad():
        out = model(input_ids=ids.to(model.device), use_cache=False, return_dict=True)
    logits = out.logits.detach()[0].float().cpu().numpy()
    for h in handles:
        h.remove()

    def w(mod) -> np.ndarray:
        return mod.weight.detach().float().cpu().numpy()

    args.out.mkdir(parents=True, exist_ok=True)
    eps = float(cfg.rms_norm_eps)
    meta_common = dict(model_id=args.model_id, seq=args.seq, tok0=args.tok0,
                       d_model=D, head_dim=H, all_heads=1, all_neurons=1,
                       head=0, kv_head=0, neuron0=0, nf=cfg.intermediate_size,
                       n_heads=cfg.num_attention_heads, n_kv_heads=cfg.num_key_value_heads,
                       intermediate=cfg.intermediate_size, rms_eps=eps,
                       rope_theta=cll._rope_theta(cfg), n_layers=NL,
                       vocab=cfg.vocab_size, token_ids=ids[0].cpu().numpy())

    rope_cos = grab["rope_cos"].numpy()
    rope_sin = grab["rope_sin"].numpy()
    for i in layers:
        L = model.model.layers[i]
        sa, mlp = L.self_attn, L.mlp
        t = {
            "h_pre": grab[f"h_pre{i}"].numpy(),
            "h_out": grab[f"h_out{i}"].numpy(),
            "attn_torch": grab[f"attn{i}"].numpy(),
            "mlp_torch": grab[f"mlp{i}"].numpy(),
            "w_in_ln": w(L.input_layernorm),
            "w_post_ln": w(L.post_attention_layernorm),
            "Wq": np.ascontiguousarray(w(sa.q_proj).T),
            "Wk": np.ascontiguousarray(w(sa.k_proj).T),
            "Wv": np.ascontiguousarray(w(sa.v_proj).T),
            "Wo": np.ascontiguousarray(w(sa.o_proj).T),
            "Wg": np.ascontiguousarray(w(mlp.gate_proj).T),
            "Wu": np.ascontiguousarray(w(mlp.up_proj).T),
            "Wd": np.ascontiguousarray(w(mlp.down_proj).T),
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
        }
        # h_mid is what the MLP half sees: the residual after attention. The layer kernel derives
        # it itself, but the reference is needed to report each layer's own drift.
        t["h_mid"] = t["h_pre"] + t["attn_torch"]
        meta = dict(meta_common, layer=i)
        p = args.out / f"layer{i}.npz"
        np.savez(p, **{k: v.astype(np.float32) for k, v in t.items()},
                 **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})
        print(f"  layer {i:2d} -> {p.name}  ({p.stat().st_size / 1e6:.0f} MB)  "
              f"|h_out| = {np.abs(t['h_out']).max():.4g}")

    mp = args.out / "model.npz"
    np.savez(mp,
             embed_out=grab[f"h_pre{layers[0]}"].numpy().astype(np.float32),
             h_final=grab["h_final"].numpy().astype(np.float32),
             w_final_ln=w(model.model.norm).astype(np.float32),
             lm_head=np.ascontiguousarray(w(model.lm_head).T).astype(np.float32),
             logits=logits.astype(np.float32),
             labels=ids_all[0, 1:].cpu().numpy(),
             **{f"meta_{k}": np.asarray(v) for k, v in meta_common.items()})
    print(f"  head    -> {mp.name}  ({mp.stat().st_size / 1e6:.0f} MB)")

    # The reference perplexity, from the model's own logits: the number the stacked kernel is
    # ultimately trying to reproduce.
    lab = ids_all[0, 1:].cpu().numpy()
    z = logits - logits.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=-1))
    nll = float(np.mean(lse - z[np.arange(len(lab)), lab]))
    print(f"\n[done] {args.out}")
    print(f"  tokens : {tokenizer.decode(ids[0][:16])!r} ...")
    print(f"  torch logits {logits.shape}  |max| = {np.abs(logits).max():.4g}")
    print(f"  REFERENCE nll = {nll:.6f}  ppl = {np.exp(nll):.4f}  (fp32, from the model's logits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
