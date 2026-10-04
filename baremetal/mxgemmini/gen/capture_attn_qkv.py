#!/usr/bin/env python3
"""Real TinyLlama Q/K/V for one attention head over a long context, for gen_attn_vpu.py --capture.

One forward pass of TinyLlama-1.1B-Chat over the first --seq tokens of the wikitext-2 test split (local HF cache,
offline, CPU). Layer --layer's own input RMSNorm, q/k/v projections and rotary embedding give head --head's Q, K
and V (GQA kv head = head // 8) at their true positions, saved as fp32 [seq][64] arrays.

    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ../../../.venv/bin/python3 capture_attn_qkv.py --seq 2112
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "out" / "layer_capture"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--seq", type=int, default=2112)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--head", type=int, default=0)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.model_id)
    model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.float32)
    model.eval()
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    kvh = a.head // (cfg.num_attention_heads // cfg.num_key_value_heads)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[:, :a.seq]
    assert ids.shape[1] == a.seq
    print(f"model {a.model_id}: layer {a.layer}, head {a.head} (kv head {kvh}), head_dim {hd}, {a.seq} tokens")
    out = model(ids, output_hidden_states=True)
    h = out.hidden_states[a.layer]                       # input to decoder layer a.layer
    layer = model.model.layers[a.layer]
    hn = layer.input_layernorm(h)
    T = a.seq
    q = layer.self_attn.q_proj(hn).view(1, T, cfg.num_attention_heads, hd).transpose(1, 2)
    k = layer.self_attn.k_proj(hn).view(1, T, cfg.num_key_value_heads, hd).transpose(1, 2)
    v = layer.self_attn.v_proj(hn).view(1, T, cfg.num_key_value_heads, hd).transpose(1, 2)
    pos = torch.arange(T)[None]
    cos, sin = model.model.rotary_emb(v, pos)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    Q, K, V = (x[0, i].numpy().astype(np.float32) for x, i in ((q, a.head), (k, kvh), (v, kvh)))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"attn_qkv_layer{a.layer}_h{a.head}_s{T}.npz"
    np.savez(path, Q=Q, K=K, V=V, layer=a.layer, head=a.head, kv_head=kvh, seq=T)
    print(f"  |Q| {np.abs(Q).max():.3g}  |K| {np.abs(K).max():.3g}  |V| {np.abs(V).max():.3g}  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
