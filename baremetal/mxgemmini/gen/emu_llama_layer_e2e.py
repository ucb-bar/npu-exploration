#!/usr/bin/env python3
"""MX emulation of llama_layer_e2e: every matmul through the RTL-exact mesh model (rtl_exact/mxmesh via
gen_llama_attn_full.mesh: truncated products, per-lane accumulator rounding, bf16 cross-tile), E4M3 + E8M0 wherever the
device quantizes, fp64 elsewhere (RMSNorm, RoPE, softmax, SwiGLU). The accuracy the device should land near."""
import numpy as np, torch, os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import gen_llama_layer_e2e as E
import gen_llama_attn_full as GA
G, FMT = E.G, E.FMT

def mx(x, axis):   # dequantized MX value
    c, s, P = G.quantize(np.ascontiguousarray(x.astype(np.float32)), axis=axis, f=FMT)
    sc = np.exp2(s.astype(np.float64) - 127.0)
    return P.astype(np.float64) * (np.repeat(sc, 32, axis=1) if axis == "row" else np.repeat(sc, 32, axis=0))

torch.set_grad_enabled(False)
mid = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(mid); model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32)
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[:, :2112]
out = model(ids, output_hidden_states=True)
hs_in = out.hidden_states[5][0].numpy().astype(np.float64); hs_out = out.hidden_states[6][0].numpy().astype(np.float64)
lay = model.model.layers[5]; cfg = model.config; eps = cfg.rms_norm_eps
W = {k: getattr(m, n).weight.T.numpy().astype(np.float64) for k, m, n in
     (("q", lay.self_attn, "q_proj"), ("k", lay.self_attn, "k_proj"), ("v", lay.self_attn, "v_proj"),
      ("o", lay.self_attn, "o_proj"), ("g", lay.mlp, "gate_proj"), ("u", lay.mlp, "up_proj"), ("d", lay.mlp, "down_proj"))}
Wq = {k: G.quantize(np.ascontiguousarray(v.astype(np.float32)), axis="col", f=FMT) for k, v in W.items()}
def mm(A, Bq):   # A [M][K] fp -> MX rows; Bq = (codes, scales, P) of B [K][N]; RTL-exact mesh, BF16 out
    _, sa, pa = G.quantize(np.ascontiguousarray(A.astype(np.float32)), axis="row", f=FMT)
    return GA.mesh(pa, sa, Bq[2], Bq[1]).astype(np.float64)
ln1 = lay.input_layernorm.weight.numpy().astype(np.float64); ln2 = lay.post_attention_layernorm.weight.numpy().astype(np.float64)
cos, sin = model.model.rotary_emb(torch.zeros(1, 1, 2112, 64), torch.arange(2112)[None])
cos, sin = cos[0].numpy().astype(np.float64), sin[0].numpy().astype(np.float64)
hn = E.rms(hs_in, ln1, eps)
Kall = E.rope((hn @ W["k"]).reshape(2112, 4, 64).transpose(1, 0, 2), cos[None], sin[None]); Vall = (hn @ W["v"]).reshape(2112, 4, 64).transpose(1, 0, 2)
q0 = 2048; h = hs_in[q0:].astype(np.float32).astype(np.float64)
R = E.layer_ref(h, W, ln1, ln2, eps, cos[q0:], sin[q0:], Kall[:, :q0], Vall[:, :q0], 32, 4, 64)
# MX pipeline
xn1 = E.rms(h, ln1, eps)
q = E.rope(mm(xn1, Wq["q"]).reshape(64, 32, 64).transpose(1, 0, 2), cos[q0:][None], sin[q0:][None])
kn = E.rope(mm(xn1, Wq["k"]).reshape(64, 4, 64).transpose(1, 0, 2), cos[q0:][None], sin[q0:][None]); vn = mm(xn1, Wq["v"]).reshape(64, 4, 64).transpose(1, 0, 2)
K = np.concatenate([Kall[:, :q0], kn], 1); V = np.concatenate([Vall[:, :q0], vn], 1)
o = np.zeros((32, 64, 64))
for hh in range(32):
    g = hh // 8
    kq = G.quantize(np.ascontiguousarray(K[g].T.astype(np.float32)), axis="col", f=FMT)
    vq = G.quantize(np.ascontiguousarray(V[g].astype(np.float32)), axis="col", f=FMT)
    s = mm(q[hh], kq) / 8; s = np.where(np.arange(2112)[None] > (q0 + np.arange(64))[:, None], -np.inf, s)
    p = np.exp(s - s.max(1, keepdims=True)); l = p.sum(1, keepdims=True)
    o[hh] = mm(p, vq) / l          # P un-normalized (online softmax), 1/l at the end
ya = mm(o.transpose(1, 0, 2).reshape(64, 2048), Wq["o"])
hmid = h + ya
xn2 = E.rms(hmid, ln2, eps)
gg = mm(xn2, Wq["g"]); uu = mm(xn2, Wq["u"])
ym = mm(gg / (1 + np.exp(-gg)) * uu, Wq["d"])
hout = hmid + ym
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
print(f"emu Yattn {rel(ya, R['yattn']) * 1e6:.0f} ppm, Ymlp {rel(ym, R['ymlp']) * 1e6:.0f} ppm, h_out {rel(hout, R['hout']) * 1e6:.0f} ppm, "
      f"update {rel(hout - h, R['hout'] - h) * 1e6:.0f} ppm vs fp64; update vs TinyLlama {rel(hout - h, hs_out[q0:] - h) * 1e6:.0f} ppm")
print(f"emu xn1 {rel(mx(xn1, 'row'), R['xn1']) * 1e6:.0f} ppm, O {rel(o.transpose(1, 0, 2).reshape(64, 2048), R['o']) * 1e6:.0f} ppm, "
      f"O(MX) {rel(mx(o.transpose(1, 0, 2).reshape(64, 2048), 'row'), R['o']) * 1e6:.0f} ppm")
