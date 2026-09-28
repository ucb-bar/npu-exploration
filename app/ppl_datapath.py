"""wikitext2 perplexity under the MX format and the MxGemmini datapath, at real context, on a GPU.

`ablate_mx.py` answered "format or datapath?" at the captures' 32-token window, where the baseline
is 28.51 rather than the benchmark 7.88. This runs the same question at any seqlen, on the live HF
model rather than the npz captures, through the SAME verified arithmetic:

    bf16          the unquantized model -- the baseline (7.88 at seqlen 2048)
    mx-exact      MXFP8 operands (E4M3 + E8M0 per-32 block scales), EXACT fp32 accumulation.
                  This is MXFP8 as the literature means it: the format's cost, perfect datapath.
    mx-bf16acc    ... plus the hardware's CROSS-TILE behaviour: each 32-element group reduced
                  exactly, then accumulated into the result in bf16 via `bf16_accum_add`, the
                  golden's own function. Isolates bf16 cross-tile accumulation from the narrow
                  per-lane accumulator.
    rtl-exact     the full datapath, via MXQuant's `MXLinearSim` under `rtl_exact/` --
                  e4m3 TRUNCATED products and the per-lane schedule
                  [(4,4)]*8 + [(4,5)]*2 + [(4,6)]*5 + [(8,7)]*1. `verify_rtl_exact.py` gates this
                  configuration as BIT-IDENTICAL to the hardware (65536/65536, max abs diff 0), so
                  it is the hardware's perplexity, not a lookalike model's.

Run in that order and the cost decomposes: format, then cross-tile, then the lanes and the product
quantizer. `ablate_mx.py` measured format +1.84 ppl and datapath +2.42 at seqlen 32; this says
whether those proportions survive at real context.

GPU. Everything here is device-agnostic. `rtl-exact` only became viable on a GPU once
`fp_quantize_rne` / `fp_add_exact` stopped being scalar Python loops over `.cpu().tolist()` (see
llama_layer_hw_plan.md 13.10) -- before that it forced a device round trip per k-step.

    python3 -m app.ppl_datapath --mode bf16       --seqlen 2048 --windows 16
    python3 -m app.ppl_datapath --mode mx-exact   --seqlen 2048 --windows 16
    python3 -m app.ppl_datapath --mode mx-bf16acc --seqlen 2048 --windows 16
    python3 -m app.ppl_datapath --mode rtl-exact  --seqlen 512  --windows 4    # the expensive one

SCOPE, stated because it bounds the answer: this quantizes the LINEAR layers (q/k/v/o, gate/up/down
and, with --lm-head, the head). The kernel additionally quantizes attention's two per-head matmuls
(Q@K^T and P@V), which this does not -- so the reported cost is a LOWER BOUND. Run
`--seqlen 32 --windows 1 --mode mx-exact` and compare against ablate_mx.py's 30.4450, which does
quantize them, to size that gap on the same tokens.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

NPU = Path(__file__).resolve().parent.parent
ROCC = NPU.parent / "software" / "gemmini-rocc-tests"
for _p in (str(NPU), str(ROCC), str(NPU / "rtl_exact"), str(NPU / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WIKITEXT2 = ("Salesforce/wikitext", "wikitext-2-raw-v1")
BLOCK = 32
MODES = ("bf16", "mx-exact", "mx-bf16acc", "rtl-exact")


# --- MX quantization, block 32 along the reduction axis -------------------------------------------

def _mx_dequant(t, dim: int, block: int = BLOCK):
    """Quantize to MXFP8 (E4M3 codes + one E8M0 scale per `block` along `dim`) and return the
    dequantized value -- what an exact matmul would multiply.

    Scale is `2**floor(log2 amax)` with no log2_pmax term, which is MXQuant's convention and the
    hardware's (chain_seam_hw_notes.md 8/9.4). E4M3 uses torch's native fp8: the operands are
    block-normalized into [0, 2), so the one code where `float8_e4m3fn` differs from this repo's
    table (e=15, reserved for NaN there) is unreachable.
    """
    import torch
    K = t.shape[dim]
    assert K % block == 0, f"reduction dim {K} is not a multiple of {block}"
    tm = t.movedim(dim, -1)
    shp = tm.shape
    g = tm.reshape(-1, K // block, block)
    amax = g.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(amax.clamp_min(torch.finfo(torch.float32).tiny)))
    e = e.clamp(-127, 127)
    X = torch.exp2(e)
    X = torch.where(amax > 0, X, torch.ones_like(X))
    P = (g / X).to(torch.float8_e4m3fn).to(torch.float32)
    return (P * X).reshape(shp).movedim(-1, dim)


def _mx_linear_cls():
    """`nn.Linear` with the matmul replaced by an MX arithmetic model. Weight quantized once.

    Built lazily so the module imports without torch, and as an `nn.Module` subclass because
    assigning a plain object as a child raises ("torch.nn.Module or None expected").
    """
    import torch
    import torch.nn as nn

    class MXLinear(nn.Module):
        def __init__(self, lin, mode: str):
            super().__init__()
            self.mode = mode
            self.register_buffer("bias_t", None if lin.bias is None
                                 else lin.bias.detach().to(torch.float32))
            W = lin.weight.detach().to(torch.float32)      # [out, in]; reduction is `in`
            self.register_buffer("Wq", _mx_dequant(W, dim=1))

        forward = _mx_linear_forward

    return MXLinear


def _mx_linear_forward(self, x):  # noqa: E301  (bound as MXLinear.forward)
        import torch
        import fp8_matmul_model as FM
        xs = x.shape
        xf = x.reshape(-1, xs[-1]).to(torch.float32)
        Aq = _mx_dequant(xf, dim=-1)
        if self.mode == "mx-exact":
            out = Aq @ self.Wq.T
        else:                                               # mx-bf16acc
            # Each 32-element group is reduced exactly, then folded into the running result in
            # bf16 -- `bf16_accum_add` is the golden's own cross-tile step, so this is the
            # hardware's accumulation with a perfect in-group accumulator.
            out = torch.zeros(xf.shape[0], self.Wq.shape[0], dtype=torch.float32, device=x.device)
            for k0 in range(0, xf.shape[-1], BLOCK):
                tile = Aq[:, k0:k0 + BLOCK] @ self.Wq[:, k0:k0 + BLOCK].T
                out = FM.bf16_accum_add(out, tile)
        if self.bias_t is not None:
            out = out + self.bias_t
        return out.reshape(*xs[:-1], self.Wq.shape[0]).to(x.dtype)


def _target_linears(model, lm_head: bool):
    """The decoder's projections, and optionally the head."""
    import torch.nn as nn
    out = []
    for layer in model.model.layers:
        sa, mlp = layer.self_attn, layer.mlp
        out += [(sa, "q_proj"), (sa, "k_proj"), (sa, "v_proj"), (sa, "o_proj"),
                (mlp, "gate_proj"), (mlp, "up_proj"), (mlp, "down_proj")]
    if lm_head:
        out.append((model, "lm_head"))
    return [(p, n) for p, n in out if isinstance(getattr(p, n), nn.Linear)]


def _progress_wrap(sim, idx: int, total: int, name: str):
    """Report each projection as it completes.

    `rtl-exact` is minutes-to-hours per window and the model forward is opaque from outside, so
    without this a run is indistinguishable from a hang -- which is exactly how the first GPU
    attempt looked. Prints to stderr so it does not interleave with the per-window results.
    """
    import time
    import torch.nn as nn

    class Progress(nn.Module):
        def __init__(self):
            super().__init__()
            self.sim = sim

        def forward(self, x):
            t0 = time.time()
            out = self.sim(x)
            print(f"    [rtl] {idx + 1:3d}/{total} {name:<10s} {tuple(x.shape)} "
                  f"{time.time() - t0:6.2f}s", file=sys.stderr, flush=True)
            return out

    return Progress()


def install_mode(model, mode: str, lm_head: bool) -> str:
    if mode == "bf16":
        return "unmodified"
    targets = _target_linears(model, lm_head)
    if mode in ("mx-exact", "mx-bf16acc"):
        cls = _mx_linear_cls()
        for parent, name in targets:
            setattr(parent, name, cls(getattr(parent, name), mode))
        return f"{len(targets)} linears -> {mode}"

    # rtl-exact: MXQuant's own simulator under the verified config.
    import verify_rtl_exact as V
    sys.path.insert(0, str(V._prodacc_dir(None)))
    import eval_complete as EC
    import rtl_datapath
    cfg = rtl_datapath.load_config()
    rtl_datapath.install(EC, cfg)
    assert rtl_datapath.is_installed(EC), "rtl_datapath.install() did not take"
    sims = []
    for parent, name in targets:
        lin = getattr(parent, name)
        sim = EC.MXLinearSim(lin, cfg.mx_fmt, False, cfg.product[0], cfg.product[1],
                             cfg.acc_schedule, 0, 0, window=cfg.window)
        setattr(parent, name, _progress_wrap(sim, len(sims), len(targets), name))
        sims.append(sim)
    return (f"{len(targets)} linears -> MXLinearSim, rtl_exact installed "
            f"(prod e{cfg.product[0]}m{cfg.product[1]}, {len(cfg.acc_schedule)}-lane schedule)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=16, help="0 = the whole test split")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, cpu (default: auto)")
    ap.add_argument("--threads", type=int, default=32,
                    help="CPU threads for torch intra-op parallelism. The default matters: torch "
                         "would otherwise take HALF the visible cores (128 of 256 on this box), "
                         "which is antisocial on a shared machine and makes timings unreproducible. "
                         "Ignored in effect on CUDA. 0 = leave torch's default alone.")
    ap.add_argument("--lm-head", action="store_true",
                    help="quantize lm_head too. The kernel does; it feeds logits directly, so it "
                         "is commonly left in higher precision and is worth measuring both ways.")
    args = ap.parse_args()

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
        try:                      # only settable before any parallel work has started
            torch.set_num_interop_threads(args.threads)
        except RuntimeError:
            pass
    sys.stdout.reconfigure(line_buffering=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, use_safetensors=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(dev).eval()
    print(f"[model] {args.model_id}  dtype=bfloat16  device={dev}")
    if dev == "cpu":
        print(f"[warn] CPU. torch {torch.__version__}"
              + ("" if "+cpu" not in torch.__version__ else "  <-- CPU-ONLY BUILD"))

    t0 = time.time()
    how = install_mode(model, args.mode, args.lm_head)
    print(f"[mode]  {args.mode}: {how}   ({time.time() - t0:.1f}s)")
    if args.mode != "bf16" and not args.lm_head:
        print("[note]  lm_head NOT quantized (--lm-head to include it)")
    if args.mode != "bf16":
        print("[note]  attention's Q@K^T and P@V are NOT quantized -- the kernel does quantize "
              "them, so this is a LOWER BOUND on the cost")

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    td = load_dataset(*WIKITEXT2, split="test")
    enc = tok("\n\n".join(td["text"]), return_tensors="pt").input_ids
    total = (enc.shape[1] - 1) // args.seqlen
    n = total if args.windows == 0 else min(args.windows, total)
    print(f"[data]  {enc.shape[1]} tokens; {n} of {total} windows x {args.seqlen}\n")

    tot, cnt, t0 = 0.0, 0, time.time()
    with torch.no_grad():
        for w in range(n):
            a = w * args.seqlen
            ids = enc[:, a:a + args.seqlen].to(dev)
            lab = enc[0, a + 1:a + args.seqlen + 1].to(dev)
            lg = model(input_ids=ids, use_cache=False).logits[0].float()
            s = torch.nn.functional.cross_entropy(lg, lab, reduction="sum").item()
            tot += s
            cnt += args.seqlen
            print(f"  window {w:3d}  this {math.exp(s / args.seqlen):9.4f}   "
                  f"running {math.exp(tot / cnt):9.4f}   ({time.time() - t0:.0f}s)")

    print(f"\nRESULT  {args.mode:<11} seqlen {args.seqlen}  {n} windows  "
          f"ppl = {math.exp(tot / cnt):.4f}   nll = {tot / cnt:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
