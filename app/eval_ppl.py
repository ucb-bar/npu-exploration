"""Baseline wikitext2 perplexity for the unquantized model, the standard protocol.

This is the REFERENCE the MXFP8 kernels are graded against, computed the same way the rest of this
repo loads the model (`eval_simquant.get_model` -> bfloat16), so the comparison is like for like.

WHY IT EXISTS SEPARATELY. `kernels/captures/llama_model.py` prints a perplexity too, but over the ONE
32-token window it captures -- which is a different quantity and is NOT comparable to published
numbers. Measured on TinyLlama-1.1B-Chat, bf16:

    seqlen   32,  1 window  (the capture's) : 28.509
    seqlen   32, 20 windows                 : 31.410
    seqlen  128, 20 windows                 : 16.475
    seqlen  512,  8 windows                 : 10.901
    seqlen 2048,  4 windows                 :  9.298

Context length is the whole story: at seqlen 32 every token is predicted from at most 31 tokens and
position 0 from BOS alone, so the mean context is ~16 tokens against ~1024 at seqlen 2048. Use this
script, not the capture's line, for any number quoted against the literature.

Per-window perplexity varies a lot at seqlen 2048 (window 0 alone scores 5.55 against a 4-window
mean of 9.30), so the running mean is printed per window rather than only at the end -- convergence
should be seen, not assumed.

    # on a GPU box (device_map="auto" lands on cuda when one is visible)
    python3 -m app.eval_ppl --seqlen 2048 --windows 16

    python3 -m app.eval_ppl --seqlen 2048 --windows 16 --device cuda:0
    python3 -m app.eval_ppl --seqlen 2048 --windows 0          # 0 = the WHOLE test split
    python3 -m app.eval_ppl --dtype float32                    # a true fp32 baseline
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

MXQ_ROOT = Path(__file__).resolve().parent.parent / "MXQuant"
WIKITEXT2 = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--seqlen", type=int, default=2048, help="context per window")
    ap.add_argument("--windows", type=int, default=16,
                    help="non-overlapping windows to average; 0 = the whole test split")
    ap.add_argument("--tok0", type=int, default=0, help="first token of the test split")
    ap.add_argument("--device", default=None,
                    help="cuda, cuda:0, cpu. Default: whatever device_map='auto' chose, which is "
                         "the GPU when one is visible.")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"),
                    help="bfloat16 is the repo's baseline (TinyLlama's native dtype, and what "
                         "MXQuant uses). float32 gives a true fp32 reference.")
    args = ap.parse_args()

    import torch
    sys.stdout.reconfigure(line_buffering=True)
    if str(MXQ_ROOT) not in sys.path:
        sys.path.insert(0, str(MXQ_ROOT))

    # Prefer MXQuant's own loader so the baseline is loaded EXACTLY as the rest of the repo loads
    # it. Fall back to plain transformers when MXQuant is not importable or when dtype/device are
    # overridden -- `get_model` pins bfloat16 and device_map="auto". The fallback is what lets this
    # script run in a MINIMAL venv (torch + transformers + datasets + sentencepiece + accelerate),
    # which matters because the repo's own .venv ships a CPU-ONLY torch and must not be disturbed:
    # the bit-exact mesh goldens come from a torch-based model, so changing that build under the
    # pipeline risks perturbing artifacts this repo's claims rest on.
    model = None
    how = ""
    if args.dtype == "bfloat16" and args.device is None and (MXQ_ROOT / "eval_simquant.py").exists():
        try:
            esq = _load("eval_simquant", MXQ_ROOT / "eval_simquant.py")
            model = esq.get_model(args.model_id, args.seqlen, args.seqlen, gpu=0)
            how = "eval_simquant.get_model (MXQuant's own loader)"
        except ImportError as e:
            # eval_simquant pulls in mxquant.* (and so qtorch); a minimal GPU venv will not have it.
            # The fallback below is EQUIVALENT for seqlen <= max_position_embeddings: get_model only
            # adds rope scaling (inactive at 2048), a vocab-32001 resize (TinyLlama is 32000) and
            # init-skip monkeypatches that cannot matter for pretrained weights. Verified to give
            # the identical perplexity.
            print(f"[load]  MXQuant unavailable ({e}); using transformers directly -- equivalent "
                  f"for seqlen <= 2048")
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, use_safetensors=True, trust_remote_code=True,
            torch_dtype=getattr(torch, args.dtype),
            device_map=("auto" if args.device is None else None))
        if args.device is not None:
            model = model.to(args.device)
        how = "transformers.from_pretrained (MXQuant not needed)"
    model.eval()
    dev = next(model.parameters()).device
    print(f"[model] {args.model_id}  dtype={next(model.parameters()).dtype}  device={dev}")
    print(f"[load]  via {how}")
    if dev.type == "cpu":
        cuda_build = "+cpu" not in torch.__version__
        print(f"[warn] running on CPU -- ~30 min for 16 windows of 2048, minutes on a GPU.")
        print(f"[warn] torch {torch.__version__}, cuda_available={torch.cuda.is_available()}"
              + ("" if cuda_build else "  <-- CPU-ONLY BUILD: it cannot use a GPU on any machine"))

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    td = load_dataset(*WIKITEXT2, split="test")
    enc = tok("\n\n".join(td["text"]), return_tensors="pt").input_ids
    total = (enc.shape[1] - args.tok0 - 1) // args.seqlen
    n = total if args.windows == 0 else min(args.windows, total)
    print(f"[data] test split {enc.shape[1]} tokens; {n} of {total} windows x {args.seqlen} "
          f"= {n * args.seqlen} ({100 * n * args.seqlen / enc.shape[1]:.1f}% of it)\n")

    tot, cnt = 0.0, 0
    with torch.no_grad():
        for w in range(n):
            a = args.tok0 + w * args.seqlen
            ids = enc[:, a:a + args.seqlen].to(dev)
            lab = enc[0, a + 1:a + args.seqlen + 1].to(dev)
            lg = model(input_ids=ids, use_cache=False).logits[0].float()
            s = torch.nn.functional.cross_entropy(lg, lab, reduction="sum").item()
            tot += s
            cnt += args.seqlen
            print(f"  window {w:3d}  this {math.exp(s / args.seqlen):8.3f}   "
                  f"running {math.exp(tot / cnt):8.3f}")

    print(f"\nRESULT  wikitext2  {args.model_id}  {args.dtype}  seqlen {args.seqlen}  "
          f"{n} windows  ppl = {math.exp(tot / cnt):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
