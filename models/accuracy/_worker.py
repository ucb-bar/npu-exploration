"""One accuracy-model worker: load the language model, put the recipe's Scheme into its linear layers, sum the
negative log-likelihood over a slice of the WikiText-2 samples, write one JSON part.

    .venv/bin/python -m models.accuracy._worker --recipe config/recipes/baseline.json --samples 0:4 --out part.json
    .venv/bin/python -m models.accuracy._worker --recipe none --samples 0:16 --out bf16.json      # the unpatched model
    .venv/bin/python -m models.accuracy._worker --recipe config/recipes/wide_acc.json --dry-run  # the patch table only

Always a separate process: models/accuracy/accuracy.py launches one per GPU (CUDA_VISIBLE_DEVICES), so the
pipeline process never initialises CUDA. Data sampling, model loading and the loss are MXQuant's
(complete_integration_e2e/eval_complete.py: load_data_sampled, load_model, eval_sampled), taken from
microscaling-quant/experiments/llm_ppl.py -- experiments/ is not part of the mxq package the submodule pins,
so the four functions live here -- which keeps the numbers comparable to MXQuant's published ones and to mxq's
recorded runs (hw_fp8 7.343833, bf16 7.188465).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import models  # noqa: E402  -- puts the mxq submodule on sys.path

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def load_model(model_id: str, seqlen: int):
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_id)
    ctx = getattr(config, "max_position_embeddings", None)
    if ctx and seqlen > ctx:
        config.rope_scaling = {"type": "linear", "factor": float(math.ceil(seqlen / ctx))}
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_safetensors=True,
                                                 torch_dtype=torch.bfloat16, device_map="auto")
    return model.eval().bfloat16()


def load_samples(model_id: str, seqlen: int, nsamples: int, seed: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([x for x in data["text"] if x.strip()])
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    nbatch = ids.numel() // seqlen
    ids = ids[:nbatch * seqlen].reshape(nbatch, seqlen)
    torch.manual_seed(seed)
    if nsamples < nbatch:
        ids = ids[torch.randperm(nbatch)[:nsamples]]
    return ids


@torch.no_grad()
def sample_nll(model, ids):
    device = next(model.parameters()).device
    ids = ids.unsqueeze(0).to(device)
    logits = model(ids, use_cache=False).logits[0, :-1, :]
    nll = -F.log_softmax(logits.float(), dim=-1).gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
    return nll.sum().item(), nll.numel()


def describe(obj):
    """A Scheme, rule list or any piece of one as plain JSON: functions by their dotted name, partials with
    their bound keywords, an Arithmetic by its name. Read off the objects, so the record says what ran."""
    from functools import partial
    from mxq import Scheme
    from mxq.matmul import Arithmetic
    if isinstance(obj, Scheme):
        return {"name": obj.name, "a": describe(obj.a), "b": describe(obj.b), "reduce": describe(obj.reduce)}
    if isinstance(obj, partial):
        return {"call": describe(obj.func), **{k: describe(v) for k, v in obj.keywords.items()}}
    if isinstance(obj, Arithmetic):
        return obj.name
    if isinstance(obj, type) or callable(obj):
        public = [m for m in obj.__module__.split(".") if not m.startswith("_")]
        return ".".join(public + [obj.__qualname__])
    if isinstance(obj, (list, tuple)):
        return [describe(v) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True, help="recipe .json path, or 'none' for the unpatched bf16 model")
    ap.add_argument("--rules", default="mxquant_layers")
    ap.add_argument("--rounding-mode", default=None, help="operand rounding (default: config.scheme.ROUNDING)")
    ap.add_argument("--scale-floor", type=float, default=None, help="block-max floor (default: 2^-23)")
    ap.add_argument("--no-compiled", action="store_true", help="do not torch.compile the arithmetic (5-7x slower, same bits)")
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=None, help="token rows per reducer call (default: by output width)")
    ap.add_argument("--samples", default=None, help="START:END sample indices (default: all)")
    ap.add_argument("--out", type=Path, default=None, help="part JSON to write")
    ap.add_argument("--dry-run", action="store_true", help="print the patch table and stop")
    args = ap.parse_args()

    from config import scheme as S
    from config.recipe import load
    from models.accuracy import rules as R
    from mxq.nn import patch

    if not args.no_compiled:
        # mxq.nn.patch smoke-tests every Scheme once on a tiny CPU matmul before touching the model. With
        # torch.compile that takes inductor's C++ backend, whose vectorised codegen miscompiles this kernel in
        # torch 2.14 (VectorizedN<int,2> - Vectorized<int>). Scalar CPU codegen is fine, and only that smoke
        # test ever runs on the CPU: the layers run on the GPU through Triton. Verified bit-identical to the
        # uncompiled arithmetic either way.
        import torch._inductor.config as inductor_config
        inductor_config.cpp.simdlen = 1
    recipe = None if args.recipe == "none" else load(args.recipe)
    knobs = {k: v for k, v in (("rounding_mode", args.rounding_mode), ("scale_floor", args.scale_floor)) if v is not None}
    sch = None if recipe is None else S.scheme(recipe, compiled=not args.no_compiled, **knobs)
    rule_list = None if sch is None else R.build(args.rules, sch)

    if args.dry_run and sch is None:
        print("no recipe: the bf16 model, nothing patched")
        return 0
    model = load_model(args.model_id, args.seqlen)
    table = []
    if rule_list is not None:
        handle = patch(model, rule_list, chunk=args.chunk, dry_run=args.dry_run)
        table = handle.table
        if args.dry_run:
            print(handle)
            print(f"\n{sum(1 for r in table if r[4])} of {len(table)} linear layers get {sch.name}")
            return 0
    if args.out is None:
        ap.error("--out is required unless --dry-run")

    ids = load_samples(args.model_id, args.seqlen, args.nsamples, args.seed)
    lo, hi = (int(v) for v in args.samples.split(":")) if args.samples else (0, ids.shape[0])
    t0, per_sample = time.time(), []
    for i in range(lo, hi):
        nll, n = sample_nll(model, ids[i])
        per_sample.append({"index": i, "nll": nll, "tokens": n})
        print(f"  [{'bf16' if sch is None else sch.name}] sample {i + 1}/{ids.shape[0]}  nll/token {nll / n:.6f}  "
              f"{time.time() - t0:.0f}s", flush=True)

    record = {
        "model_id": args.model_id, "seqlen": args.seqlen, "nsamples": args.nsamples, "seed": args.seed,
        "rules": None if sch is None else args.rules,
        "recipe": None if recipe is None else recipe.name,
        "build_id": None if recipe is None else recipe.build_id(),
        "format": None if recipe is None else S.format_name(recipe),
        "rounding_mode": None if sch is None else knobs.get("rounding_mode", S.ROUNDING),
        "scale_floor": None if sch is None else knobs.get("scale_floor", S.scale_floor_default()),
        "compiled": None if sch is None else not args.no_compiled,
        "mxq_commit": models.mxq_commit(),
        "scheme": None if sch is None else describe(sch),
        "rule_list": None if rule_list is None else [[describe(sel), None if s is None else s.name] for sel, s in rule_list],
        "layers": table,
        "per_sample": per_sample,
        "seconds": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
