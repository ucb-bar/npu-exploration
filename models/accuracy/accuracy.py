"""The accuracy model: what the recipe's arithmetic does to a language model.

The question it answers: with every linear layer of TinyLlama computed the way THIS recipe's machine computes a
matmul (its operand format and rounding, its product format, its 16-lane accumulator ladder), what is the
WikiText-2 perplexity, and how far is it from the bf16 model? The arithmetic comes from config/scheme.py, the same
functions the mxquant model grades spike against, so a perplexity is tied to a build_id that VERDICT has proven.

    from models import accuracy
    m = accuracy.run(recipe, gpus="0,1,2,3")          # minutes; cached under results/accuracy/<key>.json
    print(accuracy.line(m))
    .venv/bin/python -m models.accuracy --config baseline --gpus 0,1,2,3

Measured the way MXQuant's published numbers were: 16 samples x 2048 tokens of the WikiText-2 test split, seed 0,
attention projections left in bf16 (rules "mxquant_layers"). One subprocess per GPU (models/accuracy/_worker.py),
always a subprocess even for one GPU, so the caller never initialises CUDA. Unavailable without a GPU or the
transformers / datasets / accelerate packages: run() raises and the pipeline reports UNAVAILABLE.

Recipes mxq cannot run at model level -- codebook (LUT) formats, non-uniform product lists -- are refused by
config.scheme before any GPU work.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import models
from config import scheme as _scheme

REPO = models.REPO
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
RESULTS = REPO / "results" / "accuracy"
_PACKAGES = ("transformers", "datasets", "accelerate")


def available() -> tuple[bool, str]:
    if not models.paths():
        return False, models.mxq_missing()
    missing = [m for m in _PACKAGES if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"missing packages {missing} (pip install -r requirements.txt)"
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        return False, "no CUDA GPU; the model runs a 1.1B-parameter network through mxq's arithmetic"
    return True, f"{n} GPU(s), mxq {models.mxq_commit()}"


def _settings(recipe, *, model_id: str, nsamples: int, seqlen: int, seed: int, rules: str,
              rounding_mode: str, scale_floor: float) -> dict:
    """Everything the number depends on. The cache key hashes this; the record carries it.

    ``compiled`` is not in it: torch.compile changes the speed, not the bits (mxq tests that)."""
    d = {"model_id": model_id, "nsamples": nsamples, "seqlen": seqlen, "seed": seed,
         "mxq_commit": models.mxq_commit()}
    if recipe is not None:
        d.update(recipe=recipe.name, build_id=recipe.build_id(), format=_scheme.format_name(recipe),
                 rules=rules, rounding_mode=rounding_mode, scale_floor=float(scale_floor))
    return d


def key(recipe, *, model_id: str = MODEL_ID, nsamples: int = 16, seqlen: int = 2048, seed: int = 0,
        rules: str = "mxquant_layers", rounding_mode: str = _scheme.ROUNDING,
        scale_floor: float | None = None) -> str:
    """The cache key of one measurement; ``recipe=None`` is the bf16 model."""
    floor = _scheme.scale_floor_default() if scale_floor is None else float(scale_floor)
    s = _settings(recipe, model_id=model_id, nsamples=nsamples, seqlen=seqlen, seed=seed, rules=rules,
                  rounding_mode=rounding_mode, scale_floor=floor)
    return hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]


def run(recipe, *, model_id: str = MODEL_ID, nsamples: int = 16, seqlen: int = 2048, seed: int = 0,
        gpus: str | None = None, rules: str = "mxquant_layers", rounding_mode: str = _scheme.ROUNDING,
        scale_floor: float | None = None, compiled: bool = True, results_dir: Path = RESULTS,
        force: bool = False, tel=None) -> dict:
    """Perplexity of the recipe machine and of the bf16 model, both cached. Raises when it cannot run."""
    _scheme.scheme(recipe, rounding_mode=rounding_mode, scale_floor=scale_floor)   # refuse before any GPU work
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    floor = _scheme.scale_floor_default() if scale_floor is None else float(scale_floor)
    common = dict(model_id=model_id, nsamples=nsamples, seqlen=seqlen, seed=seed, rules=rules,
                  rounding_mode=rounding_mode, scale_floor=floor)
    q = _measure(recipe, gpus=gpus, compiled=compiled, results_dir=results_dir, force=force, tel=tel, **common)
    b = _measure(None, gpus=gpus, compiled=compiled, results_dir=results_dir, force=force, tel=tel, **common)
    return {
        "perplexity": q["perplexity"], "bf16_perplexity": b["perplexity"],
        "delta": q["perplexity"] - b["perplexity"],
        "model_id": model_id, "nsamples": nsamples, "seqlen": seqlen, "seed": seed, "rules": rules,
        "recipe": recipe.name, "build_id": recipe.build_id(), "format": _scheme.format_name(recipe),
        "rounding_mode": rounding_mode, "scale_floor": floor, "compiled": compiled,
        "scheme": q["scheme"],
        "layers_quantized": sum(1 for row in q["layers"] if row[4]), "layers_total": len(q["layers"]),
        "seconds": q["seconds"], "gpus": gpus, "mxq_commit": q["mxq_commit"],
        "cached": q["cached"], "key": q["key"], "path": str(q["path"]),
        "bf16_key": b["key"], "bf16_path": str(b["path"]),
    }


def line(m: dict) -> str:
    """The PPL line run_kernel.py prints."""
    short = m["model_id"].rsplit("/", 1)[-1]
    return (f"PPL      {m['perplexity']:.4f}   bf16 {m['bf16_perplexity']:.4f}  ({m['delta']:+.4f})   "
            f"{short}  {m['nsamples']}x{m['seqlen']}   rules {m['rules']}   "
            f"{m['seconds']:.0f} s{' [cached]' if m.get('cached') else ''}")


def dry_run(recipe, *, rules: str = "mxquant_layers", rounding_mode: str = _scheme.ROUNDING,
            scale_floor: float | None = None, model_id: str = MODEL_ID, seqlen: int = 2048) -> int:
    """Print which layer gets the recipe's Scheme (loads the model, patches nothing, runs no sample)."""
    _scheme.scheme(recipe, rounding_mode=rounding_mode, scale_floor=scale_floor)
    cmd = _worker_cmd(recipe, results_dir=RESULTS, rules=rules, rounding_mode=rounding_mode,
                      scale_floor=_scheme.scale_floor_default() if scale_floor is None else float(scale_floor),
                      compiled=False, model_id=model_id, seqlen=seqlen, nsamples=1, seed=0) + ["--dry-run"]
    return subprocess.run(cmd, cwd=REPO).returncode


def _worker_cmd(recipe, *, results_dir: Path, rules: str, rounding_mode: str, scale_floor: float,
                compiled: bool, model_id: str, seqlen: int, nsamples: int, seed: int) -> list[str]:
    if recipe is None:
        rpath = "none"
    else:                                   # the worker re-reads the recipe: a Scheme holds partials, not JSON
        results_dir.mkdir(parents=True, exist_ok=True)
        rfile = results_dir / f"{recipe.name}_{recipe.build_id()}.recipe.json"
        if not rfile.exists():
            rfile.write_text(json.dumps(recipe.raw, indent=1))
        rpath = str(rfile)
    cmd = [sys.executable, "-m", "models.accuracy._worker", "--recipe", rpath, "--rules", rules,
           "--rounding-mode", rounding_mode, "--scale-floor", repr(scale_floor),
           "--model-id", model_id, "--seqlen", str(seqlen), "--nsamples", str(nsamples), "--seed", str(seed)]
    if not compiled:
        cmd.append("--no-compiled")
    return cmd


def _measure(recipe, *, gpus: str | None, compiled: bool, results_dir: Path, force: bool, tel, model_id: str,
             nsamples: int, seqlen: int, seed: int, rules: str, rounding_mode: str, scale_floor: float) -> dict:
    """One perplexity: from the cache, or measured by one worker per GPU and merged in sample order."""
    settings = _settings(recipe, model_id=model_id, nsamples=nsamples, seqlen=seqlen, seed=seed, rules=rules,
                         rounding_mode=rounding_mode, scale_floor=scale_floor)
    k = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    path = results_dir / f"{k}.json"
    what = "bf16 model" if recipe is None else f"{recipe.name} ({rules})"
    if path.exists() and not force:
        rec = json.loads(path.read_text())
        if tel:
            tel.log("accuracy", f"{what}: cached {rec['perplexity']:.6f}  ({path})")
        return {**rec, "cached": True, "key": k, "path": path}

    results_dir.mkdir(parents=True, exist_ok=True)
    base = _worker_cmd(recipe, results_dir=results_dir, rules=rules, rounding_mode=rounding_mode,
                       scale_floor=scale_floor, compiled=compiled, model_id=model_id, seqlen=seqlen,
                       nsamples=nsamples, seed=seed)
    devices = gpus.split(",") if gpus else [None]
    bounds = [round(i * nsamples / len(devices)) for i in range(len(devices) + 1)]
    log = results_dir / f"{k}.log"
    if tel:
        tel.log("accuracy", f"{what}: {nsamples} samples on {len(devices)} worker(s)  (log {log})")
    t0, parts, procs = time.time(), [], []
    with open(log, "a") as lf:
        lf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  {what}  {json.dumps(settings)}\n")
        lf.flush()
        for g, lo, hi in zip(devices, bounds, bounds[1:]):
            if lo == hi:
                continue
            part = results_dir / f"{k}.part{lo}_{hi}.json"
            parts.append(part)
            env = dict(os.environ)
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = g
            procs.append(subprocess.Popen(base + ["--samples", f"{lo}:{hi}", "--out", str(part)],
                                          cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
    if any(codes):
        tail = "".join(log.read_text().splitlines(keepends=True)[-15:])
        raise RuntimeError(f"accuracy worker failed (exit {codes}); last lines of {log}:\n{tail}")

    records = [json.loads(p.read_text()) for p in parts]
    per_sample = sorted((s for r in records for s in r["per_sample"]), key=lambda s: s["index"])
    total, tokens = 0.0, 0
    for s in per_sample:                    # same order of addition as the single-process loop
        total += s["nll"]
        tokens += s["tokens"]
    rec = {**records[0], "per_sample": per_sample, "seconds": time.time() - t0, "gpus": gpus,
           "perplexity": math.exp(total / tokens), "settings": settings, "key": k}
    path.write_text(json.dumps(rec, indent=1))
    for p in parts:
        p.unlink()
    if tel:
        tel.log("accuracy", f"{what}: {rec['perplexity']:.6f}  in {rec['seconds']:.0f} s  -> {path}")
    return {**rec, "cached": False, "path": path}
