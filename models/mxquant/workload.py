"""The mxquant model, perplexity path: what the recipe's arithmetic does to a language model.

The question: with every linear layer of the workload's model computed the way THIS recipe's machine computes
a matmul (operand format and rounding, product format, the 16-lane accumulator ladder), what is the WikiText-2
perplexity, and how far is it from the bf16 model? The arithmetic is the same Scheme (``config/scheme.py``)
the bit path (``kernel.py``) grades spike against -- ``tests/selftest_workload.py`` holds the two paths to
the same bits on a linear layer -- so a perplexity is tied to a ``build_id`` that VERDICT has proven.

    from models import mxquant
    m = mxquant.evaluate("tinyllama", hw, run, gpus="0,1,2,3")    # minutes; cached under results/accuracy/<key>.json
    print(mxquant.workload.line(m))
    .venv/bin/python -m models.mxquant --workload tinyllama --hw baseline --run default --gpus 0,1,2,3

One subprocess per GPU (``_worker.py``, on mxq's ``experiments/llm_ppl.py``), always a subprocess even for one
GPU, so the caller never initialises CUDA. Unavailable without a GPU or the transformers / datasets /
accelerate packages: ``evaluate`` raises.

Every operand format runs. The four codebook (LUT) formats run through the chip's 16-entry tables, one per 2**G
tokens of A and per 2**G output channels of B (``mxq.block.lut``, the rule ``compiler/codebook.py`` calls; G and
the fit from the run recipe's ``lut``); the record's ``codebook`` says what was and was not modelled (no C table:
a layer's output is not requantized). Recipes mxq cannot run (non-uniform product lists) are refused before any
GPU work.
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
from config import recipe as _recipe
from config import scheme as _scheme
from models.mxquant import workloads as _workloads

REPO = models.REPO
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
        return False, "no CUDA GPU; the perplexity path runs a whole language model through mxq's arithmetic"
    return True, f"{n} GPU(s), mxq {models.mxq_commit()}"


def environment() -> dict:
    """The torch and transformers versions. The bf16 model's own loss moves with them (2026-09-24: 7.1885 under
    torch 2.9.1 / transformers 4.57.3, 7.1989 under torch 2.14.0 / transformers 5.17.0, same samples), so a
    perplexity is only comparable to another taken in the same environment; the key and the record carry it."""
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__}


def _settings(w: _workloads.Workload, recipe, run) -> dict:
    """Everything the number depends on. The cache key hashes this; the record carries it.

    ``recipe=None`` is the bf16 model, which depends on neither recipe. ``compiled`` is not in it:
    torch.compile changes the speed, not the bits (mxq tests that). The torch and transformers versions
    are (see :func:`environment`)."""
    d = {"model_id": w.model_id, "nsamples": w.nsamples, "seqlen": w.seqlen, "seed": w.seed,
         "mxq_commit": models.mxq_commit(), **environment()}
    if recipe is not None:
        d.update(recipe=recipe.name, build_id=recipe.build_id(), format=_scheme.mxq_format(run.operand_fmt),
                 rules=w.rules, rounding_mode=run.rounding, scale_floor=float(run.scale_floor))
        if _scheme.is_codebook(run.operand_fmt):
            d["codebook"] = _scheme.lut_record(run)
        if run.reduce != "hardware":
            d["reduce"] = run.reduce
    return d


def key(workload, recipe, run: _recipe.Run | None = None, **overrides) -> str:
    """The cache key of one measurement; ``recipe=None`` is the bf16 model."""
    w = _workloads.build(workload, **overrides)
    s = _settings(w, recipe, run or _recipe.Run())
    return hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]


def evaluate(workload, recipe, run: _recipe.Run, *, gpus: str | None = None, compiled: bool = True,
             results_dir: Path = RESULTS, force: bool = False, tel=None, **overrides) -> dict:
    """Perplexity of the workload on the hardware recipe driven by the run recipe, and of the bf16 model,
    both cached.

    ``workload`` is a registered name or a :class:`workloads.Workload`; ``overrides`` replace its fields
    (``nsamples=4``, ``model_id=...``). ``run.reduce`` is how the codes are multiplied
    (``config.scheme.REDUCERS``; "exact" and "bf16_tiles" are the ablations that split the recipe's cost into
    format, cross-block rounding and the array). Raises when it cannot run."""
    w = _workloads.build(workload, **overrides)
    _recipe.check(recipe, run, "perplexity")
    _scheme.scheme(recipe, run)                     # refuse before any GPU work
    _check(w)
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    common = dict(gpus=gpus, compiled=compiled, results_dir=results_dir, force=force, tel=tel)
    q = _measure(w, recipe, run, **common)
    b = _measure(w, None, run, **common)
    return {
        "perplexity": q["perplexity"], "bf16_perplexity": b["perplexity"],
        "delta": q["perplexity"] - b["perplexity"],
        "workload": w.name, "model_id": w.model_id, "nsamples": w.nsamples, "seqlen": w.seqlen, "seed": w.seed,
        "rules": w.rules,
        "recipe": recipe.name, "build_id": recipe.build_id(), "run": run.name, "run_id": run.run_id(),
        "dtype": run.operand_fmt, "format": _scheme.mxq_format(run.operand_fmt),
        "codebook": _scheme.lut_record(run),
        "rounding_mode": run.rounding, "scale_floor": run.scale_floor, "reduce": run.reduce, "compiled": compiled,
        "scheme": q["scheme"],
        "layers_quantized": sum(1 for row in q["layers"] if row[4]), "layers_total": len(q["layers"]),
        "seconds": q["seconds"], "gpus": gpus, "mxq_commit": q["mxq_commit"], **environment(),
        "cached": q["cached"], "key": q["key"], "path": str(q["path"]),
        "bf16_key": b["key"], "bf16_path": str(b["path"]),
    }


def line(m: dict) -> str:
    """The PPL line."""
    note = f"  [LUT G={m['codebook']['group']}]" if isinstance(m.get("codebook"), dict) else ""
    if m.get("reduce") not in (None, "hardware"):
        note = f"  reduce {m['reduce']}" + note
    return (f"PPL      {m['perplexity']:.4f}   bf16 {m['bf16_perplexity']:.4f}  ({m['delta']:+.4f})   "
            f"{m.get('workload', m['model_id'].rsplit('/', 1)[-1])} {m['dtype'] if m.get('dtype') else ''}  "
            f"{m['nsamples'] or 'all'}x{m['seqlen']}{'' if m.get('seed', 0) is not None else ' in order'}   "
            f"{'rules ' + m['rules'] if m.get('rules') else 'bf16 model, no recipe'}   "
            f"{m['seconds']:.0f} s{' [cached]' if m.get('cached') else ''}{note}")


def dry_run(workload, recipe, run: _recipe.Run, **overrides) -> int:
    """Print which layer gets the recipe's Scheme (loads the model, patches nothing, runs no sample)."""
    w = _workloads.build(workload, **overrides)
    _check(w)
    _recipe.check(recipe, run, "perplexity")
    _scheme.scheme(recipe, run)
    cmd = _worker_cmd(w, recipe, run, results_dir=RESULTS, compiled=False) + ["--dry-run"]
    return subprocess.run(cmd, cwd=REPO).returncode


def _content_file(results_dir: Path, name: str, raw: dict, kind: str) -> str:
    """The worker re-reads the recipe (a Scheme holds partials, not JSON): write it once, content-addressed."""
    results_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(raw, sort_keys=True)
    f = results_dir / f"{name}_{hashlib.sha256(text.encode()).hexdigest()[:16]}.{kind}.json"
    if not f.exists():
        f.write_text(text)
    return str(f)


def _worker_cmd(w: _workloads.Workload, recipe, run, *, results_dir: Path, compiled: bool) -> list[str]:
    cmd = [sys.executable, "-m", "models.mxquant._worker"]
    if recipe is None:
        cmd += ["--hw", "none"]
    else:
        run_raw = {"name": run.name, "description": run.description, **run.fields()}
        cmd += ["--hw", _content_file(results_dir, recipe.name, recipe.raw, "hardware"),
                "--run", _content_file(results_dir, run.name, run_raw, "run"), "--rules", w.rules]
    cmd += ["--model-id", w.model_id, "--seqlen", str(w.seqlen), "--nsamples", str(w.nsamples)]
    cmd += ["--sequential"] if w.seed is None else ["--seed", str(w.seed)]
    if not compiled:
        cmd.append("--no-compiled")
    return cmd


def _check(w: _workloads.Workload) -> None:
    """Refuse a workload no worker could run, before any GPU work or subprocess."""
    from models.mxquant import rules
    if w.nsamples < 0:
        raise ValueError(f"nsamples must be 0 (the whole split) or positive, got {w.nsamples}")
    if w.rules not in rules.NAMES:
        raise ValueError(f"unknown rule list {w.rules!r}; choose from {', '.join(rules.NAMES)}")


def _count(w: _workloads.Workload) -> int:
    """How many samples ``nsamples=0`` means: the loader's answer (tokenizer + dataset, CPU, no CUDA)."""
    from experiments.llm_ppl import load_samples
    return load_samples(w.model_id, w.seqlen, 0, w.seed).shape[0]


def bf16(workload, *, gpus: str | None = None, results_dir: Path = RESULTS, force: bool = False, tel=None,
         **overrides) -> dict:
    """Perplexity of the unpatched bf16 model alone (what ``evaluate`` measures beside every recipe)."""
    w = _workloads.build(workload, **overrides)
    _check(w)
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    b = _measure(w, None, _recipe.Run(), gpus=gpus, compiled=False, results_dir=results_dir, force=force, tel=tel)
    return {"perplexity": b["perplexity"], "bf16_perplexity": b["perplexity"], "delta": 0.0, "workload": w.name,
            "model_id": w.model_id, "nsamples": w.nsamples, "seqlen": w.seqlen, "seed": w.seed, "rules": None,
            "recipe": None, "build_id": None, "dtype": None, "format": None, "codebook": None, "reduce": None,
            "seconds": b["seconds"], "gpus": gpus, "mxq_commit": b["mxq_commit"], **environment(),
            "cached": b["cached"], "key": b["key"], "path": str(b["path"]), "bf16_key": b["key"], "bf16_path": str(b["path"])}


def _measure(w: _workloads.Workload, recipe, run, *, gpus: str | None, compiled: bool, results_dir: Path,
             force: bool, tel) -> dict:
    """One perplexity: from the cache, or measured by one worker per GPU and merged in sample order."""
    settings = _settings(w, recipe, run)
    k = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    path = results_dir / f"{k}.json"
    what = "bf16 model" if recipe is None else f"{recipe.name} + run {run.name} ({w.rules})"
    if path.exists() and not force:
        rec = json.loads(path.read_text())
        if tel:
            tel.log("mxquant", f"{what}: cached {rec['perplexity']:.6f}  ({path})")
        return {**rec, "cached": True, "key": k, "path": path}

    results_dir.mkdir(parents=True, exist_ok=True)
    base = _worker_cmd(w, recipe, run, results_dir=results_dir, compiled=compiled)
    devices = [g.strip() for g in gpus.split(",") if g.strip()] if gpus else [None]
    total = w.nsamples or _count(w)
    bounds = [round(i * total / len(devices)) for i in range(len(devices) + 1)]
    log = results_dir / f"{k}.log"
    if tel:
        tel.log("mxquant", f"{what}: {total} samples on {len(devices)} worker(s)  (log {log})")
    t0, parts, procs = time.time(), [], []
    try:
        with open(log, "a") as lf:
            lf.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}  {what}  {json.dumps(settings)}\n")
            lf.flush()
            for g, lo, hi in zip(devices, bounds, bounds[1:]):
                if lo == hi:
                    continue
                part = results_dir / f"{k}.part{lo}_{hi}.{os.getpid()}.json"
                parts.append(part)
                env = dict(os.environ)
                if g is not None:
                    env["CUDA_VISIBLE_DEVICES"] = g
                procs.append(subprocess.Popen(base + ["--samples", f"{lo}:{hi}", "--out", str(part)],
                                              cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT))
            codes = [p.wait() for p in procs]
        if any(codes):
            tail = "".join(log.read_text().splitlines(keepends=True)[-15:])
            raise RuntimeError(f"perplexity worker failed (exit {codes}); last lines of {log}:\n{tail}")
        records = [json.loads(p.read_text()) for p in parts]
    except BaseException:                   # a failed or interrupted run leaves no workers and no part files
        for p in procs:
            if p.poll() is None:
                p.kill()
        for p in parts:
            p.unlink(missing_ok=True)
        raise

    per_sample = sorted((s for r in records for s in r["per_sample"]), key=lambda s: s["index"])
    total, tokens = 0.0, 0
    for s in per_sample:                    # same order of addition as the single-process loop
        total += s["nll"]
        tokens += s["tokens"]
    rec = {**records[0], "per_sample": per_sample, "seconds": time.time() - t0, "gpus": gpus,
           "perplexity": math.exp(total / tokens), "settings": settings, "key": k}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)                   # a reader never sees a half-written cache file
    for p in parts:
        p.unlink()
    if tel:
        tel.log("mxquant", f"{what}: {rec['perplexity']:.6f}  in {rec['seconds']:.0f} s  -> {path}")
    return {**rec, "cached": False, "path": path}
