"""Workloads: the language-model evaluations the perplexity path can run, registered like kernels.

A workload is a model and how it is measured; the recipe says what machine it runs on. ``tinyllama`` is
measured the way MXQuant's published numbers were: 16 samples x 2048 tokens of the WikiText-2 test split
(mxq ``experiments/llm_ppl.py`` loads it), seed 0, attention projections left in bf16 (``rules.py``).
``nsamples=0`` is the whole split (165 samples at 2048) and ``seed=None`` the first ``nsamples`` in order;
the per-sample perplexity ranges 4 to 18, so 16 samples carry about +-0.8 of sample choice.

    from models.mxquant import workloads
    w = workloads.build("tinyllama", nsamples=4)          # a registered one, any field overridden
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Workload:
    name: str
    model_id: str                 #: a Hugging Face causal LM
    seqlen: int = 2048            #: tokens per sample
    nsamples: int = 16            #: WikiText-2 test samples of ``seqlen`` tokens; 0 = the whole test split
    seed: int | None = 0          #: which samples: drawn at random by the seed (MXQuant's protocol), None = the first nsamples in order
    rules: str = "mxquant_layers" #: which linear layers get the Scheme (rules.py)


WORKLOADS = {
    "tinyllama": Workload("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
}


def build(workload: str | Workload, **overrides) -> Workload:
    """A registered workload by name, or one passed in, with any of its fields overridden."""
    if isinstance(workload, str):
        if workload not in WORKLOADS:
            raise ValueError(f"unknown workload {workload!r}; registered: {', '.join(WORKLOADS)}")
        workload = WORKLOADS[workload]
    bad = set(overrides) - {f.name for f in fields(Workload)} | {"name"} & set(overrides)
    if bad:
        raise ValueError(f"a workload has no field {sorted(bad)}")
    return replace(workload, **overrides) if overrides else workload
