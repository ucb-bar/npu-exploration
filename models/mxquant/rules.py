"""Which linear layers of the language model run through the recipe's Scheme.

A rule list is what ``mxq.nn.patch`` takes: ``(selector, Scheme | None)`` pairs, first match wins, ``None``
leaves the layer in bf16. The lists are written out here, as functions of the Scheme, so a name in a results
record ("mxquant_layers") always means the same layers.

    mxquant_layers   every nn.Linear except the attention projections; lm_head included. This is the layer set
                     MXQuant's eval_complete.py quantizes, so perplexities are comparable to its published numbers.
    all_linear       every nn.Linear, attention projections included.
"""
from __future__ import annotations

from torch import nn

import models  # noqa: F401  -- puts the mxq submodule on sys.path
from mxq.nn import is_attention


def mxquant_layers(scheme) -> list:
    return [(is_attention, None), (nn.Linear, scheme)]


def all_linear(scheme) -> list:
    return [(nn.Linear, scheme)]


NAMES = {"mxquant_layers": mxquant_layers, "all_linear": all_linear}


def build(name: str, scheme) -> list:
    if name not in NAMES:
        raise ValueError(f"unknown rule list {name!r}; choose from {', '.join(NAMES)}")
    return NAMES[name](scheme)
