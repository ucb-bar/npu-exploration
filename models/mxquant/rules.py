"""Which linear layers of the language model run through the recipe's Scheme.

A rule list is what ``mxq.nn.patch`` takes: ``(selector, Scheme | None)`` pairs, first match wins, ``None``
leaves the layer in bf16. The lists are written out here, as functions of the Scheme, so a name in a results
record ("mxquant_layers") always means the same layers.

    mxquant_layers   every nn.Linear except the attention projections; lm_head included. This is the layer set
                     MXQuant's eval_complete.py quantizes, so perplexities are comparable to its published numbers.
    all_linear       every nn.Linear, attention projections included.
    linears_no_head  every nn.Linear except lm_head, attention projections included: the decoder's projections
                     alone.
    mlp_linears                    the MLP linears alone: neither the attention projections nor lm_head.
    attention_projections          the attention projections alone.
    lm_head                        lm_head alone.
    attention_projections_lm_head  the attention projections and lm_head; the MLP linears stay bf16.
    mxquant_layers_core, all_linear_core
                     the same, plus the attention core: S = Q·Kᵀ and O = P·V of every attention module through the
                     Scheme too (mxq.nn.attend; scale, mask and softmax between them, at the run's vector precision).
"""
from __future__ import annotations

from torch import nn

import models  # noqa: F401  -- puts the mxq submodule on sys.path
from mxq.nn import is_attention


def mxquant_layers(scheme) -> list:
    return [(is_attention, None), (nn.Linear, scheme)]


def all_linear(scheme) -> list:
    return [(nn.Linear, scheme)]


def is_lm_head(name: str, module, parent) -> bool:
    return name.rsplit(".", 1)[-1] == "lm_head"


def linears_no_head(scheme) -> list:
    return [(is_lm_head, None), (nn.Linear, scheme)]


def mlp_linears(scheme) -> list:
    return [(is_attention, None), (is_lm_head, None), (nn.Linear, scheme)]


def attention_projections(scheme) -> list:
    return [(is_attention, scheme), (nn.Linear, None)]


def lm_head(scheme) -> list:
    return [(is_lm_head, scheme), (nn.Linear, None)]


def attention_projections_lm_head(scheme) -> list:
    return [(is_attention, scheme), (is_lm_head, scheme), (nn.Linear, None)]


def is_attention_module(name: str, module, parent) -> bool:
    """An attention module: it holds q_proj and k_proj (the module whose core mxq.nn.attend computes)."""
    return hasattr(module, "q_proj") and hasattr(module, "k_proj")


def with_core(rules):
    """A rule list, plus the attention core of every attention module through the same Scheme."""
    def build(scheme) -> list:
        return [(is_attention_module, (scheme, scheme))] + rules(scheme)
    return build


NAMES = {"mxquant_layers": mxquant_layers, "all_linear": all_linear, "linears_no_head": linears_no_head,
         "mlp_linears": mlp_linears, "attention_projections": attention_projections, "lm_head": lm_head,
         "attention_projections_lm_head": attention_projections_lm_head,
         "mxquant_layers_core": with_core(mxquant_layers), "all_linear_core": with_core(all_linear)}


def build(name: str, scheme) -> list:
    if name not in NAMES:
        raise ValueError(f"unknown rule list {name!r}; choose from {', '.join(NAMES)}")
    return NAMES[name](scheme)
