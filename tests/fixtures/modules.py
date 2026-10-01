"""Plain PyTorch modules the compile selftest traces and compiles (``compile_kernel.py --module``).

    .venv/bin/python compile_kernel.py --module tests/fixtures/modules.py:Attn
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from kernels.trace import RoPE


class Attn(nn.Module):
    """Single-head causal attention, d=64: five matmuls and one softmax, one graph ELF."""

    def __init__(self, d: int = 64):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(d, d, dtype=torch.bool), diagonal=1))
        self.scale = 1.0 / math.sqrt(d)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        s = torch.matmul(q, k.t()) * self.scale
        p = F.softmax(s.masked_fill(self.mask, float("-inf")), dim=-1)
        return torch.matmul(p, v)


class SwiGLU(nn.Module):
    """gate/up/down with silu(g)*u: two matmuls, one swiglu, one matmul."""

    def __init__(self, d: int = 64, h: int = 64):
        super().__init__()
        self.g = nn.Linear(d, h, bias=False)
        self.u = nn.Linear(d, h, bias=False)
        self.d = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.d(F.silu(self.g(x)) * self.u(x))


class Biased(nn.Linear):
    """A Linear WITH a bias: what the tracer refuses."""

    def __init__(self, d: int = 64):
        super().__init__(d, d, bias=True)


class RopeChain(nn.Sequential):
    """Linear -> RoPE (fixed tables, llama-style duplicated halves) -> Linear."""

    def __init__(self, d: int = 64, m: int = 64):
        pos = torch.arange(m).float()[:, None]
        freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
        ang = torch.cat([pos * freq, pos * freq], dim=-1)
        super().__init__(nn.Linear(d, d, bias=False), RoPE(ang.cos(), ang.sin()),
                         nn.Linear(d, d, bias=False))
