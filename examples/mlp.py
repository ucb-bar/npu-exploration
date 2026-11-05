"""A full-precision PyTorch model, as you would write it: a SwiGLU MLP block.

    .venv/bin/python npu.py --hw baseline --model examples/mlp.py:MLP
"""
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, d: int = 64, hidden: int = 128):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
