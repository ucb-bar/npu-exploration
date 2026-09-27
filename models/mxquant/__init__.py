"""The mxquant model: the exact bits the recipe machine must produce for a kernel, computed on mxq.

mxquant.py is the model (run, compare, line); block.py is MXQuant's block-quantizer API computed by
mxq, shared with the wire-operand encoder in app/mxq_golden.py on purpose: the operands the ELF
carries and the operands the model multiplies must be the same bytes.
"""
from .mxquant import TIER, Unavailable, available, compare, line, run  # noqa: F401
