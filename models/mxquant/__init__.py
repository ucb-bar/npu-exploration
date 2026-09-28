"""The mxquant model: the recipe's arithmetic, on mxq, reached two ways.

    run(spec, recipe, dtype=, edges=)   the bit path (kernel.py): one kernel's exact output bits, graded
                                        against spike -- VERDICT is spike bits == these bits
    evaluate(workload, recipe, dtype=)  the perplexity path (workload.py): a whole language model with the
                                        same Scheme in its linear layers, measured on WikiText-2 (GPU, minutes)

Both build the Scheme with config/scheme.py from the recipe alone; tests/selftest_workload.py holds them to
the same bits on a linear layer. workloads.py registers what evaluate can run; rules.py names which layers
get the Scheme; block.py is MXQuant's block-quantizer API computed by mxq, shared with the wire-operand
encoder in app/mxq_golden.py on purpose: the operands the ELF carries and the operands the model
multiplies must be the same bytes. `python -m models.mxquant` runs the perplexity path alone.
"""
from .kernel import TIER, Unavailable, available, compare, line, run  # noqa: F401
from .workload import evaluate  # noqa: F401
from .workloads import WORKLOADS, Workload  # noqa: F401
