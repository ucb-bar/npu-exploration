"""(hardware, run) -> mxq: the one place a machine's arithmetic is spelled out for a Python model.

The hardware recipe (``config/hardware/*.json``) says WHICH MX-Gemmini this is; the run recipe
(``config/run/*.json``) how it is driven. mxq (``microscaling-quant/``) computes what such a machine's
matmul does, given four things: how each operand is block-quantized, the *Arithmetic* (how a product, a
lane add and a block add are rounded), the *schedule* (one accumulator format per PE lane) and the
*window* (the PE column depth). This module derives all four from the two recipes and nothing else, so
the mxquant model's two paths (``models/mxquant``: bits per kernel, perplexity per workload) run the same
numbers, and a perplexity is tied to the ``build_id`` that VERDICT was proved against.

    recipe field                              mxq argument
    ----------------------------------------  ------------------------------------------------------
    run.operand_fmt, hw mx.scaleSize          block.mxgemmini.quantize(fmt, block_size,
    run.rounding, run.scale_floor                 rounding_mode, scale_floor)
    run.scale "ocp"                               block.ocp.quantize(fmt, block_size, rounding_mode) instead:
                                                  the block max at the format max, not in [1, 2)
    hw types.meshProdPrecisionList            matmul.MXGEMMINI(prod_e, prod_m, prod_floor)   one product format
    hw types.prodFloor
    hw types.meshAccPrecisionList             schedule = [(expWidth, sigWidth - 1)] x dim
    hw array.meshRows                         window
    run.reduce                                which reducer (REDUCERS)
    run.lut.group, run.lut.fit.max_iters      block.lut.quantize(group, max_iters)            a LUT format's operands;
                                              Scheme(rows=2**group)                           MXLinear keeps 2**G tokens together
    run.vector.softmax, run.vector.rmsnorm    mxq.nn.patch(vector={...})                      not the Scheme: the vector ops
                                                                                              between the matmuls

Every knob is passed explicitly, never left to mxq's defaults: the RTL rounds operands
round-to-nearest-even since 2026-09-10 and floors the block max at FLT_EPSILON = 2^-23, while mxq's
own defaults follow an older fixture (see ``mxq/block/mxgemmini.py``).

Deliberately ignored, because none of them changes a matmul's value: ``array.tileRows``,
``array.tileColumns``, ``MxFloat.isRecoded``, ``MxFloat.pad``, ``mx.scaleSizeOut``, ``scratchpad``,
``implementation``.

Refused (``RecipeError``): a per-lane product list that is not uniform (mxq has one product format
per Arithmetic); an accumulator list whose length is not the mesh dimension. A run with a ``lut`` block
(``uses_lut``; only a codebook format, ``is_codebook``, may have one) runs through the chip's tables: ``mxq.block.lut``, the rule ``compiler/codebook.py``
also calls, so both operands (A: 2**G tokens per table, B: 2**G output channels per table) see the 16
entries the chip would. A layer's output is not requantized through a C table here (no chained stage),
and the chip's table capacity is enforced on the kernel path (and perf's as-measured model) only;
``lut_record`` says so in the record. A codebook format without a ``lut`` block is quantized straight to
its grid (LUT off), perplexity path only. Here a token or channel count that is not a multiple of 2**G gets one
last table over the leftover rows (mxq.lut), so any layer runs; the chip's loader takes whole groups only.
"""
from __future__ import annotations

from functools import partial

import models  # noqa: F401  -- puts the mxq submodule on sys.path
from config.recipe import Hardware, RecipeError, Run

#: operand format (run.operand_fmt, the compiler's spelling) -> mxq's format table key. The four CODEBOOK
#: formats travel as 4-bit indices into a table per 2**G rows / columns on this hardware (mxq.lut).
MXQ_FORMAT = {"fp8_e4m3": "MXFP8_E4M3", "fp8_e4m3_quad": "MXFP8_E4M3", "fp8_e5m2": "MXFP8_E5M2",
              "fp6_e3m2": "MXFP6_E3M2", "fp6_e2m3": "MXFP6_E2M3", "fp4_e2m1": "MXFP4"}
CODEBOOK = frozenset({"fp8_e4m3_quad", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3"})

#: How the codes are multiplied (``scheme(reduce=)``): the recipe's array; mxq's exact float64 product (the
#: format's cost alone); or fp32 inside each 32-block and the hardware's bf16 step across blocks.
REDUCERS = ("hardware", "exact", "bf16_tiles")


def mxq_format(dtype: str) -> str:
    """An operand format name -> the mxq element format it quantizes to."""
    try:
        return MXQ_FORMAT[dtype]
    except KeyError:
        raise RecipeError(f"operand format {dtype!r} has no mxq format; known: {sorted(MXQ_FORMAT)}") from None


def is_codebook(dtype: str) -> bool:
    """Does this hardware send ``dtype`` through a codebook (LUT) rather than as element codes?"""
    mxq_format(dtype)
    return dtype in CODEBOOK


def uses_lut(run: Run) -> bool:
    """Does this run send its operands through the chip's tables? Yes exactly when the run recipe has a ``lut``
    block. A codebook format (``is_codebook``) without one is quantized straight to its element grid (LUT off):
    the perplexity path can, since MXLinear quantizes in software; the chip's requantizer cannot, so the kernel
    path refuses it (``config.recipe.check``)."""
    return run.lut is not None


def quantizer(hw: Hardware, run: Run):
    """``V -> (P, X)`` for one operand, blocks along axis 0 (K): the run's format, rounding and floor; a LUT
    format then through its tables (``run.lut``: one per 2**G columns of V)."""
    from mxq import block
    if uses_lut(run):
        return partial(block.lut.quantize, fmt=mxq_format(run.operand_fmt), axis=0, block_size=hw.block,
                       rounding_mode=run.rounding, scale_floor=run.scale_floor, group=run.lut.group,
                       max_iters=run.lut.fit.max_iters)
    if run.scale == "ocp":                  # OCP placement: block max at the format max; no scale floor there
        return partial(block.ocp.quantize, fmt=mxq_format(run.operand_fmt), axis=0, block_size=hw.block,
                       rounding_mode={"rne": "even", "ties_away": "nearest"}[run.rounding])
    return partial(block.mxgemmini.quantize, fmt=mxq_format(run.operand_fmt), axis=0, block_size=hw.block,
                   rounding_mode=run.rounding, scale_floor=run.scale_floor)


def rows(run: Run) -> int:
    """Token rows one activation quantizer call must keep together: 2**G for a LUT format, else 1."""
    return 1 << run.lut.group if uses_lut(run) else 1


def vector(run: Run) -> dict | None:
    """What mxq.nn.patch takes as ``vector`` for this run: None (no vector block: as transformers computes
    them) or {"softmax": None | "bf16", "rmsnorm": None | "bf16"}."""
    return None if run.vector is None else run.vector.mxq()


def lut_record(run: Run) -> dict | None:
    """What the record and cache key say about a LUT format's tables, or None for a direct format."""
    if not uses_lut(run):
        return None
    return {"group": run.lut.group, "max_iters": run.lut.fit.max_iters, "rule": "mxq.lut",
            "tables": "A and B; no C (a layer's output is not requantized); capacity not enforced"}


def product(recipe: Hardware) -> tuple[int, int]:
    """The one product format ``(e, m)``; refuses a per-lane list that is not uniform."""
    prods = sorted({(p.e, p.m) for p in recipe.prod})
    if len(prods) != 1:
        raise RecipeError(f"{recipe.name}: mxq has one product format per Arithmetic, but "
                          f"meshProdPrecisionList has {prods}")
    return prods[0]


def schedule(recipe: Hardware) -> list[tuple[int, int]]:
    """One ``(e, m)`` per PE lane, lane = k % dim."""
    sched = [(a.e, a.m) for a in recipe.acc]
    if len(sched) != recipe.dim:
        raise RecipeError(f"{recipe.name}: {len(sched)} accumulator lanes for a {recipe.dim}-deep column")
    return sched


def datapath(recipe: Hardware):
    """``(Arithmetic, schedule, window)`` of the hardware this recipe describes."""
    from mxq import matmul
    return mxgemmini(recipe), schedule(recipe), recipe.dim


def mxgemmini(recipe: Hardware):
    """mxq's MXGEMMINI arithmetic for the recipe's product format, with its product flush
    (``types.prodFloor``: a product below 2^prodFloor is zero; null = no flush; mxq 93c7047 and later)."""
    from mxq import matmul
    pe, pm = product(recipe)
    return matmul.MXGEMMINI(pe, pm, prod_floor=recipe.prod_floor)


def shipped_datapath(recipe: Hardware):
    """``(Arithmetic, schedule, window)`` of MXQuant's published simulator on this recipe's ladder.
    This is what the informational "as shipped" line is computed with."""
    from mxq import matmul
    pe, pm = product(recipe)
    return matmul.MXQUANT(pe, pm), schedule(recipe), recipe.dim


def _bf16_tiles(recipe: Hardware):
    """The hardware's cross-block step with a perfect in-block accumulator: fp32 products and adds inside each
    block (the window is the whole block), the finished block folded into the output by MXGEMMINI's own
    ``tile_add`` (both rounded to bf16, added exactly, rounded to bf16)."""
    from mxq import matmul
    return matmul.Arithmetic("bf16_tiles", product=lambda a, b: a * b, acc_add=lambda S, p, e, m: S + p,
                             tile_add=mxgemmini(recipe).tile_add)


def mxq_config(hw: Hardware, run: Run, *, compiled: bool = False):
    """The two recipes as mxq's TorchAO config (``mxq.nn.torchao.MXQConfig``): the same Scheme as ``scheme()``,
    as plain fields, for tools that only call ``torchao.quantize_`` (Model2MLIR, Hugging Face ``TorchAoConfig``).
    Needs torchao. ``bf16_tiles`` is this repo's diagnostic reducer, not mxq's, so it is refused here."""
    from config import recipe as _recipe
    from mxq.nn.torchao import MXQConfig
    _recipe.check(hw, run, "perplexity")
    if run.reduce == "bf16_tiles":
        raise RecipeError(f"run {run.name}: reduce bf16_tiles is npu-exploration's diagnostic; MXQConfig has "
                          "hardware and exact")
    return MXQConfig(fmt=mxq_format(run.operand_fmt), rounding_mode=run.rounding, scale_floor=run.scale_floor,
                     block_size=hw.block, prod=list(product(hw)), prod_floor=hw.prod_floor,
                     ladder=[list(e) for e in schedule(hw)], size=hw.dim, reduce=run.reduce, compiled=compiled,
                     name=hw.name if run.reduce == "hardware" else f"{hw.name}/{run.reduce}",
                     **({"lut": {"group": run.lut.group, "max_iters": run.lut.fit.max_iters}}
                        if uses_lut(run) else {}))


def scheme(recipe: Hardware, run: Run, *, compiled: bool = False):
    """The two recipes as one mxq ``Scheme``: quantizer for both operands, and how the codes are multiplied.
    A codebook format runs through the chip's tables, see ``quantizer``. ``run.reduce`` is one of
    ``REDUCERS``: "hardware" is the recipe's array (its Arithmetic, schedule and window; ``compiled=True``
    fuses it through torch.compile, GPU, bit-identical, 5-7x faster per layer); "exact" is mxq's
    ``fp64_accum``; "bf16_tiles" is fp32 inside each block, bf16 across (``compiled`` applies to it too). The
    three share the quantizers, so their differences are the multiply's alone."""
    from mxq import Scheme, fp64_accum, matmul
    q = quantizer(recipe, run)
    reduce = run.reduce
    if reduce == "hardware":
        arith, sched, window = datapath(recipe)
        if compiled:
            arith = matmul.compiled(arith)
        r = partial(matmul.systolic, arith=arith, schedule=sched, size=window, block_size=recipe.block)
    elif reduce == "exact":
        r = partial(fp64_accum, block_size=recipe.block)
    elif reduce == "bf16_tiles":
        arith = _bf16_tiles(recipe)
        if compiled:                        # gated bit-identical to eager on the GPU (0/984576 differ, 2026-09-28)
            arith = matmul.compiled(arith)
        r = partial(matmul.systolic, arith=arith, schedule=[(8, 7)] * recipe.block, size=recipe.block,
                    block_size=recipe.block)
    else:
        raise RecipeError(f"reduce {reduce!r}; choose from {', '.join(REDUCERS)}")
    return Scheme(recipe.name if reduce == "hardware" else f"{recipe.name}/{reduce}", a=q, b=q, reduce=r,
                  rows=rows(run))
