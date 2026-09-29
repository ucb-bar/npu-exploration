"""Recipe -> mxq: the one place a machine's arithmetic is spelled out for a Python model.

A recipe (``config/recipes/*.json``) says WHICH MX-Gemmini this is. mxq (``microscaling-quant/``)
computes what such a machine's matmul does, given four things: how each operand is block-quantized,
the *Arithmetic* (how a product, a lane add and a block add are rounded), the *schedule* (one
accumulator format per PE lane) and the *window* (the PE column depth). This module derives all four
from the recipe and nothing else, so the mxquant model's two paths (``models/mxquant``: bits per kernel,
perplexity per workload) run the same numbers, and a perplexity is tied to the ``build_id`` that VERDICT
was proved against.

    recipe field                              mxq argument
    ----------------------------------------  ------------------------------------------------------
    runtime.operand_fmt, software.block       block.mxgemmini.quantize(fmt, block_size,
                                                  rounding_mode="rne", scale_floor=HARDWARE_FLOOR)
    types.meshProdPrecisionList               matmul.MXGEMMINI(prod_e, prod_m)   one product format
    types.meshAccPrecisionList                schedule = [(expWidth, sigWidth - 1)] x dim
    array.meshRows                            window

Both operand knobs are passed explicitly, never left to mxq's defaults: the RTL rounds operands
round-to-nearest-even since 2026-09-10 and floors the block max at FLT_EPSILON = 2^-23, while mxq's
own defaults follow an older fixture (see ``mxq/block/mxgemmini.py``).

Deliberately ignored, because none of them changes a matmul's value: ``array.tileRows``,
``array.tileColumns``, ``MxFloat.isRecoded``, ``MxFloat.pad``, ``mx.scaleSizeOut``,
``software.target_code_exp``, ``software.seam``, ``software.intermediate_dtype``,
``runtime.out_dtype``, ``supported_backends``.

Refused (``RecipeError``): a per-lane product list that is not uniform (mxq has one product format
per Arithmetic); an accumulator list whose length is not the mesh dimension. A codebook (LUT) format
is not refused: mxq has no codebooks, so a Scheme quantizes it on the format's full element grid
(``is_codebook`` says which formats the hardware sends through a table; the bit path grades those
through the wire operands the compiler emitted, ``models/mxquant/kernel.py`` ``_device_operands``).
"""
from __future__ import annotations

import os
from functools import partial

import models  # noqa: F401  -- puts the mxq submodule on sys.path
from config.recipe import Recipe, RecipeError

#: operand format spelled the recipe's way -> mxq's format table key
FORMAT = {"fp8": "MXFP8_E4M3", "fp6": "MXFP6_E3M2", "fp4": "MXFP4"}
#: ... and spelled the compiler's way (--dtype). The four CODEBOOK formats travel as 4-bit indices into a
#: per-row-pair table on this hardware (compiler/codebook.py); mxq quantizes them on their full element grid.
MXQ_FORMAT = {"fp8_e4m3": "MXFP8_E4M3", "fp8_e4m3_quad": "MXFP8_E4M3", "fp8_e5m2": "MXFP8_E5M2",
              "fp6_e3m2": "MXFP6_E3M2", "fp6_e2m3": "MXFP6_E2M3", "fp4_e2m1": "MXFP4"}
CODEBOOK = frozenset({"fp8_e4m3_quad", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3"})
_RECIPE_DTYPE = {"fp8": "fp8_e4m3", "fp6": "fp6_e3m2", "fp4": "fp4_e2m1"}

#: The hardware's operand rounding since 2026-09-10 (mx_fp_math.h, RNE for every format).
ROUNDING = "rne"
#: How the codes are multiplied (``scheme(reduce=)``): the recipe's array; mxq's exact float64 product (the
#: format's cost alone); or fp32 inside each 32-block and the hardware's bf16 step across blocks.
REDUCERS = ("hardware", "exact", "bf16_tiles")


def format_name(recipe: Recipe) -> str:
    try:
        return FORMAT[recipe.operand_fmt]
    except KeyError:
        raise RecipeError(f"{recipe.name}: operand_fmt {recipe.operand_fmt!r} has no mxq format; "
                          f"known: {sorted(FORMAT)}") from None


def mxq_format(dtype: str) -> str:
    """A --dtype name -> the mxq element format it quantizes to."""
    try:
        return MXQ_FORMAT[dtype]
    except KeyError:
        raise RecipeError(f"operand format {dtype!r} has no mxq format; known: {sorted(MXQ_FORMAT)}") from None


def is_codebook(dtype: str) -> bool:
    """Does this hardware send ``dtype`` through a codebook (LUT) rather than as element codes?"""
    mxq_format(dtype)
    return dtype in CODEBOOK


def recipe_dtype(recipe: Recipe) -> str:
    """The recipe's default operand format, spelled the compiler's way."""
    try:
        return _RECIPE_DTYPE[recipe.operand_fmt]
    except KeyError:
        raise RecipeError(f"{recipe.name}: operand_fmt {recipe.operand_fmt!r} has no format; known: {sorted(_RECIPE_DTYPE)}") from None


def scale_floor_default() -> float:
    from mxq import scale_factor
    return scale_factor.HARDWARE_FLOOR


def quantizer(recipe: Recipe, *, fmt: str | None = None, rounding_mode: str = ROUNDING,
              scale_floor: float | None = None):
    """``V -> (P, X)`` for one operand, blocks along axis 0 (K), in the hardware's convention.
    ``fmt`` is an mxq format name (default: the recipe's operand format)."""
    from mxq import block
    return partial(block.mxgemmini.quantize, fmt=fmt or format_name(recipe), axis=0, block_size=recipe.block,
                   rounding_mode=rounding_mode,
                   scale_floor=scale_floor_default() if scale_floor is None else scale_floor)


def product(recipe: Recipe) -> tuple[int, int]:
    """The one product format ``(e, m)``; refuses a per-lane list that is not uniform."""
    prods = sorted({(p.e, p.m) for p in recipe.prod})
    if len(prods) != 1:
        raise RecipeError(f"{recipe.name}: mxq has one product format per Arithmetic, but "
                          f"meshProdPrecisionList has {prods}")
    return prods[0]


def schedule(recipe: Recipe) -> list[tuple[int, int]]:
    """One ``(e, m)`` per PE lane, lane = k % dim."""
    sched = [(a.e, a.m) for a in recipe.acc]
    if len(sched) != recipe.dim:
        raise RecipeError(f"{recipe.name}: {len(sched)} accumulator lanes for a {recipe.dim}-deep column")
    return sched


def datapath(recipe: Recipe):
    """``(Arithmetic, schedule, window)`` of the hardware this recipe describes."""
    from mxq import matmul
    pe, pm = product(recipe)
    return mxgemmini(pe, pm), schedule(recipe), recipe.dim


#: MxFPMul PROD_FLOOR: the hardware flushes a product below 2^-16 to zero (rtl_exact.rtl_datapath.PROD_FLOOR).
PROD_FLOOR = -16


def mxgemmini(pe: int, pm: int, prod_floor: int | None = PROD_FLOOR):
    """mxq's MXGEMMINI arithmetic for one product format, with the hardware's product flush
    (mxq 93c7047 and later). ``MXG_PROD_FLOOR=none`` switches the flush off, Nicolas's A/B control
    for the FPGA runs; a hidden knob until the recipe carries it."""
    from mxq import matmul
    if os.environ.get("MXG_PROD_FLOOR") == "none":
        prod_floor = None
    return matmul.MXGEMMINI(pe, pm, prod_floor=prod_floor)


def shipped_datapath(recipe: Recipe):
    """``(Arithmetic, schedule, window)`` of MXQuant's published simulator on this recipe's ladder.
    This is what the informational "as shipped" line is computed with."""
    from mxq import matmul
    pe, pm = product(recipe)
    return matmul.MXQUANT(pe, pm), schedule(recipe), recipe.dim


def _bf16_tiles(recipe: Recipe):
    """The hardware's cross-block step with a perfect in-block accumulator: fp32 products and adds inside each
    block (the window is the whole block), the finished block folded into the output by MXGEMMINI's own
    ``tile_add`` (both rounded to bf16, added exactly, rounded to bf16)."""
    from mxq import matmul
    hw = mxgemmini(*product(recipe))
    return matmul.Arithmetic("bf16_tiles", product=lambda a, b: a * b, acc_add=lambda S, p, e, m: S + p,
                             tile_add=hw.tile_add)


def scheme(recipe: Recipe, *, dtype: str | None = None, compiled: bool = False, rounding_mode: str = ROUNDING,
           scale_floor: float | None = None, reduce: str = "hardware"):
    """The recipe as one mxq ``Scheme``: quantizer for both operands, and how the codes are multiplied.
    ``dtype`` is the operand format (default: the recipe's); a codebook format runs on its full element grid,
    see ``is_codebook``. ``reduce`` is one of ``REDUCERS``: "hardware" is the recipe's array (its Arithmetic,
    schedule and window; ``compiled=True`` fuses it through torch.compile, GPU, bit-identical, 5-7x faster per
    layer); "exact" is mxq's ``fp64_accum``; "bf16_tiles" is fp32 inside each block, bf16 across (``compiled``
    applies to it too). The three share the quantizers, so their differences are the multiply's alone."""
    from mxq import Scheme, fp64_accum, matmul
    q = quantizer(recipe, fmt=mxq_format(dtype) if dtype else None, rounding_mode=rounding_mode,
                  scale_floor=scale_floor)
    if reduce == "hardware":
        arith, sched, window = datapath(recipe)
        if compiled:
            arith = matmul.compiled(arith)
        r = partial(matmul.systolic, arith=arith, schedule=sched, window=window, block_size=recipe.block)
    elif reduce == "exact":
        r = partial(fp64_accum, block_size=recipe.block)
    elif reduce == "bf16_tiles":
        arith = _bf16_tiles(recipe)
        if compiled:                        # gated bit-identical to eager on the GPU (0/984576 differ, 2026-09-28)
            arith = matmul.compiled(arith)
        r = partial(matmul.systolic, arith=arith, schedule=[(8, 7)] * recipe.block, window=recipe.block,
                    block_size=recipe.block)
    else:
        raise RecipeError(f"reduce {reduce!r}; choose from {', '.join(REDUCERS)}")
    return Scheme(recipe.name if reduce == "hardware" else f"{recipe.name}/{reduce}", a=q, b=q, reduce=r)
