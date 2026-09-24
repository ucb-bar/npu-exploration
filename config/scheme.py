"""Recipe -> mxq: the one place a machine's arithmetic is spelled out for a Python model.

A recipe (``config/recipes/*.json``) says WHICH MX-Gemmini this is. mxq (``microscaling-quant/``)
computes what such a machine's matmul does, given four things: how each operand is block-quantized,
the *Arithmetic* (how a product, a lane add and a block add are rounded), the *schedule* (one
accumulator format per PE lane) and the *window* (the PE column depth). This module derives all four
from the recipe and nothing else, so the mxquant model (``models/mxquant``) and the accuracy model
(``models/accuracy``) run the same numbers, and a perplexity is tied to the ``build_id`` that VERDICT
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
per Arithmetic); an accumulator list whose length is not the mesh dimension; and, for a MODEL-LEVEL
Scheme only, a codebook (LUT) operand path -- mxq has no codebooks, so the accuracy model cannot run
those formats. The mxquant model still grades them, through the wire operands the compiler emitted.
"""
from __future__ import annotations

from functools import partial

import models  # noqa: F401  -- puts the mxq submodule on sys.path
from config.recipe import Recipe, RecipeError

#: operand format spelled the recipe's way -> mxq's format table key
FORMAT = {"fp8": "MXFP8_E4M3", "fp6": "MXFP6_E3M2", "fp4": "MXFP4"}

#: The hardware's operand rounding since 2026-09-10 (mx_fp_math.h, RNE for every format).
ROUNDING = "rne"


def format_name(recipe: Recipe) -> str:
    try:
        return FORMAT[recipe.operand_fmt]
    except KeyError:
        raise RecipeError(f"{recipe.name}: operand_fmt {recipe.operand_fmt!r} has no mxq format; "
                          f"known: {sorted(FORMAT)}") from None


def scale_floor_default() -> float:
    from mxq import scale_factor
    return scale_factor.HARDWARE_FLOOR


def quantizer(recipe: Recipe, *, rounding_mode: str = ROUNDING, scale_floor: float | None = None):
    """``V -> (P, X)`` for one operand, blocks along axis 0 (K), in the hardware's convention."""
    from mxq import block
    return partial(block.mxgemmini.quantize, fmt=format_name(recipe), axis=0, block_size=recipe.block,
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
    return matmul.MXGEMMINI(pe, pm), schedule(recipe), recipe.dim


def shipped_datapath(recipe: Recipe):
    """``(Arithmetic, schedule, window)`` of MXQuant's published simulator on this recipe's ladder.
    This is what the informational "as shipped" line is computed with."""
    from mxq import matmul
    pe, pm = product(recipe)
    return matmul.MXQUANT(pe, pm), schedule(recipe), recipe.dim


def refuse_codebooks(recipe: Recipe) -> None:
    """A model-level Scheme cannot run a codebook (LUT) operand path: mxq has no codebooks."""
    from app import mxformats
    raw = recipe.raw
    if raw.get("runtime", {}).get("use_lut") or raw.get("mx", {}).get("enable_lut"):
        raise RecipeError(f"{recipe.name}: use_lut/enable_lut is set; mxq has no codebooks, so this recipe "
                          "cannot run at model level (the mxquant model still grades it via wire operands)")
    f = mxformats.get(recipe.operand_mlir_dtype, where="config.scheme", proven_only=False)
    if f.lut:
        raise RecipeError(f"{recipe.name}: operand format {f.name} is codebook-indexed on this hardware; "
                          "mxq has no codebooks, so it cannot run at model level")


def scheme(recipe: Recipe, *, compiled: bool = False, rounding_mode: str = ROUNDING,
           scale_floor: float | None = None):
    """The recipe as one mxq ``Scheme``: quantizer for both operands, the hardware Arithmetic, the
    recipe's schedule and window. ``compiled=True`` fuses the arithmetic through torch.compile
    (GPU; bit-identical, 5-7x faster per layer)."""
    from mxq import Scheme, matmul
    refuse_codebooks(recipe)
    q = quantizer(recipe, rounding_mode=rounding_mode, scale_floor=scale_floor)
    arith, sched, window = datapath(recipe)
    if compiled:
        arith = matmul.compiled(arith)
    return Scheme(recipe.name, a=q, b=q,
                  reduce=partial(matmul.systolic, arith=arith, schedule=sched, window=window,
                                 block_size=recipe.block))
