"""The recipe pair each operand format runs on, for tests that drive the compiler directly.

A LUT format needs a build whose LUT unit serves it (config.recipe.LUT_SERVES) and its run recipe's
``lut`` block; nothing here invents a setting -- every one is read from config/hardware/ and config/run/.
"""
from __future__ import annotations

from config.recipe import load_hardware, load_run, lut_settings

#: operand format -> (hardware recipe, run recipe). A LUT format's build is one whose mx.lut serves it at
#: the compiler's entry width; its run recipe is config/run/<format>.json.
PAIRS = {
    "fp8_e4m3": ("baseline", "default"),
    "fp4_e2m1": ("baseline", "fp4_e2m1"),
    "fp6_e3m2": ("baseline", "fp6_e3m2"),
    "fp6_e2m3": ("lut_fp6e2m3", "fp6_e2m3"),
    "fp8_e5m2": ("lut_fp8e5m2", "fp8_e5m2"),
    "fp8_e4m3_quad": ("lut_fp8e4m3", "fp8_e4m3_quad"),
}


def recipes(fmt: str):
    """``(Hardware, Run)`` for ``fmt``."""
    hw, run = PAIRS[fmt]
    return load_hardware(hw), load_run(run)


def settings(fmt: str):
    """The compiler's codebook settings for ``fmt`` (None for a direct format), from its two recipes."""
    return lut_settings(*recipes(fmt))
