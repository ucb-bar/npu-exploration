"""One YAML per experiment: ``hardware`` (read by every step) and one section per step.

    from config.npu_config import load
    cfg = load("config/baseline.yaml")
    cfg["capture"]["shape"], cfg["hardware"]["mx"]["block"], cfg["quantize"]["default"]

Every section may be omitted and takes the defaults below. Unknown keys are refused, so a typo is
an error rather than a silently ignored setting. Precision lists are written as runs,
``"4,5 x8 + 4,6 x2 + 8,8"`` = eight (exp 4, sig 5), two (4, 6), one (8, 8).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "capture": {"shape": [64, 64], "seed": 0, "input": None},
    "hardware": {
        "mesh": {"rows": 16, "cols": 16},
        "mx": {"block": 32, "formats": ["fp8_e4m3"]},
        "scratchpad": {"banks": 4, "rows": 4096},
        "precision": {"prod": "4,4 x16", "acc": "4,5 x8 + 4,6 x2 + 4,7 x5 + 8,8"},
        "clock_ns": 2.0,
    },
    "quantize": {"default": "fp8_e4m3", "types": {}, "layers": {}},
    "compile": {"target": "spike", "vpu": ["add", "elementwise", "reduce", "softmax", "unary"]},
}
# sections whose keys are names chosen by the user, not fixed settings
_FREE = {("quantize", "types"), ("quantize", "layers")}


class ConfigError(ValueError):
    pass


def load(path: str | Path | None) -> dict:
    """The YAML at ``path`` (or only the defaults when ``None``), merged over ``DEFAULTS``."""
    raw = {} if path is None else (yaml.safe_load(Path(path).read_text()) or {})
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping with {', '.join(DEFAULTS)}")
    cfg = _merge(copy.deepcopy(DEFAULTS), raw, ())
    hw = cfg["hardware"]
    hw["precision"] = {k: parse_precision(v) for k, v in hw["precision"].items()}
    if len(hw["precision"]["prod"]) != hw["mesh"]["rows"]:
        raise ConfigError("hardware.precision.prod: one entry per mesh row")
    return cfg


def _merge(base: dict, over: dict, where: tuple) -> dict:
    for k, v in over.items():
        here = (*where, k)
        if k not in base:
            raise ConfigError(f"unknown key {'.'.join(here)}; known: {', '.join(base)}")
        if isinstance(base[k], dict) and here not in _FREE:
            if not isinstance(v, dict):
                raise ConfigError(f"{'.'.join(here)}: expected a mapping")
            _merge(base[k], v, here)
        else:
            base[k] = v
    return base


def parse_precision(spec) -> list[list[int]]:
    """``"4,5 x8 + 8,8"`` -> ``[[4, 5]] * 8 + [[8, 8]]``; an explicit list passes through."""
    if isinstance(spec, list):
        return [[int(x) for x in e] for e in spec]
    out = []
    for run in str(spec).split("+"):
        pair, _, count = run.strip().partition("x")
        exp, sig = (int(x) for x in pair.split(","))
        out += [[exp, sig]] * int(count or 1)
    return out
