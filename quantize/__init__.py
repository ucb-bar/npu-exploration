"""quantize: a captured model's fp32 weights -> MX codes and E8M0 scales (mxq).

    from quantize import run
    run(Path("build/MLP/capture"), cfg, Path("build/MLP/quantize"))

Reads ``capture/weights.safetensors`` and ``capture/layers.json``, and the config's ``hardware`` and
``quantize`` sections. Writes ``out/weights.mx.safetensors``, one entry pair per matmul weight:

    <param>.codes   uint8  [K][N]   B = W^T of y = x W^T; 4-bit formats pack two columns per
                                    byte, [K][N/2], even column in the low nibble
    <param>.scales  uint8  [K/32][N]  E8M0, one per 32 values down K
    metadata        {<param>: format, "__default__": quantize.default}
                    (the default is the format of matmuls without a weight, e.g. attention)

That is the byte layout the compiler's checkpoint packer emits, so compile embeds these bytes in
place of packing the fp32 weight itself. A weight's format is the first match of
``quantize.layers`` (module name, globs allowed), ``quantize.types`` (module class), then
``quantize.default``. With ``debug=True`` also ``out/debug/``: errors.json (per weight, relative
error of the decoded weight vs fp32) and config.yaml (the sections this step read).
"""
from __future__ import annotations

import fnmatch
import json
import shutil
from pathlib import Path

import numpy as np

MATMUL_TYPES = {"Linear"}       # module classes whose 2-D weight is a matmul operand (y = x W^T)


def pick_format(module: str, cls: str, q: dict) -> str:
    for pattern, fmt in q["layers"].items():
        if fnmatch.fnmatchcase(module, pattern):
            return fmt
    return q["types"].get(cls, q["default"])


def run(capture: Path, cfg: dict, out: Path, *, debug: bool = False) -> Path:
    from safetensors.numpy import load_file, save_file
    from compiler import formats
    from compiler.operands import quantize_operand, wire_to_px

    hw, q = cfg["hardware"], cfg["quantize"]
    if hw["mx"]["block"] != 32:
        raise ValueError(f"hardware.mx.block {hw['mx']['block']}: the quantizer blocks by 32")
    layers = json.loads((Path(capture) / "layers.json").read_text())
    unknown = [p for p in q["layers"] if not any(fnmatch.fnmatchcase(m, p) for m in layers)]
    if unknown:
        raise ValueError(f"quantize.layers {unknown} match no module; modules: {sorted(layers)}")

    tensors, meta, errors = {}, {"__default__": q["default"]}, {}
    for name, w in load_file(str(Path(capture) / "weights.safetensors")).items():
        module, _, leaf = name.rpartition(".")
        cls = layers.get(module)
        if leaf != "weight" or cls not in MATMUL_TYPES or w.ndim != 2:
            continue
        fmt = formats.get(pick_format(module, cls, q), where=f"quantize {name}", proven_only=False)
        if fmt.name not in hw["mx"]["formats"]:
            raise ValueError(f"{name}: {fmt.name} is not in hardware.mx.formats {hw['mx']['formats']}")
        if fmt.lut:
            raise ValueError(f"{name}: codebook format {fmt.name} needs LUT settings, not supported yet")
        B = np.ascontiguousarray(w.T, np.float32)                  # [K][N]
        codes, scales, _ = quantize_operand(B, side="b", dtype=fmt.name)
        tensors[f"{name}.codes"] = np.ascontiguousarray(codes, np.uint8)
        tensors[f"{name}.scales"] = np.ascontiguousarray(scales, np.uint8)
        meta[name] = fmt.name
        if debug:
            P, X = wire_to_px(codes, scales, side="b", dtype=fmt.name)
            deq = P.numpy() * np.repeat(X.numpy(), 32, axis=0)
            errors[name] = {"format": fmt.name, "shape_kn": list(B.shape),
                            "rel_error": float(np.linalg.norm(deq - B) / np.linalg.norm(B))}
    if not tensors:
        raise ValueError(f"{capture}: no matmul weights (modules of {sorted(MATMUL_TYPES)})")

    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    save_file(tensors, str(out / "weights.mx.safetensors"), metadata=meta)
    if debug:
        import yaml
        (out / "debug").mkdir()
        (out / "debug" / "errors.json").write_text(json.dumps(errors, indent=1) + "\n")
        (out / "debug" / "config.yaml").write_text(
            yaml.safe_dump({"hardware": hw, "quantize": q}, sort_keys=False))
    return out
