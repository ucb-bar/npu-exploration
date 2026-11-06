"""capture: a PyTorch model -> its MLIR graph, weights and example input (model2MLIR).

    from capture import run
    run("examples/mlp.py:MLP", Path("build/MLP/capture"), m=64, k=64)

Writes, into ``out``, only what later steps read:

    model.mlir                           the graph (linalg on tensors)
    weights.safetensors(.manifest.json)  fp32 weights, and which graph argument each one is
    inputs.npz, input_order.json         the example input, embedded in the ELF by compile
    extra.npz                            non-weight tensors (buffers, e.g. RoPE tables), if any
    layers.json                          {module name: class name} for modules that own weights
    golden.npy                           fp32 output of the model on that input (read by evaluate)

With ``debug=True`` also ``out/debug/``: capture_receipt.json (hashes of the files above) and
region_goldens.npz (fp32 tensors at every module boundary).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
M2M = REPO / "compiler" / "model2mlir"

OUTPUTS = ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
           "inputs.npz", "input_order.json", "extra.npz", "golden.npy", "layers.json")
DEBUG = ("capture_receipt.json", "region_goldens.npz")


def load_model(ref: str, *, seed: int = 0):
    """``FILE.py:Name`` -> an nn.Module. A class is instantiated with no arguments after
    ``torch.manual_seed(seed)``; an instance is used as is."""
    import torch
    file, _, attr = ref.rpartition(":")
    path = Path(file)
    if not file or not attr or not path.is_file():
        raise ValueError(f"model must be an existing FILE.py:Name, got {ref!r}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    obj = getattr(mod, attr, None)
    torch.manual_seed(seed)
    module = obj() if isinstance(obj, type) else obj
    if not isinstance(module, torch.nn.Module):
        raise ValueError(f"{ref} is not an nn.Module (got {type(module).__name__})")
    return module.eval()


def run(ref: str, out: Path, *, m: int, k: int, seed: int = 0, input_npy: Path | None = None,
        debug: bool = False) -> Path:
    """Capture ``ref`` on a ``[m][k]`` input (``randn``, or ``input_npy``) into ``out``."""
    import numpy as np
    import torch
    if str(M2M) not in sys.path:
        sys.path.insert(0, str(M2M))
    from m2m.capture.bundle import write_bundle
    from m2m.capture.provenance import write_capture_receipt

    module = load_model(ref, seed=seed)       # randn below continues this seed, as npu.py does
    x =(torch.from_numpy(np.ascontiguousarray(np.load(input_npy), np.float32))
         if input_npy is not None else torch.randn(m, k))

    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    write_bundle(module, (x,), out, capture_regions=debug)
    write_capture_receipt(out, source_path=ref.rpartition(":")[0])
    layers = {n: type(mod).__name__ for n, mod in module.named_modules()
              if n and any(True for _ in mod.parameters(recurse=False))}
    (out / "layers.json").write_text(json.dumps(layers, indent=1) + "\n")

    dbg = out / "debug"
    for p in sorted(out.iterdir()):
        if p.name in OUTPUTS or p == dbg:
            continue
        if debug and p.name in DEBUG:
            dbg.mkdir(exist_ok=True)
            p.rename(dbg / p.name)
        elif p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    extra = out / "extra.npz"
    if extra.is_file() and not np.load(extra).files:
        extra.unlink()
    return out
