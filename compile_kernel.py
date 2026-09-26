"""PyTorch kernel -> ELF for MX-Gemmini. Compile mode: no spike run, no grade.

One command turns a kernel and a hardware recipe into a directory the RTL team can run as is:

    source scripts/env.sh
    .venv/bin/python compile_kernel.py --list
    .venv/bin/python compile_kernel.py --kernel mlp3                            # out/compile/mlp3/spike/
    .venv/bin/python compile_kernel.py --kernel attention --target mx_rocket    # the RTL build
    .venv/bin/python compile_kernel.py --kernel linear --config wide_acc --dtype fp4_e2m1 --out /tmp/lin

The directory holds ``mx_gemmini_rocket.elf`` and its ``main.c``, ``command_buffer.json`` (what the
backend emitted from, minus the byte arrays), ``operands.npz`` (the input and every wire operand),
``expected.npy`` (the bits the ELF must print: the mxquant model of this recipe, bit-exact against
spike on every graded kernel) and ``manifest.json`` (recipe, format, lowering, target, tool and
repo heads, hashes). ``--target`` changes only the build define: the C is byte-identical.

Refused rather than approximated: a kernel the lowering can only run one ELF per matmul (a host
stage with a Python function), a graph kernel in a format other than fp8_e4m3 (the host-side
re-quantizer in mx_host.h encodes E4M3 only), a recipe whose mesh the backend does not plan for,
shape violations, and a recipe the mxquant model cannot follow. Exit 0 compiled, 1 the build
failed, 2 refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The backend package registers itself with merlin whenever merlin is importable. This compiler is
# built without merlin, so the process refuses to load it rather than depending on the environment.
sys.modules.setdefault("merlin", None)

from compiler.lower import DEFAULT_DTYPE, lower, wire_paths  # noqa: E402

TARGETS = ("spike", "mx_rocket")
GRAPH_DTYPE = "fp8_e4m3"        #: the only format mx_host.h can re-quantize on the host


def compile(spec, recipe, *, dtype: str = DEFAULT_DTYPE, target: str = "spike", out: Path,
            allow_lossy_chain: bool = False, tel=None) -> dict:
    """Lower ``spec`` on ``recipe``'s machine, build the ELF into ``out``, write the expected bits
    and the manifest. Returns the manifest. Raises ``ValueError`` for anything refused."""
    from grade.telemetry import Telemetry
    tel = tel or Telemetry()
    wire_paths()
    import numpy as np
    import backend as mx
    from backend import mxgemm_emit
    from config.recipe import RecipeError
    from models import mxquant, mxq_commit

    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
    if recipe.dim != mxgemm_emit.DEFAULT_GEOMETRY["dim"]:
        raise ValueError(f"recipe dim={recipe.dim} but the backend plans for "
                         f"{mxgemm_emit.DEFAULT_GEOMETRY['dim']} (mxgemm_emit.DEFAULT_GEOMETRY)")
    tel.log("recipe", f"{recipe.describe()}   build_id={recipe.build_id()}")
    tel.log("kernel", f"{spec.describe()}   ({len(spec.stages)} stage"
                      f"{'s' if len(spec.stages) != 1 else ''})")
    errs = spec.validate(dim=recipe.dim, block=recipe.block)
    if errs:
        raise ValueError(f"{spec.name}: {len(errs)} shape violation(s):\n  " + "\n  ".join(errs))

    low = lower(spec, dtype, allow_lossy_chain=allow_lossy_chain,
                warn=lambda m: tel.log("warning", m))
    if low.kind == "per_stage":
        host = [st.name for st in spec.stages if not st.on_mesh and not st.emittable]
        raise ValueError(
            f"{spec.name} cannot be compiled ahead of time: host stage(s) {host} carry a Python "
            "function, not an op the emitter knows (app/mxhost.OPS), so the only lowering is one "
            "ELF per matmul, each fed by the previous run. run_kernel.py drives that path.")
    if low.kind == "graph" and dtype != GRAPH_DTYPE:
        raise ValueError(
            f"{spec.name} lowers as a graph, and a graph re-quantizes every intermediate on the "
            f"host with mx_host.h, which encodes {GRAPH_DTYPE} only; {dtype} is refused")
    cb = low.cb
    if low.kind == "graph":
        n_mesh = sum(r["where"] == "mesh" for r in low.stages)
        tel.log("lower", f"{len(low.stages)} step(s) -> ONE command buffer via the GRAPH path "
                         f"({n_mesh} mesh, {len(low.stages) - n_mesh} host, "
                         f"{len(cb['graph_operands'])} baked operands)")
    else:
        tel.log("lower", f"{len(low.stages)} stage(s) -> ONE command buffer "
                         f"({len(cb['commands'])} commands, {len(cb['tensors'])} leaf tensors)")

    # The expected bits come before the build: a kernel the model cannot follow is refused whole.
    try:
        ref = mxquant.run(spec, recipe, dtype=dtype, edges=low.edges, shipped=False)
    except (mxquant.Unavailable, RecipeError) as exc:
        raise ValueError(f"no expected bits for {spec.name} on {recipe.name}: {exc}") from exc

    gcc = mx.runner.gcc_path()                      # MxRunnerError when the toolchain is missing
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    elf = mx.compile_command_buffer(cb, out, target=target)
    main_c = out / "main.c"
    tel.log("compile", f"{low.kind} -> {elf} ({elf.stat().st_size} B)   target {target}")

    y = np.asarray(ref["y"], np.float32)
    np.save(out / "expected.npy", y)
    tel.log("expected", f"{ref['tier']} {tuple(y.shape)} -> expected.npy")

    (out / "command_buffer.json").write_text(json.dumps(
        {k: v for k, v in cb.items() if k not in ("mx_operands", "graph_operands", "graph_consts")},
        indent=1, default=_json_default) + "\n")
    saved = {"x": spec.x.numpy()}
    if low.kind == "graph":
        for name, ops in cb["graph_operands"].items():
            base, how = name.split(":")
            for key, val in ops.items():
                saved[f"{base}_{how.replace('.', '')}_{key}"] = np.asarray(val)
        for name, val in cb["graph_consts"].items():
            saved[f"{name}_f32"] = np.asarray(val, np.float32)
    else:
        for i, bundle in enumerate(cb["mx_operands"]):
            for key, val in bundle.items():
                saved[f"s{i}_{key}"] = np.asarray(val)
    np.savez_compressed(out / "operands.npz", **saved)

    mesh = spec.mesh_stages
    mnk = spec.stage_mnk()
    manifest = {
        "kernel": spec.name,
        "lowering": low.kind, "target": target, "dtype": dtype,
        "intermediate_dtype": (cb["tensors"] and next(iter(cb["tensors"].values()))["dtype"]
                               if low.kind == "fused" and len(spec.stages) > 1 else None),
        "m": spec.m,
        "dims": [mnk[mesh[0].name][1]] + [mnk[s.name][2] for s in mesh],
        "stages": low.stages,
        "edges": {k: v["via"] for k, v in low.edges.items()},
        "recipe": {"name": recipe.name, "build_id": recipe.build_id(), "path": str(recipe.path),
                   "hardware": recipe.hardware()},
        "repo_head": _git_head(REPO), "mxq_head": mxq_commit(), "gcc": str(gcc),
        "elf": {"path": str(elf), "bytes": elf.stat().st_size, "sha256": _sha256(elf)},
        "main_c_sha256": _sha256(main_c),
        "expected": {"file": "expected.npy", "shape": list(y.shape), "tier": ref["tier"],
                     "model": ref["model"]},
        "files": sorted([p.name for p in out.iterdir()] + ["manifest.json"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=_json_default) + "\n")
    tel.log("manifest", f"{out / 'manifest.json'}")
    return manifest


def _json_default(v):
    import numpy as np
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    from config.recipe import RecipeError, list_recipes
    from config.recipe import load as load_recipe
    from grade.telemetry import Telemetry
    from kernels.registry import build, list_kernels

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list kernels, recipes and targets, then exit")
    ap.add_argument("--kernel", default="linear", help="kernel name (see --list)")
    ap.add_argument("--config", default="baseline",
                    help="hardware recipe: a name in config/recipes/ or a path to a .json")
    ap.add_argument("--dtype", default=DEFAULT_DTYPE, help="MX operand format (app/mxformats.py)")
    ap.add_argument("--m", type=int, default=64, help="batch rows")
    ap.add_argument("--k", type=int, default=64, help="in_features")
    ap.add_argument("--h", type=int, default=64, help="hidden width (chained kernels)")
    ap.add_argument("--n", type=int, default=64, help="out_features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-lossy-chain", action="store_true",
                    help="chain a format app/mxformats.chain_refusal would refuse")
    ap.add_argument("--target", choices=TARGETS, default="spike",
                    help="build define: spike (-DSPIKE_SIM) or mx_rocket (-DMX_ROCKET, the RTL build)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default out/compile/<kernel>/<target>)")
    a = ap.parse_args(argv)

    if a.list:
        print("kernels:")
        for name, desc in list_kernels().items():
            print(f"  {name:10s} {desc}")
        print("\nhardware recipes (--config):")
        for name, desc in list_recipes().items():
            print(f"  {name:14s} {desc}")
        print("\ntargets (--target): " + ", ".join(TARGETS))
        return 0

    tel = Telemetry()
    wire_paths()
    from backend import MxEmitError, MxRunnerError
    try:
        recipe = load_recipe(a.config)
        spec = build(a.kernel, m=a.m, k=a.k, h=a.h, n=a.n, seed=a.seed)
        out = a.out or REPO / "out" / "compile" / spec.name / a.target
        compile(spec, recipe, dtype=a.dtype, target=a.target, out=out,
                allow_lossy_chain=a.allow_lossy_chain, tel=tel)
    except RecipeError as exc:
        tel.log("error", f"bad recipe: {exc}")
        return 2
    except ValueError as exc:
        tel.log("error", str(exc))
        return 2
    except (MxRunnerError, MxEmitError) as exc:
        tel.log("error", f"build failed: {exc}")
        return 1
    except Exception as exc:
        tel.log("error", f"{type(exc).__name__}: {exc}")
        return 2
    print(f"COMPILED  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
