"""npu.py -- your hardware + your PyTorch model -> quantized, compiled, checked on spike, costed.

    source scripts/env.sh
    .venv/bin/python npu.py --hw config/hardware/baseline.json --model my_model.py:Block

Two inputs:

    --hw     a hardware config (a name in config/hardware/ or a path to a .json): the accelerator
    --model  FILE.py:Name, a plain PyTorch nn.Module at full precision (or --kernel NAME, a built-in example)

Four steps, each skippable with --skip (or pick a subset with --only):

    quantize   the bit-exact quantized model of your hardware: the bits it must produce, error vs fp32
    compile    lower the model to RoCC instructions and build a bare-metal ELF for the accelerator
    evaluate   run that ELF on spike and check it bit for bit against the quantized model (needs compile)
    cost       estimate the hardware's area, power and the model's cycles on it

    .venv/bin/python npu.py --hw baseline --kernel linear                         # all four steps
    .venv/bin/python npu.py --hw baseline --model m.py:Block --skip evaluate      # compile only, no spike
    .venv/bin/python npu.py --hw wide_acc --model m.py:Block --only quantize,cost # no toolchain needed
    .venv/bin/python npu.py --list

Exit 0 on PASS (or when nothing is graded), 1 on FAIL or a failed build, 2 on a refused input.
Results: results/<timestamp>_<model>/ when evaluate or quantize run; out/compile/<model>/<target>/ for the
deployable ELF when compile runs without evaluate (or with --target mx_rocket / --out).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The backend package registers itself with merlin whenever merlin is importable. Until the merlin
# (MLIR) path is wired in, the compile step lowers directly and refuses to load merlin, so the result
# does not depend on what happens to be on the path.
sys.modules.setdefault("merlin", None)

from compiler.lower import lower, refuse_graph_dtype, wire_paths  # noqa: E402

STEPS = ("quantize", "compile", "evaluate", "cost")
TARGETS = ("spike", "mx_rocket")


# ---------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------

def load_module(ref: str, *, m: int, k: int, seed: int, recipe, input_npy: Path | None = None):
    """``FILE.py:Name`` -> a traced KernelSpec, named ``Name``. TraceError is a ValueError.

    ``Name`` is an nn.Module class (instantiated with no arguments after ``torch.manual_seed(seed)``)
    or an instance. The input is ``randn(m, k)`` unless ``input_npy`` names a ``[M][K]`` .npy file.
    Supported ops: bias-free Linear, matmul, causal softmax, add, silu*mul, nn.RMSNorm, RoPE with
    fixed tables (kernels/trace.py)."""
    import numpy as np
    import torch
    from kernels.trace import trace
    file, _, attr = ref.rpartition(":")
    if not file or not attr:
        raise ValueError(f"--model must be FILE.py:Name, got {ref!r}")
    path = Path(file)
    if not path.exists():
        raise ValueError(f"--model: {path} does not exist")
    ms = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(mod)
    obj = getattr(mod, attr, None)
    if obj is None:
        raise ValueError(f"--model: {path} has no attribute {attr!r}")
    torch.manual_seed(seed)
    module = obj() if isinstance(obj, type) else obj
    if not isinstance(module, torch.nn.Module):
        raise ValueError(f"--model: {ref} is not an nn.Module (got {type(module).__name__})")
    if input_npy is not None:
        x = np.load(input_npy)
        if x.ndim != 2:
            raise ValueError(f"--input {input_npy}: expected a 2-D [M][K] array, got shape {x.shape}")
        x = torch.from_numpy(np.ascontiguousarray(x, np.float32))
    else:
        x = torch.randn(m, k)
    spec = trace(module, x, name=attr, dim=recipe.dim, block=recipe.block)
    spec.module = ref
    return spec


def resolve_steps(skip: str | None, only: str | None) -> tuple[str, ...]:
    """``--skip a,b`` / ``--only a,b`` -> the steps to run, in pipeline order."""
    def parse(arg):
        names = {t.strip() for t in arg.split(",") if t.strip()}
        bad = names - set(STEPS)
        if bad:
            raise ValueError(f"unknown step(s) {', '.join(sorted(bad))}; choose from {', '.join(STEPS)}")
        return names
    if skip and only:
        raise ValueError("--skip and --only are mutually exclusive")
    chosen = parse(only) if only else set(STEPS) - (parse(skip) if skip else set())
    if "evaluate" in chosen and "compile" not in chosen:
        raise ValueError("evaluate runs the compiled ELF on spike, so it needs compile; skip evaluate too")
    if not chosen:
        raise ValueError("every step is skipped; nothing to do")
    return tuple(s for s in STEPS if s in chosen)


# ---------------------------------------------------------------------------------------------------
# compile: the deployable ELF directory
# ---------------------------------------------------------------------------------------------------

def compile(spec, recipe, run=None, *, target: str = "spike", out: Path, tel=None) -> dict:
    """Lower ``spec`` for the hardware ``recipe`` driven by the run config ``run`` (default
    config/run/default.json), build the ELF into ``out``, write the expected bits and a manifest.
    Returns the manifest. Raises ``ValueError`` for anything refused.

    ``out`` holds ``mx_gemmini_rocket.elf`` and its ``main.c``, ``command_buffer.json``, ``operands.npz``
    (the input and every wire operand), ``expected.npy`` (the bits the ELF must print: the quantized
    model of this hardware) and ``manifest.json``. ``target`` changes only the build define
    (``spike``: -DSPIKE_SIM, ``mx_rocket``: -DMX_ROCKET, the RTL build); the C is byte-identical."""
    from grade.telemetry import Telemetry
    tel = tel or Telemetry()
    wire_paths()
    import numpy as np
    import backend as mx
    from config.recipe import RecipeError, check, emitter_params, load_run, lut_settings
    from models import mxquant, mxq_commit

    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
    run = run or load_run("default")
    check(recipe, run, "kernel")
    dtype = run.operand_fmt
    tel.log("recipe", f"{recipe.describe()}   build_id={recipe.build_id()}")
    tel.log("run", f"{run.describe()}   run_id={run.run_id()}")
    tel.log("kernel", f"{spec.describe()}   ({len(spec.stages)} stage"
                      f"{'s' if len(spec.stages) != 1 else ''})")
    errs = spec.validate(dim=recipe.dim, block=recipe.block)
    if errs:
        raise ValueError(f"{spec.name}: {len(errs)} shape violation(s):\n  " + "\n  ".join(errs))

    lut = lut_settings(recipe, run)
    low = lower(spec, dtype, allow_lossy_chain=run.allow_lossy_chain,
                warn=lambda m: tel.log("warning", m), lut=lut)
    if low.kind == "per_stage":
        host = [st.name for st in spec.stages if not st.on_mesh and not st.emittable]
        raise ValueError(
            f"{spec.name} cannot be compiled ahead of time: host stage(s) {host} carry a Python "
            "function, not an op the emitter knows (kernels/host_ops.OPS), so the only lowering is one "
            "ELF per matmul, each fed by the previous run. Run it with the evaluate step instead.")
    refuse_graph_dtype(low, spec.name, dtype)
    cb = low.cb
    cb["params"] = emitter_params(recipe, run)      # the emitters' scratchpad and LUT plan, from the recipes
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
        ref = mxquant.run(spec, recipe, dtype=dtype, edges=low.edges, shipped=False, lut=lut)
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
        "kernel": spec.name, "module": getattr(spec, "module", None),
        "lowering": low.kind, "target": target, "dtype": dtype,
        "intermediate_dtype": (cb["tensors"] and next(iter(cb["tensors"].values()))["dtype"]
                               if low.kind == "fused" and len(spec.stages) > 1 else None),
        "m": spec.m,
        "dims": [mnk[mesh[0].name][1]] + [mnk[s.name][2] for s in mesh],
        "stages": low.stages,
        "edges": {k: v["via"] for k, v in low.edges.items()},
        "recipe": {"name": recipe.name, "build_id": recipe.build_id(), "path": str(recipe.path),
                   "hardware": recipe.hardware()},
        "run": {"name": run.name, "run_id": run.run_id(), "path": str(run.path), **run.fields()},
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


# ---------------------------------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------------------------------

def _print_list() -> None:
    from config.recipe import list_hardware, list_runs
    from kernels.registry import list_kernels
    print("steps (--skip / --only): " + ", ".join(STEPS))
    print("\nhardware configs (--hw):")
    for name, desc in list_hardware().items():
        print(f"  {name:14s} {desc}")
    print("\nrun configs (--run):")
    for name, desc in list_runs().items():
        print(f"  {name:14s} {desc}")
    print("\nbuilt-in example models (--kernel):")
    for name, desc in list_kernels().items():
        print(f"  {name:14s} {desc}")
    print("\ntargets (--target): " + ", ".join(TARGETS))


def main(argv: list[str] | None = None) -> int:
    from config.recipe import RecipeError, load_hardware, load_run, removed_flag
    from grade.telemetry import Telemetry
    from kernels.registry import build

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list steps, hardware configs, run configs and built-in models, then exit")
    ap.add_argument("--hw", "--config", dest="hw", default="baseline",
                    help="hardware config: a name in config/hardware/ or a path to a .json")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--model", "--module", dest="model", default=None,
                     help="FILE.py:Name -- your PyTorch nn.Module, at full precision")
    src.add_argument("--kernel", default=None, help="a built-in example model instead (see --list)")
    ap.add_argument("--input", type=Path, default=None,
                    help="with --model: a [M][K] float .npy input instead of randn(m, k)")
    ap.add_argument("--run", default="default",
                    help="run config: a name in config/run/ or a path to a .json "
                         "(operand format, rounding, pass threshold)")
    ap.add_argument("--skip", default=None, help=f"comma list of steps to skip: {', '.join(STEPS)}")
    ap.add_argument("--only", default=None, help="comma list of the only steps to run")
    ap.add_argument("--m", type=int, default=64, help="batch rows (input rows)")
    ap.add_argument("--k", type=int, default=64, help="in_features")
    ap.add_argument("--h", type=int, default=64, help="hidden width (built-in chained models)")
    ap.add_argument("--n", type=int, default=64, help="out_features (built-in models)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", choices=TARGETS, default="spike",
                    help="compile step build define: spike, or mx_rocket for the RTL/FPGA build")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the compile step writes the ELF (default out/compile/<model>/<target>)")
    gone = removed_flag(sys.argv[1:] if argv is None else argv)
    if gone:
        print(f"[error   ] {gone}", file=sys.stderr)
        return 2
    a = ap.parse_args(argv)
    if a.list:
        _print_list()
        return 0

    tel = Telemetry()
    wire_paths()
    from backend import MxEmitError, MxRunnerError
    try:
        steps = resolve_steps(a.skip, a.only)
        tel.log("steps", " -> ".join(s if s in steps else f"({s} skipped)" for s in STEPS))
        recipe, run_recipe = load_hardware(a.hw), load_run(a.run)
        if a.input is not None and a.model is None:
            raise ValueError("--input applies to --model only; built-in models make their own input")
        if a.model:
            spec = load_module(a.model, m=a.m, k=a.k, seed=a.seed, recipe=recipe, input_npy=a.input)
        else:
            spec = build(a.kernel or "linear", m=a.m, k=a.k, h=a.h, n=a.n, seed=a.seed)

        # compile without evaluate (or an explicit RTL target / output dir): the deployable ELF directory.
        compiled = None
        if "compile" in steps and ("evaluate" not in steps or a.target != "spike" or a.out is not None):
            compiled = a.out or REPO / "out" / "compile" / spec.name / a.target
            compile(spec, recipe, run_recipe, target=a.target, out=compiled, tel=tel)

        # quantize / evaluate / cost: one pipeline run, the models in parallel with spike.
        models = (("mxquant",) if "quantize" in steps else ()) \
            + (("spike",) if "evaluate" in steps else ()) \
            + (("ppa", "perf") if "cost" in steps else ())
        res = None
        if models:
            from grade.pipeline import run
            res = run(spec, recipe=recipe, run_recipe=run_recipe, telemetry=tel,
                      models=("reference",) + models)
    except RecipeError as exc:
        tel.log("error", f"bad config: {exc}")
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

    if compiled is not None:
        print(f"COMPILED {compiled}")
    if res is None or res["metrics"] is None:
        return 0
    m = res["metrics"]
    # VERDICT belongs to evaluate + quantize (spike == quantized model, bit for bit); evaluate alone is
    # graded against fp32; quantize alone prints the quantized model's numbers and NO VERDICT; cost
    # alone prints only its PPA and PERF lines.
    if m.get("correctness_vs_mxquant") is not None or ("quantize" in steps and "evaluate" not in steps):
        from models import mxquant
        print(mxquant.line(m))
    elif "evaluate" in steps:
        from models import reference
        print(reference.line(m, tol=run_recipe.fp32_tol))
    if m.get("ppa"):
        from models.ppa import ppa
        print(ppa.line(m["ppa"]))
    if m.get("perf"):
        from models.perf import perf
        print(perf.line(m["perf"]))
    print(f"RESULTS  {res['run_dir']}")
    return {True: 0, False: 1, None: 0}[m["pass"]]


if __name__ == "__main__":
    raise SystemExit(main())
