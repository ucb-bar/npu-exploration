"""compile: a captured, quantized model -> one bare-metal ELF (merlin + mx-gemmini-mlir).

    from compiler import run
    run(Path("build/MLP/capture"), Path("build/MLP/quantize"), cfg, Path("build/MLP/compile"))

Reads ``capture/`` (graph, fp32 weights, input), ``quantize/weights.mx.safetensors`` and the
config's ``hardware`` and ``compile`` sections. Writes

    model.elf   every matmul on the MX array with the quantized weights embedded as given, the ops
                between them on the VPU (the classes in ``compile.vpu``) or the host CPU, the input
                embedded; prints its outputs as ``OUT`` lines
    run.json    what running it needs: {"isa": ..., "memory": "<base>:<bytes>"} (spike --isa, -m)

Internally it binds mx-gemmini-mlir's inputs in a work directory: the capture receipt it checks,
the execution policy (format per matmul from the quantized weights, VPU or host per op from
``compile.vpu``), then runs ``tools.build_merlin_stage``. With ``debug=True`` that work directory
is kept as ``out/debug/`` (policy, catalog, generated C, build log); otherwise only the ELF stays.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MX = HERE / "mx-gemmini-mlir"
MERLIN = HERE / "merlin" / "src"
M2M = HERE / "model2mlir"
LLVM = REPO / "toolchain" / "llvm-mlir-a47bddc"
# TODO: generate from cfg["hardware"]; today the one profile matching the stock VPU RTL
PROFILE = MX / "profiles" / "gemmini-mx-cleanup-266c593" / "MxE4M3Fp4VpuGemminiRocketConfig.json"

STAGE_FILES = ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
               "inputs.npz", "input_order.json", "extra.npz", "golden.npy")
POLICY_FORMAT = {"fp8_e4m3": "mxfp8", "fp4_e2m1": "mxfp4"}
VPU_CLASSES = ("add", "elementwise", "reduce", "softmax", "unary")
DRAM_BASE = 0x80000000      # spike/fesvr DRAM base (merlin runtime/backends/spike_arena.py)


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(map(str, (MX, MERLIN, M2M)))
    return env


def _stage(capture: Path, stage: Path) -> None:
    """The capture files their build reads, hard-linked (their checks refuse symlinks)."""
    stage.mkdir(parents=True)
    for name in STAGE_FILES:
        src = capture / name
        if src.is_file():
            try:
                os.link(src, stage / name)
            except OSError:
                shutil.copy2(src, stage / name)
    subprocess.run([sys.executable, "-c",
                    "import sys; from m2m.capture.provenance import write_capture_receipt; "
                    "write_capture_receipt(sys.argv[1])", str(stage)],
                   check=True, env=_env(), capture_output=True)


def _policy(stage: Path, quantized: Path, vpu: list[str], out: Path) -> dict:
    """Their execution policy, with each matmul's format taken from its quantized weight."""
    from safetensors import safe_open
    unknown = sorted(set(vpu) - set(VPU_CLASSES))
    if unknown:
        raise ValueError(f"compile.vpu {unknown}: choose from {', '.join(VPU_CLASSES)}")
    res = subprocess.run([sys.executable, "-m", "tools.make_execution_policy", "--stage", str(stage),
                          "--profile", str(PROFILE), "--out", str(out)],
                         cwd=MX, env=_env(), capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"execution policy failed:\n{res.stderr[-2000:]}")
    policy = json.loads(out.read_text())
    with safe_open(str(quantized), "np") as f:
        formats = f.metadata() or {}
    # matmul region -> its module (prov.fqn) -> that module's quantized weight
    module_of = {}
    for line in (stage / "model.mlir").read_text().splitlines():
        if 'prov.family = "contraction"' in line:
            region = re.search(r'prov\.region_id = "([^"]+)"', line)
            fqn = re.search(r'prov\.fqn = "([^"]+)"', line)
            if region and fqn:
                module_of[region.group(1)] = fqn.group(1)
    for region in policy["matrix_formats"]:
        fmt = formats.get(f"{module_of.get(region)}.weight", formats.get("__default__"))
        if fmt not in POLICY_FORMAT:
            raise ValueError(f"{region} ({module_of.get(region)}): format {fmt} has no MX route")
        policy["matrix_formats"][region] = POLICY_FORMAT[fmt]
    for cls, sites in policy["vpu_placements"].items():
        for region, route in sites.items():
            if cls not in vpu:
                sites[region] = "host"
            elif cls == "softmax":          # their default keeps softmax on the host; asked for here
                sites[region] = "vpu"
    out.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    return policy


def run(capture: Path, quantize: Path, cfg: dict, out: Path, *, debug: bool = False) -> Path:
    c = cfg["compile"]
    if c["target"] != "spike":
        raise ValueError(f"compile.target {c['target']!r}: only spike is supported so far")
    riscv = os.environ.get("RISCV")
    if not riscv:
        raise RuntimeError("RISCV is not set; source scripts/env.sh <chipyard root>")
    capture, quantize, out = Path(capture).resolve(), Path(quantize).resolve(), Path(out).resolve()
    quantized = quantize / "weights.mx.safetensors"

    shutil.rmtree(out, ignore_errors=True)
    work = out / "debug"
    _stage(capture, work / "stage")
    _policy(work / "stage", quantized, c["vpu"], work / "policy.json")
    import numpy as np
    outputs = int(np.load(capture / "golden.npy", mmap_mode="r").size)
    cmd = [sys.executable, "-m", "tools.build_merlin_stage",
           "--stage", str(work / "stage"), "--workdir", str(work / "build"),
           "--profile", str(PROFILE), "--merlin-src", str(MERLIN), "--llvm-install", str(LLVM),
           "--riscv-root", riscv, "--structural-elf-admission",
           "--execution-policy", str(work / "policy.json"),
           "--prepack-checkpoint-weights", "--quantized-weights", str(quantized),
           "--output-dump-cap", str(outputs)]
    with open(work / "build.log", "w") as log:
        res = subprocess.run(cmd, cwd=MX, env=_env(), stdout=log, stderr=subprocess.STDOUT)
    elf = work / "build" / "model.elf"
    if res.returncode or not elf.is_file():
        tail = (work / "build.log").read_text()[-3000:]
        raise RuntimeError(f"build failed (exit {res.returncode}); kept {work}\n{tail}")
    shutil.move(str(elf), out / "model.elf")
    probe = json.loads((work / "build" / "mx_stage_probe.json").read_text())
    (out / "run.json").write_text(json.dumps(
        {"isa": probe["spike_isa"], "memory": f"{DRAM_BASE:#x}:{probe['mem_bytes']:#x}"}) + "\n")
    if not debug:
        shutil.rmtree(work)
    return out / "model.elf"
