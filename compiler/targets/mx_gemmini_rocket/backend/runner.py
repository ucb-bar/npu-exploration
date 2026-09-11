"""Run merlin command buffers on the Rocket-hosted MX Gemmini, via the chipyard bare-metal flow.

Pipeline (mirrors ``targets/gemmini/backend/gemmini.py``, which is the shape merlin expects):

    command buffer
      -> mxgemm_emit.generate_driver          C driver using gemmini.h MX intrinsics
      -> riscv64-unknown-elf-gcc              bare-metal ELF, built against gemmini-rocc-tests
      -> an oracle:
           spike --extension=gemmini          functional model, BOOTSTRAP ONLY (derived_from_rtl=False)
           the Verilator RTL sim              certification (derived_from_rtl=True) — later
      -> parse OUT/METRIC/DONE

Spike and Verilator run the *exact same ELF*; only the launch command differs.

Toolchain resolution is environment-first so this works in a plain chipyard checkout:
``MERLIN_CHIPYARD`` / ``CHIPYARD_ROOT`` or the tree this package sits in, plus optional
``MX_RISCV_GCC`` / ``MX_SPIKE`` / ``MX_LIBGEMMINI`` / ``MX_ROCC_TESTS`` overrides.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .mxgemm_emit import Transport, generate_driver

#: Oracle tiers this backend can run, and whether a number from each is an RTL claim.
ORACLE = {
    "spike": {"kind": "spike_mx_gemmini_functional", "derived_from_rtl": False},
    "verilator": {"kind": "rtl_verilator", "derived_from_rtl": True},
}


class MxRunnerError(RuntimeError):
    pass


# --- Toolchain resolution -------------------------------------------------------------------------

def _env(*names: str) -> str | None:
    for n in names:
        val = os.environ.get(n)
        if val:
            return val
    return None


def chipyard_root() -> Path:
    """The chipyard tree. Defaults to walking up from this package, which sits at
    ``<chipyard>/generators/gemmini/npu-exploration/compiler/targets/<name>/backend``."""
    root = _env("MERLIN_CHIPYARD", "CHIPYARD_ROOT")
    if root:
        return Path(root)
    # backend / <target> / targets / compiler / npu-exploration / gemmini / generators / <root>
    return Path(__file__).resolve().parents[7]


def riscv_root() -> Path:
    """``$RISCV``. Not set by default — source the chipyard ``env.sh`` first."""
    root = _env("RISCV")
    if not root:
        raise MxRunnerError(
            "$RISCV is unset — source the chipyard env.sh first "
            f"({chipyard_root() / 'env.sh'})")
    return Path(root)


def gcc_path() -> Path:
    return Path(_env("MX_RISCV_GCC") or (riscv_root() / "bin" / "riscv64-unknown-elf-gcc"))


def spike_path() -> Path:
    return Path(_env("MX_SPIKE") or (riscv_root() / "bin" / "spike"))


def libgemmini_so() -> Path:
    """The spike MX functional model.

    Prefers the IN-TREE build over ``$RISCV/lib``: the installed copy is frequently stale relative
    to ``mx_fp_math.h`` (the Makefile lists only ``gemmini.cc`` as a prerequisite, so a header change
    does not trigger a rebuild), and a stale model is silent.
    """
    override = _env("MX_LIBGEMMINI")
    if override:
        return Path(override)
    in_tree = gemmini_root() / "software/libgemmini/libgemmini.so"
    return in_tree if in_tree.exists() else riscv_root() / "lib" / "libgemmini.so"


def gemmini_root() -> Path:
    """The gemmini sources: the surrounding chipyard tree (there is no in-repo pin)."""
    return chipyard_root() / "generators" / "gemmini"


def rocc_tests_dir() -> Path:
    """gemmini-rocc-tests — supplies ``include/gemmini.h`` (the MX intrinsics) and the bare-metal
    harness (crt.S, syscalls.c, the linker script)."""
    override = _env("MX_ROCC_TESTS")
    if override:
        return Path(override)
    return gemmini_root() / "software/gemmini-rocc-tests"


def _common_dir() -> Path:
    return rocc_tests_dir() / "riscv-tests" / "benchmarks" / "common"


def available(simulator: str = "spike") -> bool:
    """True when this oracle can actually run. Checked rather than assumed: a missing toolchain
    must report NOT RUN, never a silent pass."""
    try:
        if not (gcc_path().exists() and rocc_tests_dir().exists()):
            return False
        if simulator == "spike":
            return spike_path().exists() and libgemmini_so().exists()
        return False   # verilator: later
    except MxRunnerError:
        return False


# --- Build ----------------------------------------------------------------------------------------

#: The one compile-time difference between the two substrates, mirroring how the reference tree
#: builds: `bareMetalC/Makefile:111` adds -DSPIKE_SIM when $(RUNNER) contains "spike", and the RTL
#: build passes EXTRA_CFLAGS=-DMX_ROCKET (`build_mx_rocket/`).
#:
#: The C WE emit reads neither define -- it has no target conditionals at all, because it only ever
#: targets the real RoCC. They still matter because `gemmini.h` and the test headers branch on them.
TARGET_DEFINE = {"spike": "-DSPIKE_SIM", "mx_rocket": "-DMX_ROCKET"}


def compile_command_buffer(cb: dict[str, Any], workdir: str | Path, *,
                           driver_src: str | None = None,
                           transport: Transport | None = None,
                           target: str = "spike") -> Path:
    """Emit the C driver and compile the bare-metal ELF; return the ELF path.

    ``driver_src`` overrides codegen with externally-provided C — the rest of the build/run path is
    identical (same seam merlin's gemmini backend offers for certifying a hand-written kernel).

    ``target`` selects the substrate define only. The generated source is byte-identical either way;
    that is the property Step 4 of ``planning/merlin_glue_port_plan.md`` set out to establish, and
    ``tests/selftest_mx_rocket_build.py`` asserts it.
    """
    if target not in TARGET_DEFINE:
        raise MxRunnerError(f"unknown target {target!r}; known: {sorted(TARGET_DEFINE)}")
    runtime = Path(__file__).resolve().parent / "runtime"
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    main_c = work / "main.c"
    main_c.write_text(
        driver_src if driver_src is not None else generate_driver(cb, transport=transport),
        encoding="utf-8")
    elf = work / "mx_gemmini_rocket.elf"
    rt, common = rocc_tests_dir(), _common_dir()

    # Mirrors gemmini-rocc-tests/bareMetalC/Makefile CFLAGS_BAREMETAL exactly. Both the flag set and
    # the INCLUDE ORDER matter: a wrong order shadows the riscv-tests/env syscall headers and
    # corrupts the tohost protocol ("bad syscall" on spike).
    # The target define selects the spike vs RTL variants of the MX macros in the test headers.
    cmd = [
        str(gcc_path()),
        TARGET_DEFINE[target], "-DPREALLOCATE=1", "-DMULTITHREAD=1",
        "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math",
        "-fno-common", "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
        "-march=rv64gc", "-Wa,-march=rv64gc",
        "-I", str(rt / "riscv-tests"),
        "-I", str(rt / "riscv-tests/env"),
        "-I", str(rt),
        "-I", str(common),
        # The backend's own C runtime (mx_host.h): the fp32 host side of a layer plus the MX
        # quantizer that hands its result back to the mesh. Step 6 of merlin_glue_port_plan.md.
        "-I", str(runtime),
        "-DID_STRING=", "-DPRINT_TILE=0",
        "-nostdlib", "-nostartfiles", "-static",
        "-T", str(common / "test.ld"), "-DBAREMETAL=1",
        str(main_c), "-o", str(elf),
        *(str(p) for p in sorted(common.glob("*.c"))),
        *(str(p) for p in sorted(common.glob("*.S"))),
        # AFTER the sources, not in the flags. A library named before the objects that need it
        # resolves nothing -- `expf` (SiLU, softmax) then comes back undefined at link time. The
        # reference tree hit exactly this (llama_layer_hw_plan.md section 8.4) and fixed it the same
        # way; newlib's libm also wants `__errno`, which mx_host.h stubs under -DBAREMETAL.
        "-lm", "-lgcc",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MxRunnerError(f"riscv gcc failed:\n{' '.join(cmd)}\n{proc.stderr[-4000:]}")
    return elf


# --- Run ------------------------------------------------------------------------------------------

def run_elf(elf: str | Path, simulator: str = "spike", timeout: int = 600) -> str:
    """Run the ELF on the chosen oracle; return raw console output."""
    if simulator != "spike":
        raise MxRunnerError(f"simulator {simulator!r} not wired up yet (spike only)")
    so = libgemmini_so()
    if not so.exists():
        raise MxRunnerError(f"libgemmini.so not found at {so} — build it in software/libgemmini")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{so.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [str(spike_path()), f"--extlib={so}", "--extension=gemmini", str(elf)]
    # errors="replace": a kernel that runs wild prints raw bytes, and a UnicodeDecodeError
    # traceback hides that. Decode lossily so the caller sees the garbage and can diagnose it.
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                          timeout=timeout, env=env)
    if proc.returncode != 0:
        raise MxRunnerError(
            f"spike exited {proc.returncode}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc.stdout


def parse_output(text: str) -> tuple[dict[str, list], dict[str, int]]:
    """Parse the shared OUT/METRIC/DONE console protocol into (outputs, raw metrics).

    Delegates to merlin's shared parser when merlin is importable, so this backend cannot drift from
    the protocol; falls back to an identical local parse so a standalone bring-up run needs no
    merlin on the path.
    """
    try:
        from merlin.runtime.backends.base import parse_console
    except ImportError:
        return _parse_console_local(text)
    return parse_console(text, error_cls=MxRunnerError, strip_warnings=True, tolerant_metric=True)


def _parse_console_local(text: str) -> tuple[dict[str, list], dict[str, int]]:
    outputs: dict[str, list] = {}
    raw: dict[str, int] = {}
    done = False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "OUT":
            name, rows, cols = parts[1], int(parts[2]), int(parts[3])
            vals = [int(v) for v in parts[4:]]
            if len(vals) != rows * cols:
                raise MxRunnerError(f"OUT {name}: expected {rows * cols} values, got {len(vals)}")
            outputs[name] = [vals[r * cols:(r + 1) * cols] for r in range(rows)]
        elif parts[0] == "METRIC":
            try:
                raw[parts[1]] = int(parts[2])
            except (IndexError, ValueError):
                pass
        elif parts[0] == "DONE":
            done = True
    if not done:
        raise MxRunnerError(f"run did not reach DONE; output was:\n{text[:2000]}")
    return outputs, raw


def run_command_buffer(cb: dict[str, Any], *, workdir: str | Path,
                       simulator: str = "spike", timeout: int = 600,
                       transport: Transport | None = None) -> dict[str, Any]:
    """Emit, build, run, parse. Returns outputs + metrics + the oracle's provenance."""
    elf = compile_command_buffer(cb, workdir, transport=transport)
    console = run_elf(elf, simulator=simulator, timeout=timeout)
    outputs, raw = parse_output(console)
    return {
        "outputs": outputs,
        "metrics": raw,
        "oracle": {"simulator": simulator, **ORACLE[simulator]},
        "elf": str(elf),
        "console": console,
    }
