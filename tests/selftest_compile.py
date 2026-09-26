"""compile_kernel.py: the same C as the graded path, an ELF that prints the expected bits, refusals.

Claims:
1. For every kernel x format the graded path builds, ``compile_kernel.compile`` writes a ``main.c``
   byte-identical to the one ``pipeline.run(build_only=True)`` writes -- same lowering, same
   emitter -- for BOTH targets, whose C is identical to each other.
2. The spike-target ELF, run through the backend runner, prints ``expected.npy`` bit for bit
   (linear, mlp2, attention on baseline): the compile product closes the loop without the pipeline.
3. Refusals happen before any build and need no toolchain: a shape violation, a graph kernel in a
   non-fp8 format, a host stage with a Python function, an unknown target.

Needs the RISC-V toolchain for 1 and spike for 2; each SKIPs with a reason otherwise.

    .venv/bin/python tests/selftest_compile.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

FAILURES: list[str] = []
FORMATS = ("fp8_e4m3", "fp8_e4m3_quad", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1")


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def same(a, b) -> bool:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return a.shape == b.shape and bool(((a == b) | (np.isnan(a) & np.isnan(b))).all())


class Quiet:
    def log(self, *a, **k):
        pass


def main() -> int:
    import models
    if not models.paths():
        print(f"SKIP: {models.mxq_missing()}")
        return 0
    from compiler.lower import wire_paths
    wire_paths()
    import compile_kernel as ck
    from backend import runner
    from config.recipe import load
    from grade import pipeline
    from kernels.registry import build
    from kernels.spec import HostStage, KernelSpec, Stage
    import torch
    from app import mxwire as w
    base = load("baseline")
    quiet = Quiet()

    print("[3] refusals, before any build ------------------------------------")
    check("shape violation -> exit 2", ck.main(["--kernel", "linear", "--m", "60", "--out", "/nonexistent/x"]) == 2)
    check("graph kernel in fp8_e5m2 -> exit 2",
          ck.main(["--kernel", "attention", "--dtype", "fp8_e5m2", "--out", "/nonexistent/x"]) == 2)
    check("graph kernel in fp4_e2m1 -> exit 2",
          ck.main(["--kernel", "attention", "--dtype", "fp4_e2m1", "--out", "/nonexistent/x"]) == 2)
    torch.manual_seed(0)
    fn_spec = KernelSpec("fnhost", torch.randn(64, 64), [
        Stage("L0", weight=torch.randn(64, 64)),
        HostStage("H", fn=lambda t: t * 2.0, src="L0"),
        Stage("L1", weight=torch.randn(64, 64), lhs="H")])
    check("fn= host stage -> ValueError naming it",
          _raises(lambda: ck.compile(fn_spec, base, out=Path("/nonexistent/x"), tel=quiet), "['H']"))
    check("unknown target -> ValueError",
          _raises(lambda: ck.compile(build("linear"), base, target="fpga", out=Path("/nonexistent/x"),
                                     tel=quiet), "fpga"))
    check("unknown kernel -> exit 2", ck.main(["--kernel", "nope"]) == 2)
    check("merlin is not loaded in a compile_kernel process",
          sys.modules.get("merlin") is None and not any(m.startswith("merlin.") for m in sys.modules))

    print("\n[1] main.c identical to the graded path, both targets --------------")
    try:
        runner.gcc_path()
    except Exception as exc:
        print(f"  SKIP  riscv gcc unavailable -- {exc}")
        return _finish()
    cases = [(k, d) for k in ("linear", "mlp2", "mlp3") for d in FORMATS] + [("attention", "fp8_e4m3")]
    for k, d in cases:
        spec = build(k)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pipeline.run(spec, recipe=base, dtype=d, build_only=True, models=("spike",),
                         workdir=td / "pipe", allow_lossy_chain=True, telemetry=quiet)
            ref = next((td / "pipe").glob("*/main.c")).read_bytes()
            srcs = {}
            for target in ("spike", "mx_rocket"):
                man = ck.compile(spec, base, dtype=d, target=target, out=td / target,
                                 allow_lossy_chain=True, tel=quiet)
                srcs[target] = (td / target / "main.c").read_bytes()
                check(f"{k} {d} {target}: ELF built, files listed",
                      (td / target / "mx_gemmini_rocket.elf").exists()
                      and {"main.c", "expected.npy", "operands.npz", "manifest.json",
                           "command_buffer.json"} <= set(man["files"]))
            check(f"{k} {d}: main.c == pipeline --build-only", srcs["spike"] == ref)
            check(f"{k} {d}: main.c identical for both targets", srcs["spike"] == srcs["mx_rocket"])

    print("\n[2] the spike ELF prints expected.npy ---------------------------------")
    if not runner.available("spike"):
        print("  SKIP  spike or libgemmini.so unavailable")
        return _finish()
    for k in ("linear", "mlp2", "attention"):
        with tempfile.TemporaryDirectory() as td:
            man = ck.compile(build(k), base, out=Path(td), tel=quiet)
            out, _ = runner.parse_output(runner.run_elf(man["elf"]["path"]))
            got = w.bf16_bits_to_float(np.array(out["Y0"], dtype=np.uint16))
            exp = np.load(Path(td) / "expected.npy")
            check(f"{k}: ELF output == expected.npy ({exp.size} values)", same(got.reshape(exp.shape), exp))
    return _finish()


def _raises(fn, needle: str) -> bool:
    try:
        fn()
    except ValueError as exc:
        return needle in str(exc)
    return False


def _finish() -> int:
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nALL CHECKS PASSED")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
