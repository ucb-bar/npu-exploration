"""The emitted C is ONE program for both substrates: spike and the standalone RTL.

Three claims, checked rather than asserted (``planning/merlin_glue_port_plan.md`` Step 4):

1. **The generated source is byte-identical** for both targets. It contains no ``SPIKE_SIM`` /
   ``MX_ROCKET`` conditional at all — the reference tests need those only because they also support
   a third endpoint (the Radiance MMIO command-mimic), and we do not.
2. **It compiles for `-DMX_ROCKET`**, not only for `-DSPIKE_SIM`.
3. **It drains the way the baremetal tests drain**: a flat contiguous MVOUT, never MX_READ_SMEM.
   In ``gemmini-rocc-tests/bareMetalC`` every use of ``gemmini_mx_read_smem`` sits inside an
   ``#ifdef SPIKE_SIM``; the sequence that serves both substrates is the MVOUT loop
   (``matmul_tiled_fp8_64x64.c:146-160``, ``..._requant.c:170-176``).

The reference tests are the contract this checks against — they are the ABI as exercised, and D1 is
exactly the decision to depend on that rather than on the generator's internals. This deliberately
asserts nothing about the RTL, and runs nothing on it (D6).

    .venv/bin/python tests/selftest_mx_rocket_build.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "app", REPO / "compiler" / "targets" / "mx_gemmini_rocket",
           REPO / "merlin" / "merlin" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                    # noqa: E402

import mxiface                                        # noqa: E402
from app.mxq_golden import quantize_operand           # noqa: E402
from backend import runner                            # noqa: E402
from backend.mxgemm_emit import generate_driver       # noqa: E402


def _cb(stages):
    """A command buffer for a chain of ``[(m, k, n), ...]`` with MXQuant-quantized operands."""
    rng = np.random.default_rng(0)
    ms, bundles = [], []
    for i, (m, k, n) in enumerate(stages):
        last = i == len(stages) - 1
        b_codes, b_scales, _ = quantize_operand(
            rng.standard_normal((k, n)).astype(np.float32), side="b")
        bundle = {"b_codes": b_codes, "b_scales": b_scales}
        if i == 0:
            a_codes, a_scales, _ = quantize_operand(
                rng.standard_normal((m, k)).astype(np.float32), side="a")
            bundle |= {"a_codes": a_codes, "a_scales": a_scales}
        bundles.append(bundle)
        ms.append(mxiface.MatmulStage(
            m=m, k=k, n=n, weight=f"W{i}", out="Y0" if last else f"T{i}",
            lhs="X" if i == 0 else f"T{i - 1}",
            out_dtype="bf16" if last else "f8E4M3FN"))
    iface = mxiface.chain_interface_mlir(ms)
    return mxiface.to_command_buffer(iface, bundles)


CASES = {
    "single 64x64x64": [(64, 64, 64)],
    "chain of 3": [(64, 64, 64)] * 3,
    "non-square chain": [(32, 128, 64), (32, 64, 96)],
}


def main() -> int:
    fails = checks = 0

    def check(name, ok, detail=""):
        nonlocal fails, checks
        checks += 1
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))

    print("[1] the generated C is target-independent ----------------------")
    for name, stages in CASES.items():
        src = generate_driver(_cb(stages))
        check(f"{name}: no target conditional in the source",
              "SPIKE_SIM" not in src and "MX_ROCKET" not in src)
        check(f"{name}: drains with a flat MVOUT, as the baremetal tests do",
              "gemmini_mx_read_smem" not in src and "gemmini_extended_mvout" in src,
              "mx_read_smem is #ifdef SPIKE_SIM-only in bareMetalC")

    print("\n[2] it compiles for BOTH substrates -----------------------------")
    ok, why = True, ""
    try:
        runner.gcc_path()
    except Exception as exc:                       # no toolchain: skip, do not fail
        ok, why = False, str(exc)
    if not ok:
        print(f"  SKIP  riscv gcc unavailable -- {why}")
        print(f"\n{checks} checks, {fails} failure(s)")
        return 1 if fails else 0

    for name, stages in CASES.items():
        cb = _cb(stages)
        elfs = {}
        with tempfile.TemporaryDirectory() as td:
            src_seen = set()
            for target in ("spike", "mx_rocket"):
                d = Path(td) / target
                try:
                    elf = runner.compile_command_buffer(cb, d, target=target)
                    elfs[target] = elf.stat().st_size
                except Exception as exc:
                    check(f"{name}: builds for {target}", False, str(exc)[:200])
                    continue
                src_seen.add((d / "main.c").read_text())
                check(f"{name}: builds for {target}", True, f"{elfs[target]} B")
            check(f"{name}: both targets got byte-identical C", len(src_seen) == 1)

    print(f"\n{checks} checks, {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
