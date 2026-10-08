"""Self-test of the spike cycle model's place in the pipeline (runner.run_elf(timing=...)).

The claims:
  1. PARSING: the cycle model's summary and parameter dump become the record's dicts, every printed
     counter under a key, unknown lines kept as text (a recorded summary, no toolchain);
  2. GUARDS: a run that left no summary (a libgemmini.so without the model) and a resolved geometry
     that differs from the recipe's are refused, never recorded;
  3. BITS: for the kernels the pipeline lowers (one matmul, a fused chain, the attention graph) the ELF
     prints the same OUT lines under GEMMINI_MODE=both as under func, and both runs reach DONE. The cycle
     model knows time only, never data (gemmini_perf.h); this is where that is held. Needs the toolchain
     (scripts/env.sh); skipped cleanly without it, with a line saying so.

Exit 0 on pass, 1 on failure.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "compiler" / "targets" / "mx_gemmini_rocket"))

from backend import runner  # noqa: E402

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append(ok)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{('  ' + detail) if detail else ''}")


SUMMARY = """gemmini perf [mx_rocket]: 76 commands, 1 loops, 777 fences; last activity 8206; host stalled 3494 cycles; 5506 events
  mesh: 64 tiles, 1024 busy cycles; VPU: 0 commands
  load: 8192 bytes in 640 Gets; store: 640 reads, 640 Puts; scales: 256 bytes
  memory: L2 517 hits, 133 misses; bus busy 1280, DRAM busy 1064 cycles; CPU stores 159920 lines, L1 probes 120, partial-write fills 0, dirty write-backs 0
  ports busy: sp0 r1536/w832 sp1 r0/w0 sp2 r0/w0 sp3 r256/w320 acc0 1152 acc1 0
  not modelled: MX_READ_SMEM x3
  some new line the model may print one day
"""

CONFIG = """gemmini: GEMMINI_MODE=both
gemmini perf config (preset mx_rocket):
  host.cpi                                  1     host cycles per retired instruction
  mesh.dim                                 16     mesh rows = cols = DIM; must match the kernel build
  spad.banks                                4     scratchpad banks
  spad.bank_rows                         4096     rows per bank (one row = DIM bytes)
  mem.dram_bytes_per_cycle                 16 *   DRAM channel
  mx.block                                 32     elements per E8M0 scale block
"""


def main() -> int:
    print("1. parsing a recorded summary and parameter dump")
    s = runner.parse_timing_summary(SUMMARY)
    check("header counters", (s["preset"], s["commands"], s["fences"], s["host_stalled_cycles"], s["events"])
          == ("mx_rocket", 76, 777, 3494, 5506))
    check("mesh / load / memory lines", (s["mesh_tiles"], s["mesh_busy_cycles"], s["load_gets"], s["l2_misses"],
                                         s["dram_busy_cycles"], s["dirty_writebacks"]) == (64, 1024, 640, 133, 1064, 0))
    check("ports per bank", s["ports"] == {"sp0": {"read": 1536, "write": 832}, "sp1": {"read": 0, "write": 0},
                                           "sp2": {"read": 0, "write": 0}, "sp3": {"read": 256, "write": 320},
                                           "acc0": 1152, "acc1": 0}, str(s["ports"]))
    check("not-modelled commands counted", s["not_modelled"] == {"MX_READ_SMEM": 3})
    check("an unknown line is kept, not fatal", s["unparsed"] == ["some new line the model may print one day"])
    check("every value is JSON", json.dumps(s) is not None)
    c = runner.parse_timing_config(CONFIG)
    check("preset and parameters", c["preset"] == "mx_rocket" and c["params"]["mesh.dim"] == 16
          and c["params"]["spad.bank_rows"] == 4096 and c["params"]["host.cpi"] == 1)
    check("the overridden ones are named", c["overridden"] == ["mem.dram_bytes_per_cycle"], str(c["overridden"]))
    try:
        runner.parse_timing_summary("spike said nothing useful")
        check("a summary without the header is refused", False)
    except runner.MxRunnerError:
        check("a summary without the header is refused", True)
    env = runner.timing_env({"dim": 16, "banks": 4, "rows": 4096, "block": 32}, Path("/x"))
    check("the environment switches both on, with every geometry parameter set",
          env["GEMMINI_MODE"] == "both" and env["GEMMINI_PERF_SET"] == "mesh.dim=16,spad.banks=4,spad.bank_rows=4096,mx.block=32"
          and env["GEMMINI_PERF_OUT"] == "/x/" + runner.TIMING_SUMMARY, env["GEMMINI_PERF_SET"])
    try:
        runner.timing_env({"dim": 16}, Path("/x"))
        check("an incomplete geometry is refused", False)
    except runner.MxRunnerError:
        check("an incomplete geometry is refused", True)

    print("2. guards")
    req = {"dim": 16, "banks": 4, "rows": 4096, "block": 32}
    with tempfile.TemporaryDirectory() as td:
        try:
            runner.read_timing(td, req)
            check("no summary file -> refused (a .so without the cycle model)", False)
        except runner.MxRunnerError as exc:
            check("no summary file -> refused (a .so without the cycle model)", "no cycle model" in str(exc))
        (Path(td) / runner.TIMING_SUMMARY).write_text(SUMMARY)
        (Path(td) / runner.TIMING_CONFIG).write_text(CONFIG)
        t = runner.read_timing(td, req)
        check("summary + config read back", t["mode"] == "both" and t["summary"]["mesh_busy_cycles"] == 1024
              and t["config"]["params"]["mesh.dim"] == 16)
        try:
            runner.read_timing(td, {**req, "dim": 32})
            check("mesh.dim differing from the recipe -> refused", False)
        except runner.MxRunnerError as exc:
            check("mesh.dim differing from the recipe -> refused", "mesh.dim=16" in str(exc))

    print("3. the bits do not depend on the mode (needs the toolchain)")
    if not runner.available("spike"):
        print("  skip  toolchain unavailable -- source scripts/env.sh")
    else:
        sys.path.insert(0, str(REPO))
        from models.spike.build_spike import has_cycle_model
        so = runner.libgemmini_so()
        check(f"the loaded libgemmini.so carries the cycle model ({so.name})", has_cycle_model(so), str(so))
        from compiler.lower import lower
        from config.recipe import emitter_params, load_hardware, load_run
        from kernels.registry import build as build_kernel
        hw, run = load_hardware("baseline"), load_run("default")
        work = Path(tempfile.mkdtemp(prefix="selftest_timing_"))
        try:
            for name in ("linear", "mlp2", "attention"):
                spec = build_kernel(name, m=64, k=64, h=64, n=64, seed=0)
                low = lower(spec, run.operand_fmt)
                cb = low.cb
                cb["params"] = emitter_params(hw, run)
                elf = runner.compile_command_buffer(cb, work / name)
                plain = runner.run_elf(elf)
                timed = runner.run_elf(elf, timing=req)
                outs_p, met_p = runner.parse_output(plain)
                outs_t, met_t = runner.parse_output(timed)
                check(f"{name} ({low.kind}): OUT lines identical under func and both", outs_p == outs_t)
                check(f"{name}: both reports more cycles than the instruction counter",
                      met_t.get("cycles", 0) > met_p.get("cycles", 0), f"{met_p.get('cycles')} -> {met_t.get('cycles')}")
                t = runner.read_timing(elf.parent, req)
                check(f"{name}: summary read back (mesh busy {t['summary'].get('mesh_busy_cycles')}, "
                      f"host stalled {t['summary'].get('host_stalled_cycles')})",
                      t["summary"].get("mesh_busy_cycles", 0) > 0 and t["config"]["params"]["mesh.dim"] == 16)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    print()
    if all(CHECKS):
        print(f"ALL {len(CHECKS)} CHECKS PASSED -- the cycle model runs beside the bits and leaves the same bits.")
        return 0
    print(f"{CHECKS.count(False)} of {len(CHECKS)} checks FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
