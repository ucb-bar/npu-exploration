"""Perf-model regression: every test with VCS ground truth on the exact binary, model vs VCS, phase by phase.

Runs spike in perf mode (and the replay tests in both mode with GEMMINI_PERF_REPLAY) against the in-tree
libgemmini.so, parses the numbers each test prints, and prints one table (perf_model_plan.md section 13).
`attn_flash_llama_vb` is the validation test (never used to choose a constant); its row is marked.

Usage: python3 tools/perf_regress.py [--so path/to/libgemmini.so] [--set "mem.x=1,..."] [--only substr]
Needs: source scripts/env.sh <chipyard root>.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEM = REPO.parent
VCS = Path(os.environ.get("MX_VCS_OUT", GEM.parents[1] / "sims/vcs/output"))
ISA = GEM / "software/gemmini-rocc-tests/build_mx_rocket/bareMetalC"
KER = REPO / "out/baremetal/mx_rocket"
CFG_MX = "chipyard.harness.TestHarness.MxGemminiRocketConfig"

# (name, elf, vcs log, how to read the numbers). Only tests whose ELF matches the VCS run's .dump.
DRAMLOOPS = ["dramloop", "dramloop_nc", "dramloop_kt", "dramloop_nc4", "dramloop_nc_2d", "dramloop_ls",
             "dramloop_ls4", "dramloop_nc_wait"]
# The 128x128 VCS run is from the pre-09-28 256-bit system-bus build (a TLWidthWidget splits each Put in two;
# perf_model_plan.md 13.6.1): modelled with that build's bus width.
OLD_BUS = {"mem.bus_bytes": "32"}
TESTS = [("128x128 (spad loop, old 256-bit bus build)", ISA / "matmul_tiled_fp8_128x128-baremetal", "perf",
          "matmul_tiled_fp8_128x128-baremetal", "binary differs from the VCS run: mvout VCS-equiv ~4560")]
TESTS += [(n, ISA / f"matmul_tiled_fp8_128x128_{n}-baremetal", "perf", f"matmul_tiled_fp8_128x128_{n}-baremetal", "")
          for n in DRAMLOOPS]
TESTS += [("mx_mem_bw", ISA / "mx_mem_bw-baremetal", "membw", "mx_mem_bw-baremetal", "")]
TESTS += [("llama_mlp_tiny_db", KER / "llama_mlp_tiny_db", "phase", "llama_mlp_tiny_db", "host: cpi 1 placeholder"),
          ("llama_mlp_small", KER / "llama_mlp_small", "mesh", "llama_mlp_small", "host: cpi 1 placeholder")]
REPLAY = [("llama_mlp_tiny_native_ua (replay)", KER / "llama_mlp_tiny_native_ua", "llama_mlp_tiny_native_ua")]
# VPU config (MxE4M3VpuGemminiRocketConfig, preset e4m3_vpu): every "PERF ... <n> cycles" line, in order.
CFG_VPU = "chipyard.harness.TestHarness.MxE4M3VpuGemminiRocketConfig"
VPU_TESTS = [("chain_pipelined", ISA / "chain_pipelined-baremetal", "chain_pipelined-baremetal"),
             ("attn_vpu_fa", KER / "attn_vpu_fa", "attn_vpu_fa"),
             ("attn_flash_llama7b_fused", KER / "attn_flash_llama7b_fused", "attn_flash_llama7b_fused"),
             ("attn_flash_llama_vb_fused [VALIDATION]", KER / "attn_flash_llama_vb_fused", "attn_flash_llama_vb_fused"),
             ("attn_flash_llama_2h", KER / "attn_flash_llama_2h", "attn_flash_llama_2h"),
             ("attn_flash_llama_2h_fused", KER / "attn_flash_llama_2h_fused", "attn_flash_llama_2h_fused"),
             ("mx_bench_matmul_m64_proj", KER / "mx_bench_matmul_m64_proj", "mx_bench_matmul_m64_proj"),
             ("llama_e2e_elemwise", KER / "llama_e2e_elemwise", "llama_e2e_elemwise")]


def same_binary(elf, dump):
    """The ELF is the one VCS ran: its objdump -D equals the run's .dump (minus the path line)."""
    if not dump.exists():
        return None
    a = subprocess.run(["riscv64-unknown-elf-objdump", "-D", str(elf)], capture_output=True, text=True).stdout
    return a.splitlines()[2:] == dump.read_text(errors="replace").splitlines()[2:]


def perf_cycles(text):
    out = []
    for ln in text.splitlines():
        m = re.match(r"PERF\s+(.*?)\s(\d+) cycles", ln)
        if m:
            out.append((re.sub(r"\s+", " ", m.group(1))[:28], int(m.group(2))))
    return out


def run(so, elf, mode, extra_env):
    env = dict(os.environ, GEMMINI_MODE=mode, **extra_env)
    r = subprocess.run(["spike", f"--extlib={so}", "--extension=gemmini", str(elf)], capture_output=True, text=True,
                       errors="replace", env=env, timeout=1800)
    return r.stdout + r.stderr


def kv(line):
    return {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}


def pct(m, v):
    return f"{100.0 * (m - v) / v:+6.1f}%" if v else "   n/a"


def numbers(kind, text):
    """name -> value, from what the test prints."""
    out = {}
    if kind == "perf":
        for ln in text.splitlines():
            if ln.startswith("PERF fp8"):
                d = kv(ln)
                for k in ("load", "compute", "mvout", "scales", "loops", "total"):
                    if k in d:
                        out[k] = d[k]
                break
    elif kind == "membw":
        for ln in text.splitlines():
            if ln.startswith("MEMBW"):
                out[ln.split()[1]] = kv(ln)["cyc"]
    elif kind == "phase":
        for ln in text.splitlines():
            if ln.startswith("phase"):
                for part in ln[len("phase"):].split("|"):
                    m = re.match(r"\s*(.*?)\s+(\d+)\s*$", part)
                    if m:
                        out[m.group(1)] = int(m.group(2))
    if kind in ("phase", "mesh"):
        for ln in text.splitlines():
            m = re.match(r"cycles mesh (\d+)", ln)
            if m:
                out["mesh (total)"] = int(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default=str(GEM / "software/libgemmini/libgemmini.so"))
    ap.add_argument("--set", default="", help="GEMMINI_PERF_SET overrides")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    extra = {"GEMMINI_PERF_SET": a.set} if a.set else {}
    rows = []
    for name, elf, kind, log, note in TESTS:
        if a.only and a.only not in name:
            continue
        vcs = numbers("mesh" if kind == "mesh" else kind, (VCS / CFG_MX / f"{log}.log").read_text(errors="replace"))
        env = extra
        if "old 256-bit" in name:
            sets = dict(kv.split("=") for kv in a.set.split(",") if kv)
            env = {"GEMMINI_PERF_SET": ",".join(f"{k}={v}" for k, v in {**OLD_BUS, **sets}.items())}
        mod = numbers("mesh" if kind == "mesh" else kind, run(a.so, elf, "perf", env))
        for k, v in vcs.items():
            if k in mod:
                rows.append((name, k, v, mod[k], note))
    for name, elf, log in REPLAY:
        if a.only and a.only not in name:
            continue
        rp = Path("/tmp") / f"perf_regress_{os.getpid()}.replay"
        rp.write_text(subprocess.run([sys.executable, str(REPO / "tools/rtl_replay.py"), str(VCS / CFG_MX / f"{log}.out")],
                                     capture_output=True, text=True).stdout)
        text = run(a.so, elf, "both", dict(extra, GEMMINI_PERF_REPLAY=str(rp)))
        rp.unlink()
        cmp = [tuple(int(x) for x in re.findall(r"-?\d+", ln)[:3]) for ln in text.splitlines() if ln.startswith("    rtl")]
        for i, (r, m, d) in enumerate(cmp):
            if d < -100000:
                continue   # a fence the RTL took after Gemmini was long idle
            rows.append((name, f"fence {i} (model idle - rtl retire)", r, m, "cycles, not a phase"))
    for name, elf, log in VPU_TESTS:
        if a.only and a.only not in name:
            continue
        if not elf.exists():
            continue
        match = same_binary(elf, VCS / CFG_VPU / f"{log}.dump")
        logf = VCS / CFG_VPU / f"{log}.log"
        if not logf.exists():
            continue
        if match is False:
            rows.append((name, "(ELF rebuilt since the VCS run)", 0, 0, "binary differs: not compared"))
            continue
        vcs = perf_cycles(logf.read_text(errors="replace"))
        mod = perf_cycles(run(a.so, elf, "perf", dict(extra, GEMMINI_PERF_CONFIG="e4m3_vpu")))
        for (k, v), (_, m) in zip(vcs, mod):
            rows.append((name, k, v, m, ""))
    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'test':{w}}  {'phase':28} {'VCS':>9} {'model':>9}   error")
    for name, k, v, m, note in rows:
        err = f"{m - v:+d} cyc" if note == "cycles, not a phase" else ("   --" if not v else pct(m, v))
        print(f"{name:{w}}  {k:28} {v:9d} {m:9d}  {err}" + (f"   [{note}]" if note and note != "cycles, not a phase" else ""))


if __name__ == "__main__":
    main()
