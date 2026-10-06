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
TESTS = [("128x128 (spad loop)", ISA / "matmul_tiled_fp8_128x128-baremetal", "perf",
          "matmul_tiled_fp8_128x128-baremetal", "binary differs from the VCS run: mvout VCS-equiv ~4560")]
TESTS += [(n, ISA / f"matmul_tiled_fp8_128x128_{n}-baremetal", "perf", f"matmul_tiled_fp8_128x128_{n}-baremetal", "")
          for n in DRAMLOOPS]
TESTS += [("mx_mem_bw", ISA / "mx_mem_bw-baremetal", "membw", "mx_mem_bw-baremetal", "")]
TESTS += [("llama_mlp_tiny_db", KER / "llama_mlp_tiny_db", "phase", "llama_mlp_tiny_db", "host: cpi 1 placeholder"),
          ("llama_mlp_small", KER / "llama_mlp_small", "mesh", "llama_mlp_small", "host: cpi 1 placeholder")]
REPLAY = [("llama_mlp_tiny_native_ua (replay)", KER / "llama_mlp_tiny_native_ua", "llama_mlp_tiny_native_ua")]


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
        mod = numbers("mesh" if kind == "mesh" else kind, run(a.so, elf, "perf", extra))
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
    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'test':{w}}  {'phase':28} {'VCS':>9} {'model':>9}   error")
    for name, k, v, m, note in rows:
        err = f"{m - v:+d} cyc" if note == "cycles, not a phase" else pct(m, v)
        print(f"{name:{w}}  {k:28} {v:9d} {m:9d}  {err}" + (f"   [{note}]" if note and note != "cycles, not a phase" else ""))


if __name__ == "__main__":
    main()
