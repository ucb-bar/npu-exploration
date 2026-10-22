#!/usr/bin/env python3
"""Drive src/llama_layer_mx.c over (model, format, mesh): data -> rtl_exact golden -> build -> Spike (functional + perf
model) -> one results table.

    ../../.venv/bin/python3 run_llama_mx.py --model all --fmt all            # every step, all 45 builds
    ../../.venv/bin/python3 run_llama_mx.py --model tiny --steps golden build run
    ../../.venv/bin/python3 run_llama_mx.py --model all --fmt all --steps report

steps   gen     gen/gen_llama_layer_mx.py, one blob per (model, format) (skipped when the blob exists; --force)
        golden  gen/golden_llama_layer_mx.py per mesh (writes the expect header the kernel checks itself against)
        build   make the Spike ELFs (and the MX_ROCKET ones with --rocket)
        run     Spike with libgemmini_dim<N>.so, GEMMINI_MODE=both (bit-exact math + the timing model), in parallel
        report  results/llama_mx/summary.md (+ .json) from the run logs
Logs: out/llama_mx/<elf>.log (Spike), out/llama_mx/golden_<name>.log.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gen"))
import llama_mx_common as C  # noqa: E402

NPU = HERE.parents[1]
PY = str(NPU / ".venv" / "bin" / "python3")
OUT = NPU / "out" / "baremetal"
LOGS = NPU / "out" / "llama_mx"
RESULTS = NPU / "results" / "llama_mx"
LIBG = NPU.parent / "software" / "libgemmini"
SPIKE_LIB = {16: ("libgemmini.so", "gemmini"), 8: ("libgemmini_dim8.so", "gemmini_dim8"),
             4: ("libgemmini_dim4.so", "gemmini_dim4"), 32: ("libgemmini_dim32.so", "gemmini_dim32")}
RISCV = os.environ.get("RISCV", "/bwrcq/scratch/nicorakela/radiance-cy-dev/.conda-env/riscv-tools")
# The perf model's structural parameters default to the 16x16 build (perf/params/config.h); these are the other meshes'
# (their params headers: BANK_ROWS, ACC_ROWS, MAX_BYTES 64). Measured latencies keep their DIM 16 calibration.
BANK_ROWS = {4: 8192, 8: 8192, 16: 4096, 32: 2048}
ACC_ROWS = {4: 2048, 8: 1024, 16: 512, 32: 256}


def perf_set(dim: int) -> str:
    return ",".join([f"mesh.dim={dim}", f"spad.bank_rows={BANK_ROWS[dim]}", f"acc.total_rows={ACC_ROWS[dim]}",
                     f"acc.bank_rows={ACC_ROWS[dim] // 2}", f"loop.max_block_len={64 // dim}",
                     f"dma.spad_write_bytes_per_cycle={dim}", f"st.elems_per_acc_read={2 * dim}",
                     f"sreq.cycles_per_block={64 // dim}"])


def elf(model, fmt, dim):
    return f"llama_mx_{model}_{fmt}_d{dim}"


def env():
    e = dict(os.environ, RISCV=RISCV)
    e["PATH"] = f"{RISCV}/bin:" + e["PATH"]
    return e


def sh(cmd, log=None, cwd=None):
    t0 = time.time()
    with (open(log, "w") if log else open(os.devnull, "w")) as f:
        r = subprocess.run(cmd, cwd=cwd, env=env(), stdout=f if log else None, stderr=subprocess.STDOUT if log else None)
    return r.returncode, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["tiny"])
    ap.add_argument("--fmt", nargs="+", default=["all"])
    ap.add_argument("--dim", nargs="+", type=int, default=None, help="restrict to these meshes")
    ap.add_argument("--steps", nargs="+", default=["gen", "golden", "build", "run", "report"])
    ap.add_argument("--force", action="store_true", help="regenerate blobs / goldens that exist")
    ap.add_argument("--rocket", action="store_true", help="also build the MX_ROCKET ELFs")
    ap.add_argument("--jobs", type=int, default=24, help="parallel Spike runs")
    ap.add_argument("--perf-config", default="e4m3_vpu")
    a = ap.parse_args()
    models = [k for k in C.MODELS if k != "tiny"] if a.model == ["all"] else a.model
    fmts = list(C.FORMATS) if a.fmt == ["all"] else a.fmt
    combos = [(m, f, d) for m in models for f in fmts for d in C.MESHES[f] if not a.dim or d in a.dim]
    LOGS.mkdir(parents=True, exist_ok=True)

    if "gen" in a.steps:
        for m in models:
            for f in fmts:
                if a.force or not (C.DATA / f"{C.data_name(m, f)}.bin").exists():
                    rc, dt = sh([PY, "gen_llama_layer_mx.py", "--model", m, "--fmt", f],
                                LOGS / f"gen_{C.data_name(m, f)}.log", cwd=HERE / "gen")
                    print(f"gen    {C.data_name(m, f):34s} {'ok' if rc == 0 else 'FAILED'} ({dt:.0f}s)", flush=True)
                    if rc:
                        return 1
    if "golden" in a.steps:
        for m, f, d in combos:
            if a.force or not (C.DATA / f"{C.expect_name(m, f, d)}.h").exists():
                rc, dt = sh([PY, "golden_llama_layer_mx.py", "--model", m, "--fmt", f, "--dim", str(d)],
                            LOGS / f"golden_{elf(m, f, d)}.log", cwd=HERE / "gen")
                print(f"golden {elf(m, f, d):34s} {'ok' if rc == 0 else 'FAILED'} ({dt:.0f}s)", flush=True)
                if rc:
                    return 1
    if "build" in a.steps:
        targets = [str(OUT / "spike" / elf(*c)) for c in combos]
        if a.rocket:
            targets += [str(OUT / "mx_rocket" / elf(*c)) for c in combos]
        rc, dt = sh(["make", "-j", "16", f"OUT={OUT}"] + targets, LOGS / "build.log", cwd=HERE)
        print(f"build  {len(targets)} ELFs {'ok' if rc == 0 else 'FAILED (out/llama_mx/build.log)'} ({dt:.0f}s)", flush=True)
        if rc:
            return 1
    if "run" in a.steps:
        def one(c):
            lib, ext = SPIKE_LIB[c[2]]
            e = dict(env(), GEMMINI_MODE="both", GEMMINI_PERF_CONFIG=a.perf_config, GEMMINI_PERF_SET=perf_set(c[2]))
            t0 = time.time()
            with open(LOGS / f"{elf(*c)}.log", "w") as f:
                r = subprocess.run(["spike", f"--extlib={LIBG / lib}", f"--extension={ext}", str(OUT / "spike" / elf(*c))],
                                   env=e, stdout=f, stderr=subprocess.STDOUT)
            ok = "PASSED" in (LOGS / f"{elf(*c)}.log").read_text()
            print(f"spike  {elf(*c):34s} {'PASSED' if ok and r.returncode == 0 else 'FAILED'} ({time.time() - t0:.0f}s)",
                  flush=True)
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(one, combos))
    if "report" in a.steps:
        report(combos, a.perf_config)
    return 0


def parse(log: Path):
    t = log.read_text() if log.exists() else ""
    r = {"status": "PASSED" if re.search(r"^llama_layer_mx .* PASSED", t, re.M) else ("FAILED" if t else "not run")}
    m = re.search(r"\((\d+)/(\d+) stages match", t)
    r["stages"] = f"{m.group(1)}/{m.group(2)}" if m else "-"
    m = re.search(r"PERF layer: (\d+) cycles from launch to h_out; mesh ideal (\d+) -> util ([\d.]+)%", t)
    if m:
        r.update(cycles=int(m.group(1)), ideal=int(m.group(2)), util=float(m.group(3)))
    r["phases"] = {k.strip(): int(v) for k, v in re.findall(r"PERF phase (.+?)\s+(\d+) cycles", t)}
    for k in ("Yattn", "Ymlp", "h_out"):
        m = re.search(rf"accuracy  {k}\s+rel_fro\s+(\d+) ppm", t)
        if m:
            r[f"acc_{k}"] = int(m.group(1)) / 1e4
    m = re.search(r"layer update h_out - h_pre: (\d+) ppm", t)
    if m:
        r["acc_update"] = int(m.group(1)) / 1e4
    return r


def report(combos, perf_config):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, js = [], {}
    for m, f, d in combos:
        r = parse(LOGS / f"{elf(m, f, d)}.log")
        js[elf(m, f, d)] = dict(model=m, fmt=f, dim=d, **r)
        ph = r.get("phases", {})
        att = ph.get("attention (E4M3)")
        rows.append(f"| {C.MODELS[m].name} | {f} | {d}x{d} | {r['status']} | {r['stages']} | "
                    + (f"{r['cycles']:,} | {r['ideal']:,} | {r['util']:.1f}% | " if "cycles" in r else "- | - | - | ")
                    + (f"{100 * att / r['cycles']:.0f}% | " if att and "cycles" in r else "- | ")
                    + (f"{r['acc_update']:.2f}% |" if "acc_update" in r else "- |"))
    md = ("# llama_layer_mx: one decoder layer per (model, format, mesh)\n\n"
          f"Spike + libgemmini perf model (`GEMMINI_MODE=both`, `GEMMINI_PERF_CONFIG={perf_config}`). "
          "`stages` = stage hashes equal to the rtl_exact golden (gen/golden_llama_layer_mx.py). Cycles from the first "
          "command to h_out in DRAM; `ideal` = the layer's mesh-ideal cycles at the format's rate (FP4 / FP6 LUT: the "
          "quad mesh, 4x E4M3). `accuracy` = layer update (h_out - h_pre) vs the fp64 layer on the same BF16 input "
          "(synthetic weights).\n\n"
          "| model | fmt | mesh | result | stages | cycles | ideal | util | attention share | accuracy |\n"
          "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    (RESULTS / "summary.md").write_text(md)
    (RESULTS / "summary.json").write_text(json.dumps(js, indent=1))
    print(md)


if __name__ == "__main__":
    raise SystemExit(main())
