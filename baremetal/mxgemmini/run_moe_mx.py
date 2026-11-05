#!/usr/bin/env python3
"""Drive src/moe_layer_mx.c (the gpt-oss-20b MoE block) over (input, expert format, mesh): data -> rtl_exact golden ->
build -> Spike (functional + perf model) -> one results table.

    ../../.venv/bin/python3 run_moe_mx.py --cfg all                          # every step, all 48 builds
    ../../.venv/bin/python3 run_moe_mx.py --cfg gptoss20b_t1 --fmt fp4 --dim 16 --steps golden build run

steps   gen     gen/gen_moe_mx.py: the expert blobs (once per format) and the inputs (skipped when present; --force)
        golden  gen/golden_moe_mx.py per build (writes the expect header the kernel checks itself against)
        build   make the Spike and MX_ROCKET ELFs
        run     Spike with libgemmini_dim<N>.so, GEMMINI_MODE=both (bit-exact math + the timing model), in parallel
        report  results/moe_mx/summary.md (+ .json) from the run logs
Logs: out/moe_mx/<elf>.log (Spike), out/moe_mx/golden_<elf>.log.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gen"))
import moe_mx_common as C  # noqa: E402
from run_llama_mx import NPU, PY, OUT, LIBG, SPIKE_LIB, PERF_PRESET, env, sh  # noqa: E402

LOGS = NPU / "out" / "moe_mx"
RESULTS = NPU / "results" / "moe_mx"


def elf(cfg, fmt, dim):
    return f"moe_mx_{cfg}_{fmt}_d{dim}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", nargs="+", default=["all"])
    ap.add_argument("--fmt", nargs="+", default=list(C.FORMATS))
    ap.add_argument("--dim", nargs="+", type=int, default=None, help="restrict to these meshes")
    ap.add_argument("--steps", nargs="+", default=["gen", "golden", "build", "run", "report"])
    ap.add_argument("--force", action="store_true", help="regenerate data / goldens that exist")
    ap.add_argument("--jobs", type=int, default=24, help="parallel Spike runs")
    ap.add_argument("--perf-config", default=None, help="one preset for every run (default: run_llama_mx PERF_PRESET per format and mesh)")
    a = ap.parse_args()
    cfgs = list(C.CFGS) if a.cfg == ["all"] else a.cfg
    combos = [(c, f, d) for c in cfgs for f in a.fmt for d in C.MESHES[f] if not a.dim or d in a.dim]
    LOGS.mkdir(parents=True, exist_ok=True)

    if "gen" in a.steps:
        for f in a.fmt:
            if a.force or not (C.DATA / f"{C.experts_name(f)}.bin").exists():
                rc, dt = sh([PY, "gen_moe_mx.py", "--experts", f], LOGS / f"gen_experts_{f}.log", cwd=HERE / "gen")
                print(f"gen    experts {f} {'ok' if rc == 0 else 'FAILED'} ({dt:.0f}s)", flush=True)
                if rc:
                    return 1
        todo = [c for c in cfgs if a.force or not (C.DATA / f"{C.cfg_name(c)}.bin").exists()]
        if todo:
            rc, dt = sh([PY, "gen_moe_mx.py", "--cfg", *todo], LOGS / "gen_cfg.log", cwd=HERE / "gen")
            print(f"gen    {len(todo)} inputs {'ok' if rc == 0 else 'FAILED'} ({dt:.0f}s)", flush=True)
            if rc:
                return 1
    if "golden" in a.steps:
        for c, f, d in combos:
            if a.force or not (C.DATA / f"{C.expect_name(c, f, d)}.h").exists():
                rc, dt = sh([PY, "golden_moe_mx.py", "--cfg", c, "--fmt", f, "--dim", str(d)],
                            LOGS / f"golden_{elf(c, f, d)}.log", cwd=HERE / "gen")
                print(f"golden {elf(c, f, d):34s} {'ok' if rc == 0 else 'FAILED'} ({dt:.0f}s)", flush=True)
                if rc:
                    return 1
    if "build" in a.steps:
        targets = [str(OUT / t / elf(*c)) for c in combos for t in ("spike", "mx_rocket")]
        rc, dt = sh(["make", "-j", "16", f"OUT={OUT}"] + targets, LOGS / "build.log", cwd=HERE)
        print(f"build  {len(targets)} ELFs {'ok' if rc == 0 else 'FAILED (out/moe_mx/build.log)'} ({dt:.0f}s)", flush=True)
        if rc:
            return 1
    if "run" in a.steps:
        def one(c):
            lib, ext = SPIKE_LIB[c[2]]
            e = dict(env(), GEMMINI_MODE="both", GEMMINI_PERF_CONFIG=a.perf_config or PERF_PRESET[c[1], c[2]])
            t0 = time.time()
            with open(LOGS / f"{elf(*c)}.log", "w") as fh:
                r = subprocess.run(["spike", f"--extlib={LIBG / lib}", f"--extension={ext}", str(OUT / "spike" / elf(*c))],
                                   env=e, stdout=fh, stderr=subprocess.STDOUT)
            ok = re.search(r"^moe_layer_mx .* PASSED", (LOGS / f"{elf(*c)}.log").read_text(), re.M)
            print(f"spike  {elf(*c):34s} {'PASSED' if ok and r.returncode == 0 else 'FAILED'} ({time.time() - t0:.0f}s)",
                  flush=True)
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(one, combos))
    if "report" in a.steps:
        report(combos, a.perf_config)
    return 0


def parse(log: Path):
    t = log.read_text() if log.exists() else ""
    r = {"status": "PASSED" if re.search(r"^moe_layer_mx .* PASSED", t, re.M) else ("FAILED" if t else "not run")}
    m = re.search(r"\((\d+)/(\d+) stages match", t)
    r["stages"] = f"{m.group(1)}/{m.group(2)}" if m else "-"
    m = re.search(r"PERF layer: (\d+) cycles from launch to h_out; mesh ideal (\d+) -> util ([\d.]+)%", t)
    if m:
        r.update(cycles=int(m.group(1)), ideal=int(m.group(2)), util=float(m.group(3)))
    m = re.search(r"PERF experts: (\d+) touched, (\d+) weight bytes .*expert matmuls (\d+) cycles -> ([\d.]+) B/cycle", t)
    if m:
        r.update(experts=int(m.group(1)), wbytes=int(m.group(2)), mm_cycles=int(m.group(3)), bpc=float(m.group(4)))
    r["phases"] = {k.strip(): int(v) for k, v in re.findall(r"PERF phase (.+?)\s+(\d+) cycles", t)}
    m = re.search(r"update h_out - h: (\d+) ppm", t)
    if m:
        r["acc_update"] = int(m.group(1)) / 1e4
    return r


def report(combos, perf_config):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, js = [], {}
    for c, f, d in combos:
        r = parse(LOGS / f"{elf(c, f, d)}.log")
        js[elf(c, f, d)] = dict(cfg=c, T=C.CFGS[c].T, rho=C.CFGS[c].rho, fmt=f, dim=d, **r)
        T = C.CFGS[c].T
        rows.append(f"| {c} | {T} | {f} | {d}x{d} | {r['status']} | {r['stages']} | "
                    + (f"{r['experts']} | {r['wbytes'] / 1e6:.0f} | {r['bpc']:.2f} | " if "experts" in r else "- | - | - | ")
                    + (f"{r['cycles']:,} | {r['cycles'] // T:,} | {r['util']:.1f}% | " if "cycles" in r else "- | - | - | ")
                    + (f"{r['acc_update']:.2f}% |" if "acc_update" in r else "- |"))
    md = ("# moe_layer_mx: the gpt-oss-20b MoE block per (input, expert format, mesh)\n\n"
          "Spike + libgemmini perf model (`GEMMINI_MODE=both`, `GEMMINI_PERF_CONFIG` = "
          + (f"`{perf_config}`" if perf_config else "the RTL config of each format and mesh, run_llama_mx.py PERF_PRESET")
          + "). `stages` = stage "
          "hashes equal to the rtl_exact golden (gen/golden_moe_mx.py). `experts` = distinct experts the T tokens routed "
          "to; `MB` = their weight bytes (codes + scales); `B/c` = those bytes / the expert matmul cycles. Cycles from the "
          "first command to h_out in DRAM; `util` = mesh-ideal cycles of the issued matmuls / measured. `accuracy` = "
          "the update (h_out - h) vs the fp64 block (synthetic weights).\n\n"
          "| input | T | fmt | mesh | result | stages | experts | MB | B/c | cycles | per token | util | accuracy |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    (RESULTS / "summary.md").write_text(md)
    (RESULTS / "summary.json").write_text(json.dumps(js, indent=1))
    print(md)


if __name__ == "__main__":
    raise SystemExit(main())
