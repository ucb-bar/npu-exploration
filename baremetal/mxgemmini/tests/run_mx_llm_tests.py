#!/usr/bin/env python3
"""Self-contained strict CPU attention/cache tests; optional RV64 Spike replay.

No model files, network access or FPGA lifecycle. Explicit --output-root keeps
small build evidence; otherwise a temporary directory is removed on completion.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

D = Path(__file__).resolve().parent
K = D.parent
G = D.parents[3]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def execute(command, log):
    p = subprocess.run(
        [str(x) for x in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    log.write_text(p.stdout)
    if p.returncode:
        raise RuntimeError("Command failed: " + repr(command) + "\n" + p.stdout[-2000:])
    return p.stdout


def run(args, out):
    tests = [
        ("attention", "test_mx_llm_attention.c", "MX_LLM_TEST"),
        ("cache", "test_mx_cache_lut.c", "MX_CACHE_TEST"),
    ]
    rows = []
    checksum = {}
    for target in ["host", "spike"] if args.spike else ["host"]:
        for name, source, marker in tests:
            flags = [
                "-std=gnu99",
                "-O2",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-fno-common",
                "-fno-tree-loop-distribute-patterns",
                "-fno-builtin-printf",
                "-I" + str(K / "include"),
            ]
            runtime = []
            if target == "host":
                cc = shlex.split(args.cc)
                flags += ["-no-pie"]
            else:
                cc = [args.riscv_cc]
                b = G / "software/gemmini-rocc-tests/riscv-tests/benchmarks/common"
                if not b.is_dir():
                    raise FileNotFoundError("Native runtime missing: " + str(b))
                flags += [
                    "-I" + str(b),
                    "-I" + str(G / "software/gemmini-rocc-tests"),
                    "-I" + str(G / "software/gemmini-rocc-tests/riscv-tests"),
                    "-I" + str(G / "software/gemmini-rocc-tests/riscv-tests/env"),
                    "-DBAREMETAL=1",
                    "-DSPIKE_SIM",
                    "-DPREALLOCATE=1",
                    "-DMULTITHREAD=1",
                    "-mcmodel=medany",
                    "-march=rv64gc",
                    "-nostdlib",
                    "-nostartfiles",
                    "-static",
                    "-T",
                    str(b / "test.ld"),
                ]
                runtime = [*b.glob("*.c"), *b.glob("*.S")]
            elf = out / (target + "-" + name + ".elf")
            cmd = [
                *cc,
                *flags,
                D / source,
                *runtime,
                "-lm",
                *(["-lgcc"] if target == "spike" else []),
                "-o",
                elf,
            ]
            execute(cmd, out / (target + "-" + name + "-build.log"))
            run = (
                [str(elf)]
                if target == "host"
                else [
                    args.spike,
                    *(
                        ["--extlib=" + str(args.gemmini_lib), "--extension=gemmini_mxbf16_dim16"]
                        if args.gemmini_lib
                        else []
                    ),
                    str(elf),
                ]
            )
            text = execute(run, out / (target + "-" + name + ".log"))
            matches = re.findall(
                r"^" + marker + r" PASS checks=([0-9]+) checksum=([0-9]+)$", text, re.M
            )
            if len(matches) != 1 or "FAIL" in text:
                raise RuntimeError("Missing complete passing test summary")
            checks, value = map(int, matches[0])
            assert checks > 0 and 0 <= value < 2**64
            if name in checksum and args.require_cross_target_bitexact:
                assert checksum[name] == value, "Host/Spike output checksum differs"
            checksum.setdefault(name, value)
            rows.append(
                dict(
                    target=target,
                    test=name,
                    checks=checks,
                    checksum=value,
                    elf_sha256=sha(elf),
                    compile_command=list(map(str, cmd)),
                    run_command=run,
                )
            )
    result = dict(
        pass_=True,
        targets=sorted({r["target"] for r in rows}),
        tests=rows,
        strict_fp=True,
        fpga_run=False,
        native_adapter_matches_direct_specializations=True,
        independent_fp64_relative_l2_limit=0.01,
        cross_target_bitexact_required=args.require_cross_target_bitexact,
        cross_target_checksums_equal=all(
            len({r["checksum"] for r in rows if r["test"] == n}) == 1
            for n in ["attention", "cache"]
        ),
        cache_checksum_nan_payloads_canonicalized=True,
        sources={
            str(p.relative_to(K)): sha(p)
            for p in [D / source for _, source, _ in tests]
            + [
                K / "include" / n
                for n in [
                    "mx_gqa_attention.h",
                    "qwen_gqa_attention.h",
                    "mx_mha_attention.h",
                    "mx_cache_lut.h",
                    "mx_llm_attention.h",
                    "mx_tiled_attention.h",
                    "mx_tiled_mha_attention.h",
                    "mx_cache_blocked.h",
                ]
            ]
        },
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cc", default=os.environ.get("CC", "cc"))
    p.add_argument("--riscv-cc", default=shutil.which("riscv64-unknown-elf-gcc"))
    p.add_argument("--spike")
    p.add_argument("--gemmini-lib", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument(
        "--require-cross-target-bitexact",
        action="store_true",
        help="Optional stronger constraint; separate libm implementations can differ on synthetic BF16 ties",
    )
    a = p.parse_args()
    if a.spike and not a.riscv_cc:
        p.error("--spike requires --riscv-cc or compiler on PATH")
    if a.gemmini_lib and not a.gemmini_lib.is_file():
        p.error("Missing Gemmini library")
    if a.output_root:
        a.output_root.mkdir(parents=True, exist_ok=False)
        run(a, a.output_root.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="mx-llm-test-") as t:
            run(a, Path(t))


if __name__ == "__main__":
    main()
