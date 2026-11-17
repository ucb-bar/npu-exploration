#!/usr/bin/env python3
"""Build/run native complete-phase overlap guards and seven projection regression on a matching Spike model.
No FPGA, remote commands, checkpoint downloads or repository configuration edits.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

K = Path(__file__).resolve().parent.parent
NPU = K.parents[1]
GEMMINI = NPU.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--riscv", type=Path, default=os.environ.get("RISCV"))
    p.add_argument("--rocc", type=Path, default=GEMMINI / "software/gemmini-rocc-tests")
    p.add_argument(
        "--library", type=Path, default=GEMMINI / "software/libgemmini/libgemmini_mxbf16_dim16.so"
    )
    p.add_argument("--extension", default="gemmini_mxbf16_dim16")
    p.add_argument("--out", type=Path, default=K / "build/micromix-layer-spike")
    a = p.parse_args()
    if not a.riscv:
        p.error("Set RISCV or pass --riscv to the toolchain prefix")
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    rocc = a.rocc.resolve()
    b = rocc / "riscv-tests/benchmarks/common"
    library = a.library.resolve()
    elf = a.out / "projection.elf"
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if not library.is_file():
        p.error("Missing matching model library: " + str(library))
    cc = a.riscv / "bin/riscv64-unknown-elf-gcc"
    spike = a.riscv / "bin/spike"
    flags = [
        "-std=gnu11",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fno-common",
        "-fno-tree-loop-distribute-patterns",
        "-DPREALLOCATE=1",
        "-DMULTITHREAD=1",
        "-mcmodel=medany",
        "-fno-builtin-printf",
        "-march=rv64gc",
        "-DBAREMETAL=1",
        "-DSPIKE_SIM",
    ]
    for inc in (K / "include", rocc, rocc / "riscv-tests", rocc / "riscv-tests/env", b):
        flags += ["-I" + str(inc)]
    command = [
        str(cc),
        *flags,
        "-nostdlib",
        "-nostartfiles",
        "-static",
        "-T",
        str(b / "test.ld"),
        str(K / "tests/test_micromix_layer_projection.c"),
        str(K / "tests/test_micromix_projection_other.c"),
        *map(str, b.glob("*.c")),
        *map(str, b.glob("*.S")),
        "-lm",
        "-lgcc",
        "-o",
        str(elf),
    ]
    for order in ([], ["-DMXMM_LEGACY_FIRST=1"]):
        subprocess.run(
            [
                str(cc),
                *flags,
                *order,
                "-fsyntax-only",
                str(K / "tests/test_micromix_include_order.c"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    with (a.out / "compile.log").open("w") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
    run = [str(spike), "--extlib=" + str(library), "--extension=" + a.extension, str(elf)]
    with (a.out / "spike.log").open("w") as f:
        subprocess.run(run, stdout=f, stderr=subprocess.STDOUT, check=True)
    text = (a.out / "spike.log").read_text()
    print(text, end="")
    if (
        text.count("MICROMIX_PROJECTION ") != 2
        or text.count("canonical_bits=0 prepared_bits=0") != 2
        or "MICROMIX_LAYER_PROJECTION words=57344 future_consumer_guards=2 bit_mismatches=0 pass=1"
        not in text
    ):
        raise RuntimeError("Incomplete or incorrect projection proof")
    pins = {str(f.relative_to(NPU)): sha(f) for f in (K / "include").glob("mx*micromix*.h")}
    pins["tests/test_micromix_projection.c"] = sha(K / "tests/test_micromix_projection.c")
    pins["tests/test_micromix_layer_projection.c"] = sha(
        K / "tests/test_micromix_layer_projection.c"
    )
    (a.out / "results.json").write_text(
        json.dumps(
            {
                "pass": True,
                "synthetic": True,
                "timing_usable": False,
                "independent_reference": "integer sum of explicitly decoded signed unit operands",
                "compiler_command": command,
                "run_command": run,
                "library_sha256": sha(library),
                "elf_sha256": sha(elf),
                "log_sha256": sha(a.out / "spike.log"),
                "sources": pins,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
