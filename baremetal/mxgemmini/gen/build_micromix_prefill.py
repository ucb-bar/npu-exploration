#!/usr/bin/env python3
"""Build a native one-layer MicroMX prefill example from supplied frozen data.
No checkpoint acquisition or FPGA actions. Full-vector references are mandatory;
outputs remain explicitly unqualified for target timing until measured on device.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import shutil
from pathlib import Path

K = Path(__file__).resolve().parent.parent
NPU = K.parents[1]
G = NPU.parent
MODELS = {"tinyllama": 0, "qwen25_7b": 1, "llama2_7b": 2}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--mix", choices=("mix50", "mix625"), required=True)
    p.add_argument(
        "--capture", type=Path, required=True, help="E2E capture header; same-stem .bin required"
    )
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reference-manifest", type=Path, required=True)
    p.add_argument(
        "--reuse-object-dir",
        type=Path,
        help="Optional previously built readonly binary objects with matching symbols/data",
    )
    p.add_argument("--target", choices=("spike", "fpga"), default="spike")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--riscv", type=Path, default=os.environ.get("RISCV"))
    p.add_argument("--rocc", type=Path, default=G / "software/gemmini-rocc-tests")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not a.riscv or a.repeats < 1:
        p.error("RISCV prefix and positive repetitions required")
    if a.out.exists() or a.out.is_symlink():
        p.error("Refusing existing build namespace")
    headers = [a.capture.resolve(), a.weights.resolve(), a.reference.resolve()]
    blobs = [x.with_suffix(".bin") for x in headers]
    for x in [*headers, *blobs, a.reference_manifest]:
        if not x.is_file():
            p.error("Missing supplied input: " + str(x))
    ref = json.loads(a.reference_manifest.read_text())
    expected = ref.get("blob_sha256", ref.get("sha256"))
    if (
        ref.get("mix") != a.mix
        or expected != sha(blobs[2])
        or ref.get("header_sha256") != sha(headers[2])
    ):
        p.error("Frozen numeric reference manifest mismatch")
    data_pins = {str(x): sha(x) for x in [*headers, *blobs, a.reference_manifest]}
    raw = blobs[2].read_bytes()
    if len(raw) % 2 or any((x[0] & 0x7F80) == 0x7F80 for x in struct.iter_unpack("<H", raw)):
        p.error("Nonfinite/malformed BF16 reference")
    builder_sha = sha(__file__)
    a.out = a.out.resolve()
    a.out.mkdir(parents=True)
    shutil.copyfile(__file__, a.out / "builder.py")
    if sha(a.out / "builder.py") != builder_sha:
        raise ValueError("Builder changed during snapshot")
    (a.out / "reservation.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=2)
        + "\n"
    )
    T = a.riscv / "bin"
    rocc = a.rocc.resolve()
    B = rocc / "riscv-tests/benchmarks/common"
    objects = []
    for blob in blobs:
        obj = a.out / (blob.stem + ".o")
        if a.reuse_object_dir:
            obj = (a.reuse_object_dir / (blob.stem + ".o")).resolve()
            if not obj.is_file():
                raise ValueError("Missing reusable object " + str(obj))
            # Independently verify its .data contents, not just its filename/symbol.
            extract = a.out / (blob.stem + ".verify.bin")
            subprocess.run(
                [
                    str(T / "riscv64-unknown-elf-objcopy"),
                    "-O",
                    "binary",
                    "--only-section=.data",
                    str(obj),
                    str(extract),
                ],
                check=True,
            )
            if sha(extract) != sha(blob):
                raise ValueError("Object data differs from supplied frozen blob")
            extract.unlink()
        else:
            subprocess.run(
                [
                    str(T / "riscv64-unknown-elf-objcopy"),
                    "-I",
                    "binary",
                    "-O",
                    "elf64-littleriscv",
                    "-B",
                    "riscv",
                    "--set-section-alignment",
                    ".data=64",
                    blob.name,
                    str(obj),
                ],
                cwd=blob.parent,
                check=True,
            )
        objects.append(obj)
    object_pins = {str(x): sha(x) for x in objects}
    flags = [
        "-std=gnu11",
        "-O2",
        "-ffast-math",
        "-fno-common",
        "-fno-builtin-printf",
        "-fno-tree-loop-distribute-patterns",
        "-DPREALLOCATE=1",
        "-DMULTITHREAD=1",
        "-mcmodel=medany",
        "-march=rv64gc",
        "-DBAREMETAL=1",
        "-D" + ("SPIKE_SIM" if a.target == "spike" else "MX_ROCKET"),
        "-DMICROMIX_MODEL=" + str(MODELS[a.model]),
        "-DMICROMIX_MIX=" + str(1 if a.mix == "mix50" else 2),
        "-DMICROMIX_REPEATS=" + str(a.repeats),
        "-DMXMM_NUMERIC_MODE=1",
        '-DMICROMIX_CAPTURE_HEADER="' + str(headers[0]) + '"',
        '-DMXMM_DATA_H="' + str(headers[1]) + '"',
        '-DMXMM_NUMERIC_REF_H="' + str(headers[2]) + '"',
    ]
    for inc in (K / "include", K / "src", rocc, rocc / "riscv-tests", rocc / "riscv-tests/env", B):
        flags += ["-I" + str(inc)]
    source = (
        K
        / "src"
        / (
            "micromix_prefill_tinyllama.c"
            if a.model == "tinyllama"
            else "micromix_prefill_generic.c"
        )
    )
    elf = a.out / "prefill.elf"
    cmd = [
        str(T / "riscv64-unknown-elf-gcc"),
        *flags,
        "-nostdlib",
        "-nostartfiles",
        "-static",
        "-T",
        str(B / "test.ld"),
        str(source),
        *map(str, objects),
        *map(str, B.glob("*.c")),
        *map(str, B.glob("*.S")),
        "-lm",
        "-lgcc",
        "-o",
        str(elf),
    ]
    dep = subprocess.check_output([cmd[0], *flags, "-M", str(source)], text=True)
    initial = {Path(x).resolve() for x in dep.replace("\\\n", " ").partition(":")[2].split()}
    capsule = a.out / "source"
    original_sources = {}
    for path in initial:
        try:
            relative = path.relative_to(K)
        except ValueError:
            continue
        dst = capsule / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
        original_sources[str(path)] = sha(path)
    source = capsule / source.relative_to(K)
    flags = ["-I" + str(capsule / "include"), "-I" + str(capsule / "src"), *flags]
    cmd = [
        str(T / "riscv64-unknown-elf-gcc"),
        *flags,
        "-nostdlib",
        "-nostartfiles",
        "-static",
        "-T",
        str(B / "test.ld"),
        str(source),
        *map(str, objects),
        *map(str, B.glob("*.c")),
        *map(str, B.glob("*.S")),
        "-lm",
        "-lgcc",
        "-o",
        str(elf),
    ]
    dep = subprocess.check_output([cmd[0], *flags, "-M", str(source)], text=True)
    (a.out / "dependencies.txt").write_text(dep)
    deps = {Path(x).resolve() for x in dep.replace("\\\n", " ").partition(":")[2].split()}
    deps.update(Path(x).resolve() for x in cmd if x.endswith((".c", ".S", ".ld")))
    deps.update(Path(x) for x in object_pins)
    deps.add((a.out / "builder.py").resolve())
    deps.add(Path(cmd[0]).resolve())
    for component in ("cc1", "as", "collect2", "ld"):
        tool = subprocess.check_output([cmd[0], "-print-prog-name=" + component], text=True).strip()
        path = Path(tool)
        if path.is_file():
            deps.add(path.resolve())
    for archive in ("libm.a", "libgcc.a", "libc.a"):
        path = Path(
            subprocess.check_output([cmd[0], "-print-file-name=" + archive], text=True).strip()
        )
        if path.is_file():
            deps.add(path.resolve())
    pins = {str(x): sha(x) for x in deps}
    with (a.out / "compile.log").open("x") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    symbols = subprocess.check_output(
        [str(T / "riscv64-unknown-elf-nm"), "-n", str(elf)], text=True
    )
    starts = re.findall(r"^([0-9a-f]+) \w (_binary_\w+_bin_start)$", symbols, re.M)
    if len(starts) != 3 or any(int(v, 16) % 64 for v, n in starts):
        raise ValueError("Linked data base alignment failure")
    if (
        any(sha(x) != v for x, v in pins.items())
        or sha(__file__) != builder_sha
        or any(sha(x) != v for x, v in object_pins.items())
        or any(sha(x) != v for x, v in data_pins.items())
    ):
        raise ValueError("Compiler input changed")
    result = dict(
        schema_version=1,
        model=a.model,
        mix=a.mix,
        target=a.target,
        timing_qualified=False,
        source_sha256=sha(source),
        elf_sha256=sha(elf),
        command=cmd,
        source_dependencies=pins,
        original_sources=original_sources,
        data=data_pins,
        objects=object_pins,
        reference_words=len(raw) // 2,
        reference_manifest_sha256=sha(a.reference_manifest),
        builder_sha256=builder_sha,
        builder_snapshot_sha256=sha(a.out / "builder.py"),
        linked_data_alignment=64,
        compile_inputs_verified_before_and_after=True,
    )
    (a.out / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(elf)


if __name__ == "__main__":
    main()
