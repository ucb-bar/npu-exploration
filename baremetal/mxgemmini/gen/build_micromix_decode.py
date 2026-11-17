#!/usr/bin/env python3
"""Build an offline, strict real-layer decode replay from explicit exported assets.

No downloads, checkpoint acquisition, board access or default local data paths.
A reference must independently describe the requested model and eight tokens.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

SDK = Path(__file__).resolve().parents[1]
MODELS = {
    "tiny": ("llama_decode_capture", "llama_micromix", "llama_decode_expect", "products_v2"),
    "qwen": (
        "qwen_layer_e2e",
        "qwen_micromix",
        "qwen_decode_expect",
        "products_v2_qwen7b_v2_exp32cr",
    ),
    "llama": (
        "llama7b_layer_e2e",
        "llama7b_micromix",
        "llama7b_decode_expect",
        "products_v2_llama2_7b_exp32cr_v1",
    ),
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reference_hout_checksums(raw, widths):
    """Word-FNV64 of each independent Hout vector, matching the guest protocol.

    Reference vectors are token-major, with Hout last among eight stages. This
    checks full BF16 words rather than bytes; the historical guest seed is kept.
    """
    if len(widths) != 8 or any(type(w) is not int or w <= 0 for w in widths):
        raise ValueError("Eight positive stage widths required")
    stride = sum(widths) * 2
    if len(raw) != 8 * stride:
        raise ValueError("Eight complete reference tokens required")
    offset = sum(widths[:-1]) * 2
    result = []
    for step in range(8):
        value = 1469598103934665603
        start = step * stride + offset
        for (word,) in struct.iter_unpack("<H", raw[start : start + widths[-1] * 2]):
            value = ((value ^ word) * 1099511628211) & ((1 << 64) - 1)
        result.append(value)
    return result


def data_sha(p):
    """Hash a binary object's actual .data section, not its file name."""
    b = Path(p).read_bytes()
    if b[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Require little-endian ELF64 object")
    off = struct.unpack_from("<Q", b, 40)[0]
    size, n, strings = struct.unpack_from("<HHH", b, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", b, off + i * size) for i in range(n)]
    st = sections[strings]
    names = b[st[4] : st[4] + st[5]]
    for s in sections:
        if names[s[0] :].split(b"\0", 1)[0] == b".data":
            return hashlib.sha256(memoryview(b)[s[4] : s[4] + s[5]]).hexdigest()
    raise ValueError("Missing .data")


def build(a):
    out = a.output_root.resolve()
    if out.exists():
        raise FileExistsError("Use a fresh --output-root; existing outputs are immutable")
    issue = getattr(a, "decode_issue_mode", None)
    if issue is not None and (
        type(issue) is not int or issue not in (0, 1) or (issue and a.model != "qwen")
    ):
        raise ValueError("Typed mixed Qwen issue0/1 required")
    cap, weight, ref, version = MODELS[a.model]
    weight += "_" + a.mix
    assets = [
        a.capture_header,
        a.capture_bin,
        a.weights_header,
        a.weights_bin,
        a.reference_bin,
        a.reference_manifest,
    ]
    for p in assets:
        if not p.is_file():
            raise ValueError("Missing explicit asset: " + str(p))
    if [
        a.capture_header.stem,
        a.capture_bin.stem,
        a.weights_header.stem,
        a.weights_bin.stem,
        a.reference_bin.stem,
    ] != [cap, cap, weight, weight, ref]:
        raise ValueError("Exported asset basenames must match selected model/mix")
    initial_data_pins = {str(p.resolve()): sha(p) for p in assets}
    meta = json.loads(a.reference_manifest.read_text())
    if (
        meta.get("reference_version") != version
        or meta.get("sha256") != sha(a.reference_bin)
        or meta.get("steps", 8) < 8
    ):
        raise ValueError("Wrong independent reference family/hash/steps")
    if meta.get("mix") != a.mix or meta.get("batch") != 1 or meta.get("layer") != 5:
        raise ValueError("Wrong reference mix/batch/layer")
    if a.model == "qwen" and meta.get("model") != "Qwen2.5-7B":
        raise ValueError("Wrong Qwen model")
    if a.model == "llama" and meta.get("model") != "NousResearch/Llama-2-7b-hf":
        raise ValueError("Wrong Llama model")
    widths = {
        "tiny": [2048, 256, 256, 2048, 5632, 5632, 2048, 2048],
        "qwen": [3584, 512, 512, 3584, 18944, 18944, 3584, 3584],
        "llama": [4096, 4096, 4096, 4096, 11008, 11008, 4096, 4096],
    }
    if meta.get("widths") != widths[a.model]:
        raise ValueError("Wrong stage-vector geometry")
    raw = a.reference_bin.read_bytes()
    if len(raw) != 8 * sum(widths[a.model]) * 2 or any(
        (w[0] & 0x7F80) == 0x7F80 for w in struct.iter_unpack("<H", raw)
    ):
        raise ValueError("Malformed/nonfinite full BF16 reference")
    reference_checksums = reference_hout_checksums(raw, widths[a.model])
    if a.repeats < 1:
        raise ValueError("Positive repeats required")
    profile_path = SDK / "gen/micromix_profiles.json"
    profile_sha = sha(profile_path)
    profiles = json.loads(profile_path.read_text())
    model_name = {"tiny": "tinyllama", "qwen": "qwen25_7b", "llama": "llama2_7b"}[a.model]
    selected = [
        r
        for r in profiles["rows"]
        if r["model"] == model_name and r["regime"] == "decode" and r["mix"] == a.mix
    ]
    if (
        len(selected) != 1
        or selected[0].get("qualified") is not True
        or selected[0].get("source") != "fpga"
    ):
        raise ValueError("No qualified native decode profile")
    profile = selected[0]
    if any(
        type(profile[k]) is not int
        for k in [
            "nc",
            "run",
            "fence_groups",
            "integer_pack",
            "prepared_reuse",
            "cache_lut",
            "physical_padding",
            "attention_mode",
        ]
    ):
        raise ValueError("Typed integer profile required")
    if [
        profile[k]
        for k in [
            "nc",
            "run",
            "fence_groups",
            "integer_pack",
            "prepared_reuse",
            "cache_lut",
            "physical_padding",
        ]
    ] != [256, 8, 1, 1, 0, int(a.model == "llama"), int(a.model == "llama")]:
        raise ValueError("Native decode template does not implement this profile layout/control")
    chosen_mode = profile["attention_mode"] if a.attention_mode is None else a.attention_mode
    chosen_cache = (
        profile.get("cache_time_tile", 0) if a.cache_time_tile is None else a.cache_time_tile
    )
    chosen_issue = (
        profile.get("decode_issue_mode", 0) if a.decode_issue_mode is None else a.decode_issue_mode
    )
    if (
        type(chosen_issue) is not int
        or chosen_issue not in (0, 1)
        or (chosen_issue and a.model != "qwen")
    ):
        raise ValueError("Typed mixed Qwen decode issue0/1 required")
    explicit_override = (
        a.attention_mode is not None
        or a.cache_time_tile is not None
        or a.decode_issue_mode is not None
    )
    if (
        type(chosen_cache) is not int
        or chosen_cache not in (0, 16)
        or (chosen_cache and a.model != "llama")
    ):
        raise ValueError("Cache time16 is an explicit Llama-only control")

    rocc = a.rocc_root.resolve()
    bench = rocc / "riscv-tests/benchmarks/common"
    if any(t != "host" for t in a.targets) and not (bench / "crt.S").exists():
        raise ValueError("Supply matching --rocc-root for RV64 runtime")
    # All input/reference/argument validation above precedes any write/tool call.
    if any(sha(p) != h for p, h in initial_data_pins.items()) or sha(profile_path) != profile_sha:
        raise ValueError("Input changed during validation")
    data_pins = initial_data_pins
    builder_sha = sha(__file__)
    originals = [
        SDK / "src/micromix_decode.c",
        *sorted((SDK / "src").glob("micromix_decode_*.h")),
        *sorted((SDK / "include").glob("*.h")),
        profile_path,
    ]
    original_sources = {str(p): sha(p) for p in originals}
    if original_sources[str(profile_path)] != profile_sha:
        raise ValueError("Profile changed before snapshot")
    out.mkdir(parents=True)
    commands = []
    objects = {}
    elfs = {}
    compilation_pins = {}
    capsule = out / "source"
    shutil.copyfile(__file__, out / "builder.py")
    for p in originals:
        dst = capsule / p.relative_to(SDK)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        if sha(dst) != original_sources[str(p)]:
            raise ValueError("Source changed during snapshot")
    if sha(out / "builder.py") != builder_sha:
        raise ValueError("Builder changed during snapshot")
    (capsule / "gen").mkdir(exist_ok=True)
    shutil.copyfile(out / "builder.py", capsule / "gen/build_micromix_decode.py")

    def run(cmd, **kw):
        commands.append([str(v) for v in cmd])
        subprocess.run([str(v) for v in cmd], check=True, **kw)

    for t in a.targets:
        objcopy = a.objcopy if t == "host" else a.riscv_tool_prefix + "objcopy"
        objs = []
        for binary in [a.capture_bin, a.weights_bin, a.reference_bin]:
            obj = out / (t + "_" + binary.stem + ".o")
            cached = a.objects_dir / (t + "_" + binary.stem + ".o") if a.objects_dir else None
            if cached and cached.is_file():
                if data_sha(cached) != sha(binary):
                    raise ValueError("Cached object .data differs: " + str(cached))
                os.link(cached, obj)
            else:
                cmd = [
                    objcopy,
                    "-I",
                    "binary",
                    "-O",
                    "elf64-x86-64" if t == "host" else "elf64-littleriscv",
                    "--set-section-alignment",
                    ".data=64",
                ]
                if t != "host":
                    cmd += ["-B", "riscv"]
                run([*cmd, binary.name, obj], cwd=binary.parent)
            objects[str(obj)] = sha(obj)
            objs.append(obj)
        flags = [
            "-std=gnu99",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fno-common",
            "-fno-tree-loop-distribute-patterns",
            "-fno-builtin-printf",
            "-DMICROMIX_DECODE_" + a.model.upper(),
            "-DDECODE_STEPS=8",
            "-DDECODE_REPEATS=" + str(a.repeats),
            "-DDECODE_NC=256",
            "-DDECODE_RUN=8",
            "-DDECODE_FENCE=1",
            '-DDECODE_MIX="' + a.mix + '"',
            '-DDECODE_CAPTURE_HEADER="' + a.capture_header.name + '"',
            '-DDECODE_WEIGHT_HEADER="' + a.weights_header.name + '"',
            "-DQWEN_ENABLE_GROUPED7=1",
            "-DLLAMA7B_PAIRED_ENABLE=1",
            "-DLLAMA7B_ATTN_MODE=1",
            "-DLLAMA_MHA4_EXPLICIT_FMA=1",
        ]
        flags += [
            "-DMICROMIX_DECODE_ATTENTION_MODE=" + str(chosen_mode),
            "-DMICROMIX_DECODE_CACHE_TIME_TILE=" + str(chosen_cache),
            "-DMICROMIX_DECODE_EXPLICIT_OVERRIDE=" + str(int(explicit_override)),
            "-DMICROMIX_DECODE_ISSUE_MODE=" + str(chosen_issue),
            "-DMICROMIX_DECODE_ISSUE_OVERRIDE=" + str(int(a.decode_issue_mode is not None)),
            "-DMICROMIX_DECODE_ATTENTION_OVERRIDE=" + str(int(a.attention_mode is not None)),
        ]
        for inc in [
            capsule / "include",
            a.capture_header.parent,
            a.weights_header.parent,
            rocc,
            rocc / "riscv-tests",
            rocc / "riscv-tests/env",
            bench,
        ]:
            flags += ["-I" + str(inc)]
        if t == "host":
            cc = a.cc
            flags += ["-DDECODE_HOST", "-no-pie"]
            runtime = []
        else:
            cc = a.riscv_tool_prefix + "gcc"
            flags += [
                "-DBAREMETAL=1",
                "-DPREALLOCATE=1",
                "-DMULTITHREAD=1",
                "-mcmodel=medany",
                "-march=rv64gc",
                "-nostdlib",
                "-nostartfiles",
                "-static",
                "-T",
                str(bench / "test.ld"),
                "-D" + ("SPIKE_SIM" if t == "spike" else "MX_ROCKET"),
            ]
            runtime = [bench / "syscalls.c", bench / "crt.S"]
        elf = out / (t + ".elf")
        source = capsule / "src/micromix_decode.c"
        dep = subprocess.check_output([cc, *flags, "-M", "-MT", "unit", str(source)], text=True)
        (out / (t + ".dependencies.txt")).write_text(dep)
        deps = {Path(w).resolve() for w in dep.replace("\\\n", " ").partition(":")[2].split()}
        for runtime_source in runtime:
            runtime_dep = subprocess.check_output(
                [cc, *flags, "-M", "-MT", "unit", str(runtime_source)], text=True
            )
            deps.update(
                Path(w).resolve()
                for w in runtime_dep.replace("\\\n", " ").partition(":")[2].split()
            )
        deps.update(Path(p).resolve() for p in runtime)
        deps.update(p.resolve() for p in objs)
        deps.add((out / "builder.py").resolve())
        if t != "host":
            deps.add((bench / "test.ld").resolve())
        compiler = Path(shutil.which(str(cc)) or cc).resolve()
        deps.add(compiler)
        for component in ["cc1", "as", "collect2", "ld"]:
            tool_name = subprocess.check_output(
                [str(cc), "-print-prog-name=" + component], text=True
            ).strip()
            tool = Path(shutil.which(tool_name) or tool_name)
            if tool.is_file():
                deps.add(tool.resolve())
        for archive in ["libm.a", "libgcc.a", "libc.a", "libm.so", "libc.so"]:
            lib = Path(
                subprocess.check_output([str(cc), "-print-file-name=" + archive], text=True).strip()
            )
            if lib.is_file():
                deps.add(lib.resolve())
        pins = {str(p): sha(p) for p in deps}
        run(
            [
                cc,
                *flags,
                source,
                *objs,
                *runtime,
                "-lm",
                *(["-lgcc"] if t != "host" else []),
                "-o",
                elf,
            ]
        )
        elfs[t] = dict(path=str(elf), sha256=sha(elf))
        if a.deduplicate_elf_dir:
            old = a.deduplicate_elf_dir / (t + ".elf")
            if old.is_file() and sha(old) == sha(elf):
                tmp = elf.with_suffix(".identical-tmp")
                os.link(old, tmp)
                os.replace(tmp, elf)
                elfs[t]["identical_content_hardlink"] = str(old.resolve())
        if (
            any(sha(p) != h for p, h in pins.items())
            or any(sha(p) != h for p, h in original_sources.items())
            or any(sha(p) != h for p, h in data_pins.items())
            or sha(__file__) != builder_sha
        ):
            raise ValueError("Compiler/data/source input changed")
        compilation_pins[t] = pins
        nm = a.nm if t == "host" else a.riscv_tool_prefix + "nm"
        txt = subprocess.check_output([nm, str(elf)], text=True)
        starts = {
            m.group(2): int(m.group(1), 16)
            for m in re.finditer(r"^([0-9a-fA-F]+)\s+\w\s+(_binary_\w+_bin_start)$", txt, re.M)
        }
        if len(starts) != 3 or any(v % 64 for v in starts.values()):
            raise ValueError("All three data symbols must be64-byte aligned")
        (out / (t + ".alignment.json")).write_text(json.dumps(starts, indent=2) + "\n")
    sources = {str(p.relative_to(capsule)): sha(p) for p in capsule.rglob("*") if p.is_file()}
    manifest = dict(
        schema_version=1,
        model=a.model,
        mix=a.mix,
        reference_version=version,
        valid_rows=1,
        issued_rows=32,
        steps=8,
        repeats=a.repeats,
        passes=2,
        comparison=(
            "explicit_unqualified_override_replay"
            if explicit_override
            else "selected_profile_replay"
        ),
        profile_admission=not explicit_override,
        hardware_qualified=False,
        attention_mode=chosen_mode,
        cache_time_tile=chosen_cache,
        cache_override=a.cache_time_tile is not None,
        profile=profile,
        profile_manifest_sha256=sha(profile_path),
        attention_override=a.attention_mode is not None,
        physical_capacity=2057 if a.model == "llama" else 2056,
        logical_capacity=2056,
        score_stride=2056,
        schedule=dict(nc=256, run=8, fence_groups=1),
        fast_pack=1,
        integer_pack=1,
        cache_init_inclusive=True,
        strict_fp=True,
        fpga_run=False,
        performance_qualified=False,
        assets=data_pins,
        sources=sources,
        source_capsule=str(capsule),
        original_sources=original_sources,
        source_dependencies=compilation_pins,
        builder_sha256=builder_sha,
        builder_snapshot_sha256=sha(out / "builder.py"),
        compile_inputs_verified_before_and_after=True,
        objects=objects,
        elfs=elfs,
        commands=commands,
    )
    prefix = {"tiny": "decode", "qwen": "qwen_decode", "llama": "llama7b_padded"}[a.model]
    manifest["reference_hout_checksums"] = reference_checksums
    manifest.update(
        decode_issue_mode=chosen_issue,
        decode_issue_override=a.decode_issue_mode is not None,
        issue_override_qualified=False,
        issue_mode_by_pass=[chosen_issue, chosen_issue],
        attention_mode_by_pass=[chosen_mode, chosen_mode],
        issued_rows_by_variant=[
            dict(fp4=32, fp6=16 if chosen_issue else 32, fp8=16 if chosen_issue else 32)
            for _ in range(2)
        ],
        storage_rows=32,
        scale_pitch_rows=32,
    )
    cases = [
        dict(
            id=f"{prefix}_{a.mix}_p{variant}_r{trial}_s{step}",
            region="decode_token",
            active_harts=[0],
            warmup=trial == 0,
            expected_checksums_by_hart=[reference_checksums[step]],
        )
        for variant in range(2)
        for trial in range(a.repeats)
        for step in range(8)
    ]
    (out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    manifest["cases_sha256"] = sha(out / "cases.json")
    (out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for snapshot in capsule.rglob("*"):
        if snapshot.is_file():
            snapshot.chmod(0o444)
    (out / "builder.py").chmod(0o444)
    print(out)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--mix", choices=["mix50", "mix625"], required=True)
    for n in [
        "capture-header",
        "capture-bin",
        "weights-header",
        "weights-bin",
        "reference-bin",
        "reference-manifest",
        "output-root",
        "rocc-root",
    ]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument(
        "--deduplicate-elf-dir",
        type=Path,
        help="Retain all paths, hardlink only byte-identical complete old ELFs",
    )
    p.add_argument("--objects-dir", type=Path)
    p.add_argument(
        "--targets", nargs="+", choices=["host", "spike", "fpga"], default=["host", "spike"]
    )
    p.add_argument("--cc", default="cc")
    p.add_argument("--nm", default="nm")
    p.add_argument("--objcopy", default="objcopy")
    p.add_argument("--riscv-tool-prefix", default="riscv64-unknown-elf-")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--decode-issue-mode", type=int, choices=[0, 1])
    p.add_argument("--cache-time-tile", type=int, choices=[0, 16])
    p.add_argument("--attention-mode", type=int, choices=range(7))
    build(p.parse_args())


if __name__ == "__main__":
    main()
