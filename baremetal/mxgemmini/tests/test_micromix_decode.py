#!/usr/bin/env python3
"""Offline full-vector replay of an explicitly built native decode example."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import unittest
import tempfile
import importlib.util
from unittest.mock import patch
from pathlib import Path

SDK = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inspect(text, b):
    def records(prefix):
        return [
            dict(x.split("=", 1) for x in l.split()[1:])
            for l in text.splitlines()
            if l.startswith(prefix + " ")
        ]

    n = 2 * b["repeats"] * 8
    refs = records("DECODE_REFERENCE")
    gates = records("DECODE_GATE")
    updates = records("DECODE_UPDATE")
    regions = [
        json.loads(l.split(" ", 1)[1]) for l in text.splitlines() if l.startswith("IRIS_REGION ")
    ]
    expected = {
        (p, r, s, k)
        for p in range(2)
        for r in range(b["repeats"])
        for s in range(8)
        for k in range(8)
    }
    keys = lambda rows: {
        tuple(int(x[k]) for k in ["variant", "trial", "step", "stage"]) for x in rows
    }
    if (
        len(refs) != n * 8
        or len(gates) != n * 8
        or keys(refs) != expected
        or keys(gates) != expected
    ):
        raise ValueError("Incomplete stage vectors")
    expected_updates = {
        (p, r, t, c)
        for p in range(2)
        for r in range(b["repeats"])
        for t in range(8)
        for c in (["independent"] if p == 0 else ["independent", "baseline"])
    }
    if (
        len(updates) != n * 3 // 2
        or {(int(x["variant"]), int(x["trial"]), int(x["step"]), x["comparison"]) for x in updates}
        != expected_updates
    ):
        raise ValueError("Incomplete residual-update checks")
    if any(x["pass"] != "1" or int(x["rel_l2_ppm"]) != 0 for x in refs + gates + updates) or any(
        int(x["bit_mismatches"]) for x in refs
    ):
        raise ValueError("Actual vectors differ from independent or same-profile reference")
    prefix = {"tiny": "decode", "qwen": "qwen_decode", "llama": "llama7b_padded"}[b["model"]]
    expected_ids = {
        f'{prefix}_{b["mix"]}_p{p}_r{r}_s{t}'
        for p in range(2)
        for r in range(b["repeats"])
        for t in range(8)
    }
    if len(regions) != n or {r["case_id"] for r in regions} != expected_ids:
        raise ValueError("Missing/duplicate token ROI")
    for r in regions:
        identity = re.fullmatch(
            re.escape(prefix + "_" + b["mix"]) + r"_p([01])_r([0-9]+)_s([0-7])", r["case_id"]
        )
        variant, trial, step = map(int, identity.groups())
        if (
            r["trial"] != trial
            or type(r["warmup"]) is not bool
            or r["warmup"] != (trial == 0)
            or r["valid_keys"] != 2049 + step
        ):
            raise ValueError("Wrong trial/token identity")
        if (
            r["pass"] is not True
            or r["valid_rows"] != 1
            or r["padded_rows"] != 32
            or not 2049 <= r["valid_keys"] <= 2056
        ):
            raise ValueError("Wrong token contract")
        for k in ["hart", "start_cycle", "end_cycle", "instret_delta", "checksum"]:
            if type(r[k]) is not int or not 0 <= r[k] < 2**64:
                raise ValueError("Invalid unsigned counter")
        if "reference_hout_checksums" in b and r["checksum"] != b["reference_hout_checksums"][step]:
            raise ValueError("Guest Hout checksum differs from independent reference")
    guards = records("LLAMA7B_PADDED_GUARD") if b["model"] == "llama" else []
    if b["model"] == "llama" and (
        len(guards) != n
        or any(
            g["pass"] != "1"
            or g["physical_capacity"] != "2057"
            or any(
                g[k] != "0"
                for k in ["poison_errors", "output_canary_errors", "cache_canary_errors"]
            )
            for g in guards
        )
    ):
        raise ValueError("Llama padding/future/canary check failed")
    if "MX_NATIVE_DECODE_BEGIN model=" + b["model"] not in text or "failed=0" not in text:
        raise ValueError("Missing complete native replay")
    return dict(
        pass_=True,
        regions=n,
        reference_checks=len(refs),
        same_profile_checks=len(gates),
        update_checks=len(updates),
        layout_guards=len(guards),
        bit_mismatches=0,
        maximum_relative_l2_ppm=0,
        timing_qualified=False,
        fpga_run=False,
    )


def replay(a):
    o = a.build_root.resolve()
    b = json.loads((o / "build.json").read_text())
    proof = o / "functional-proof.json"
    if proof.exists():
        raise FileExistsError("Preserve prior execution proof; choose fresh build")
    for p, h in b["assets"].items():
        if sha(p) != h:
            raise ValueError("Changed asset")
    if b.get("source_capsule"):
        cap = Path(b["source_capsule"])
        for rel, digest in b["sources"].items():
            if sha(cap / rel) != digest:
                raise ValueError("Changed compiled source capsule")
        for group in b["source_dependencies"].values():
            if any(sha(p) != h for p, h in group.items()):
                raise ValueError("Changed compiler input")
        if sha(o / "builder.py") != b["builder_snapshot_sha256"]:
            raise ValueError("Changed builder snapshot")
    results = []
    for target in a.targets:
        e = b["elfs"][target]
        elf = Path(e["path"])
        if sha(elf) != e["sha256"]:
            raise ValueError("Changed ELF")
        log = o / (target + ".log")
        if log.exists():
            raise FileExistsError("Preserve prior log")
        if target == "host":
            cmd = [str(elf)]
        else:
            if not a.spike or not a.gemmini_lib:
                raise ValueError("Supply explicit --spike and corrected --gemmini-lib")
            cmd = [
                str(a.spike),
                "--extlib=" + str(a.gemmini_lib),
                "--extension=gemmini_mxbf16_dim16",
                str(elf),
            ]
        runtime = {}
        if target == "spike":
            runtime = {
                str(a.spike.resolve()): sha(a.spike),
                str(a.gemmini_lib.resolve()): sha(a.gemmini_lib),
            }
        else:
            linked = subprocess.check_output(["ldd", str(elf)], text=True)
            runtime = {
                str(Path(p).resolve()): sha(p)
                for p in re.findall(r"(/[^\s()]+)", linked)
                if Path(p).is_file()
            }
        with log.open("x") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        result = inspect(log.read_text(), b)
        if r.returncode:
            raise ValueError("Execution failed")
        if any(sha(p) != h for p, h in runtime.items()):
            raise ValueError("Execution runtime changed")
        result.update(
            runtime_sha256=runtime,
            target=target,
            command=cmd,
            elf_sha256=sha(elf),
            log_sha256=sha(log),
        )
        results.append(result)
    proof.write_text(
        json.dumps(
            dict(
                pass_=True,
                results=results,
                build_sha256=sha(o / "build.json"),
                reference_sha256=sha(next(p for p in b["assets"] if p.endswith("_expect.bin"))),
                fpga_run=False,
                performance_qualified=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print(proof)


class BuilderPreservation(unittest.TestCase):
    def test_qualified_mode6_cli_and_unadmitted_mode7(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            out = d / "kept"
            out.mkdir()
            (out / "sentinel").write_bytes(b"immutable")
            cmd = [
                "python3",
                "-B",
                str(SDK / "gen/build_micromix_decode.py"),
                "--model",
                "qwen",
                "--mix",
                "mix50",
                "--output-root",
                str(out),
            ]
            for flag in [
                "capture-header",
                "capture-bin",
                "weights-header",
                "weights-bin",
                "reference-bin",
                "reference-manifest",
                "rocc-root",
            ]:
                cmd += ["--" + flag, str(d / "missing")]
            a = subprocess.run(
                [*cmd, "--attention-mode", "6"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertNotEqual(a.returncode, 0)
            self.assertIn("Use a fresh --output-root", a.stderr)
            self.assertNotIn("invalid choice", a.stderr)
            self.assertEqual((out / "sentinel").read_bytes(), b"immutable")
            self.assertEqual(len(list(out.iterdir())), 1)
            b = subprocess.run(
                [*cmd, "--attention-mode", "7"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(b.returncode, 2)
            self.assertIn("invalid choice", b.stderr)
            self.assertEqual(len(list(out.iterdir())), 1)

    def test_issue_cli_preservation_and_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            out = d / "kept"
            out.mkdir()
            (out / "sentinel").write_bytes(b"keep")
            cmd = [
                "python3",
                "-B",
                str(SDK / "gen/build_micromix_decode.py"),
                "--model",
                "qwen",
                "--mix",
                "mix625",
                "--output-root",
                str(out),
            ]
            for flag in [
                "capture-header",
                "capture-bin",
                "weights-header",
                "weights-bin",
                "reference-bin",
                "reference-manifest",
                "rocc-root",
            ]:
                cmd += ["--" + flag, str(d / "missing")]
            for value in [0, 1]:
                p = subprocess.run(
                    [*cmd, "--decode-issue-mode", str(value)], capture_output=True, text=True
                )
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("Use a fresh --output-root", p.stderr)
                self.assertEqual((out / "sentinel").read_bytes(), b"keep")
                self.assertEqual(len(list(out.iterdir())), 1)
            p = subprocess.run([*cmd, "--decode-issue-mode", "2"], capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn("invalid choice", p.stderr)
        spec = importlib.util.spec_from_file_location(
            "issuebuilder", SDK / "gen/build_micromix_decode.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as temp:
            for model, issue in [("tiny", 1), ("llama", 1), ("qwen", True), ("qwen", 2)]:

                class A:
                    pass

                a = A()
                a.output_root = Path(temp) / "new"
                a.model = model
                a.decode_issue_mode = issue
                with self.assertRaises(ValueError):
                    m.build(a)
                self.assertFalse(a.output_root.exists())

    def test_bad_reference_rejected_before_output(self):
        spec = importlib.util.spec_from_file_location(
            "builder", SDK / "gen/build_micromix_decode.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            names = [
                "llama_decode_capture.h",
                "llama_decode_capture.bin",
                "llama_micromix_mix50.h",
                "llama_micromix_mix50.bin",
                "llama_decode_expect.bin",
            ]
            for n in names:
                (d / n).write_bytes(b"input")
            meta = d / "reference.json"
            meta.write_text(
                json.dumps(dict(reference_version="wrong", sha256=sha(d / names[-1]), steps=8))
            )

            class A:
                pass

            a = A()
            a.output_root = d / "new"
            a.model = "tiny"
            a.mix = "mix50"
            a.capture_header = d / names[0]
            a.capture_bin = d / names[1]
            a.weights_header = d / names[2]
            a.weights_bin = d / names[3]
            a.reference_bin = d / names[4]
            a.reference_manifest = meta
            with self.assertRaises(ValueError):
                m.build(a)
            self.assertFalse(a.output_root.exists())

    def test_nonfinite_and_wrong_length_references_refuse_before_writes_or_tools(self):
        spec = importlib.util.spec_from_file_location(
            "builder", SDK / "gen/build_micromix_decode.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for malformed in ["nonfinite", "short"]:
            with self.subTest(kind=malformed), tempfile.TemporaryDirectory() as temp:
                d = Path(temp)
                names = [
                    "llama_decode_capture.h",
                    "llama_decode_capture.bin",
                    "llama_micromix_mix50.h",
                    "llama_micromix_mix50.bin",
                    "llama_decode_expect.bin",
                ]
                for n in names:
                    (d / n).write_bytes(b"input")
                widths = [2048, 256, 256, 2048, 5632, 5632, 2048, 2048]
                raw = bytearray(8 * sum(widths) * 2) if malformed == "nonfinite" else bytearray(2)
                if malformed == "nonfinite":
                    raw[0:2] = b"\x80\x7f"
                (d / names[-1]).write_bytes(raw)
                meta = d / "reference.json"
                meta.write_text(
                    json.dumps(
                        dict(
                            reference_version="products_v2",
                            sha256=sha(d / names[-1]),
                            steps=8,
                            mix="mix50",
                            batch=1,
                            layer=5,
                            widths=widths,
                        )
                    )
                )

                class A:
                    pass

                a = A()
                a.output_root = d / "new"
                a.model = "tiny"
                a.mix = "mix50"
                a.capture_header = d / names[0]
                a.capture_bin = d / names[1]
                a.weights_header = d / names[2]
                a.weights_bin = d / names[3]
                a.reference_bin = d / names[4]
                a.reference_manifest = meta
                with patch.object(m.subprocess, "run") as run, patch.object(
                    m.subprocess, "check_output"
                ) as cmd:
                    with self.assertRaises(ValueError):
                        m.build(a)
                    run.assert_not_called()
                    cmd.assert_not_called()
                self.assertFalse(a.output_root.exists())

    def test_cached_object_data_hash_is_verified(self):
        spec = importlib.util.spec_from_file_location(
            "builder", SDK / "gen/build_micromix_decode.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "tiny.bin").write_bytes(bytes(range(64)))
            subprocess.run(
                [
                    "objcopy",
                    "-I",
                    "binary",
                    "-O",
                    "elf64-x86-64",
                    "--set-section-alignment",
                    ".data=64",
                    "tiny.bin",
                    "tiny.o",
                ],
                cwd=d,
                check=True,
            )
            self.assertEqual(m.data_sha(d / "tiny.o"), sha(d / "tiny.bin"))
            (d / "tiny.bin").write_bytes(bytes(64))
            self.assertNotEqual(m.data_sha(d / "tiny.o"), sha(d / "tiny.bin"))

    def test_existing_output_rejected_before_asset_reads_or_tools(self):
        spec = importlib.util.spec_from_file_location(
            "builder", SDK / "gen/build_micromix_decode.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sentinel"
            out.mkdir()
            (out / "immutable.elf").write_bytes(b"preserve")

            class A:
                output_root = out

            with self.assertRaises(FileExistsError):
                m.build(A())
            self.assertEqual((out / "immutable.elf").read_bytes(), b"preserve")
            self.assertEqual(len(list(out.iterdir())), 1)


class ReferenceChecksums(unittest.TestCase):
    def test_token_and_stage_boundaries(self):
        spec = importlib.util.spec_from_file_location(
            "builder", SDK / "gen/build_micromix_decode.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        import struct

        widths = [1, 2, 3, 4, 5, 6, 7, 2]
        words = []
        for step in range(8):
            words.extend([step + 1] * sum(widths[:-1]) + [0x3F80, 0xBF80])
        raw = struct.pack("<" + "H" * len(words), *words)
        values = m.reference_hout_checksums(raw, widths)
        self.assertEqual(len(set(values)), 1)
        changed = bytearray(raw)
        changed[0:2] = b"\x00\x00"
        self.assertEqual(m.reference_hout_checksums(changed, widths), values)
        changed = bytearray(raw)
        offset = (3 * sum(widths) + sum(widths[:-1])) * 2
        changed[offset : offset + 2] = b"\x00\x00"
        actual = m.reference_hout_checksums(changed, widths)
        self.assertEqual([i for i, (a, b) in enumerate(zip(actual, values)) if a != b], [3])

    def test_bad_reference_geometry(self):
        spec = importlib.util.spec_from_file_location(
            "builder", SDK / "gen/build_micromix_decode.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for raw, widths in [
            (bytes(128), [1] * 7),
            (bytes(128), [1] * 7 + [True]),
            (bytes(126), [1] * 8),
        ]:
            with self.assertRaises(ValueError):
                m.reference_hout_checksums(raw, widths)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--build-root", type=Path)
    p.add_argument("--targets", nargs="+", choices=["host", "spike"], default=["host", "spike"])
    p.add_argument("--spike", type=Path)
    p.add_argument("--gemmini-lib", type=Path)
    a = p.parse_args()
    if a.build_root:
        replay(a)
    else:
        unittest.main(argv=["test_micromix_decode"])


if __name__ == "__main__":
    main()
