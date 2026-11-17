#!/usr/bin/env python3
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

K = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(K / "gen"))
from gen_micromix_weights import export, allocation
from gen_micromix_profiles import render
from micromix_formats import grid, quant


class Generators(unittest.TestCase):
    def test_rne_grid_independent(self):
        for f in (4, 6, 8):
            levels = grid(f)
            for i in range(len(levels) - 1):
                # Equal-distance midpoint chooses the even magnitude code.
                midpoint = np.float32((float(levels[i]) + float(levels[i + 1])) / 2)
                block = np.full((1, 32, 1), midpoint, np.float32)
                block[0, 0, 0] = 1
                codes, sc = quant(block, f)
                normalized = np.ldexp(midpoint, 127 - int(sc[0, 0]))
                dist = np.abs(levels.astype(float) - float(normalized))
                choices = np.flatnonzero(dist == dist.min())
                even = choices[(choices & 1) == 0]
                self.assertEqual(int(codes[0, 1, 0]), int(even[0] if len(even) else choices[0]))

    def test_export_layout_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            raw = d / "input.npz"
            weights = {
                f"W_{n}": np.fromfunction(
                    lambda k, n: ((k * 7 + n * 3) % 31 - 15) / 16, (256, 32), dtype=int
                ).astype(np.float32)
                for n in ("q", "k", "v", "o", "g", "u", "d")
            }
            np.savez(raw, **weights)
            for mix in ("fp8", "mix50", "mix625"):
                prefix = d / mix
                m = export(raw, prefix, mix, 32)
                blob = prefix.with_suffix(".bin").read_bytes()
                self.assertEqual(m["bytes"], len(blob))
                self.assertEqual(len(m["projections"]), 7)
                alternate = d / (mix + "_tiled")
                export(raw, alternate, mix, 32, column_tile=2)
                self.assertEqual(blob, alternate.with_suffix(".bin").read_bytes())
                for row in m["projections"]:
                    self.assertEqual(tuple(row["k32_groups"]), allocation(256, mix))
                    self.assertEqual(row["scale_offset"] % 64, 0)
                    for ptr in row["group_offsets"]:
                        if ptr is not None:
                            self.assertEqual(ptr % 64, 0)
                before = {p.name: p.read_bytes() for p in d.glob(mix + ".*")}
                with self.assertRaises(FileExistsError):
                    export(raw, prefix, mix, 32)
                self.assertEqual(before, {p.name: p.read_bytes() for p in d.glob(mix + ".*")})
            with self.assertRaises(ValueError):
                allocation(11040, "mix625")

    def test_profiles_exact_generated(self):
        data = json.loads((K / "gen/micromix_profiles.json").read_text())
        self.assertEqual(render(data), (K / "include/mx_micromix_profiles.h").read_text())
        for mutate in ("incomplete", "fp8", "unqualified", "host", "duplicate", "decode_reuse"):
            bad = copy.deepcopy(data)
            if mutate == "incomplete":
                bad["rows"].pop()
            if mutate == "fp8":
                bad["rows"][0]["mix"] = "fp8"
            if mutate == "unqualified":
                bad["rows"][0]["qualified"] = False
            if mutate == "host":
                bad["rows"][0]["source"] = "spike"
            if mutate == "duplicate":
                bad["rows"][1] = bad["rows"][0]
            if mutate == "decode_reuse":
                next(r for r in bad["rows"] if r["regime"] == "decode")["prepared_reuse"] = 1
            with self.assertRaises(ValueError, msg=mutate):
                render(bad)
        for model, regime, mix in [
            ("tinyllama", "decode", "mix625"),
            ("qwen25_7b", "decode", "mix50"),
            ("llama2_7b", "decode", "mix50"),
        ]:
            bad = copy.deepcopy(data)
            next(
                r
                for r in bad["rows"]
                if (r["model"], r["regime"], r["mix"]) == (model, regime, mix)
            )["attention_mode"] = 4
            with self.assertRaises(ValueError):
                render(bad)
        bad = copy.deepcopy(data)
        next(r for r in bad["rows"] if r["attention_mode"] == 4)["evidence"]["evidence_sha256"] = (
            "0" * 64
        )
        with self.assertRaises(ValueError):
            render(bad)


if __name__ == "__main__":
    unittest.main()
