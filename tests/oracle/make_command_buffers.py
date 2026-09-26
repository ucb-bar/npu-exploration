"""Capture what merlin's interface parser produced for every chain shape the tests use.

Run ONCE, with the merlin submodule present, to write ``command_buffers.json``. ``compiler/lower.py``
builds the same dict directly; ``tests/selftest_lower.py`` holds it to this record, so the compiler
never needs merlin and can never drift from what the emitter was proven on.

    .venv/bin/python tests/oracle/make_command_buffers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "app", REPO / "merlin" / "merlin" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mxiface                                   # noqa: E402  (merlin's grammar + parser)
from app import mxformats                        # noqa: E402

OUT = Path(__file__).with_name("command_buffers.json")


def chain(shapes, fmt):
    et = mxformats.get(fmt, proven_only=False)
    inter = et.mlir or et.name
    ms = []
    for i, (m, k, n) in enumerate(shapes):
        last = i == len(shapes) - 1
        ms.append(mxiface.MatmulStage(m=m, k=k, n=n, weight=f"W{i}", out="Y0" if last else f"T{i}",
                                      lhs="X" if i == 0 else f"T{i - 1}",
                                      out_dtype="bf16" if last else inter))
    return ms


def main() -> int:
    cases = []
    for fmt in mxformats.FORMATS:
        for name, shapes in (("single", [(64, 64, 64)]), ("chain2", [(64, 64, 64)] * 2),
                             ("chain3", [(64, 64, 64)] * 3),
                             ("non-square", [(32, 128, 64), (32, 64, 96)])):
            cases.append((f"{name}/{fmt}", chain(shapes, fmt), fmt))
    # the per-stage lowering's single matmul, with its own names (grade/pipeline.py per-stage path)
    for out_dtype in ("bf16", "f8E4M3FN"):
        cases.append((f"per-stage/{out_dtype}",
                      [mxiface.MatmulStage(m=64, k=64, n=64, weight="W1", out="T1", lhs="A1",
                                           out_dtype=out_dtype)], "fp8_e4m3"))
    rec = []
    for name, ms, fmt in cases:
        text = mxiface.chain_interface_mlir(ms, operand_fmt=fmt)
        bundles = [{"b_codes": [[i, 2 * i]], "b_scales": [[7]], "a_lut": None}
                   | ({"a_codes": [[1]], "a_scales": [[3]]} if i == 0 else {})
                   for i in range(len(ms))]
        rec.append({"name": name, "operand_fmt": fmt,
                    "stages": [st.__dict__ for st in ms],
                    "cb": mxiface.to_command_buffer(text),
                    "cb_with_operands": mxiface.to_command_buffer(text, bundles),
                    "bundles": bundles})
    OUT.write_text(json.dumps({"source": "merlin.targetgen.contract.interface_emit.parse_interface_mlir "
                                          "on app.mxiface.chain_interface_mlir",
                               "cases": rec}, indent=1) + "\n")
    print(f"{len(rec)} cases -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
