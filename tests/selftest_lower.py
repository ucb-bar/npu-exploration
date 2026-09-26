"""compiler/lower.py builds the command buffer merlin's parser used to build, without merlin.

Claims:
1. ``command_buffer(stages, bundles)`` equals the recorded parser output for every chain shape and
   format in ``tests/oracle/command_buffers.json`` (captured once by ``make_command_buffers.py``),
   with and without operand bundles attached.
2. ``lower()`` picks the lowering the pipeline picks: a chain is fused, a graph of emittable host
   stages is one graph, a ``fn=`` host stage or ``per_stage=True`` is per-stage; edges say how each
   intermediate reached the mesh.
3. Nothing in the compiler imports merlin.

    .venv/bin/python tests/selftest_lower.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def main() -> int:
    from compiler.lower import MatmulStage, command_buffer, lower, wire_paths
    wire_paths()

    print("[1] command_buffer == merlin's parser output (recorded) ----------")
    rec = json.loads((REPO / "tests" / "oracle" / "command_buffers.json").read_text())
    for case in rec["cases"]:
        stages = [MatmulStage(**s) for s in case["stages"]]
        got = command_buffer(stages, None, operand_fmt=case["operand_fmt"])
        check(case["name"], got == case["cb"],
              "" if got == case["cb"] else json.dumps(got)[:200])
        got2 = command_buffer(stages, case["bundles"], operand_fmt=case["operand_fmt"])
        check(case["name"] + " + operands", got2 == case["cb_with_operands"])
    check("unknown format refused",
          _raises(lambda: command_buffer([MatmulStage(64, 64, 64, "W", "Y0", "X")], None,
                                         operand_fmt="int8")))

    print("\n[2] lower() picks the lowering the pipeline picks ------------------")
    from kernels.registry import build
    from kernels.spec import HostStage, KernelSpec, Stage
    import torch
    lin = lower(build("linear"))
    check("linear -> fused, one stage, requant edge",
          lin.kind == "fused" and len(lin.stages) == 1 and lin.edges["L0"]["via"] == "requant"
          and lin.cb["commands"][1]["opcode"] == "MATMUL_RESIDENT")
    m3 = lower(build("mlp3"))
    check("mlp3 -> fused, 3 stages, every edge requant, cb has 3 operand bundles",
          m3.kind == "fused" and len(m3.stages) == 3
          and all(e["via"] == "requant" for e in m3.edges.values())
          and len(m3.cb["mx_operands"]) == 3)
    att = lower(build("attention"))
    check("attention -> graph, host edges, operand_fmt recorded",
          att.kind == "graph" and all(e["via"] == "host" for e in att.edges.values())
          and att.cb["commands"] == [] and att.cb["graph"]["operand_fmt"] == "fp8_e4m3")
    ps = lower(build("attention"), per_stage=True)
    check("per_stage=True -> per_stage, no cb, records carry shapes",
          ps.kind == "per_stage" and ps.cb is None
          and all(("m" in r) == (r["where"] == "mesh") for r in ps.stages))
    torch.manual_seed(0)
    fn_spec = KernelSpec("fnhost", torch.randn(64, 64), [
        Stage("L0", weight=torch.randn(64, 64)),
        HostStage("H", fn=lambda t: t * 2.0, src="L0"),
        Stage("L1", weight=torch.randn(64, 64), lhs="H")])
    check("a fn= host stage -> per_stage", lower(fn_spec).kind == "per_stage")
    # mxformats.chain_refusal refuses nothing today (a live invariant, not an oversight), so the
    # accepted-lossy-chain warning has no case to fire on; the hook must at least be inert.
    warned = []
    m2 = lower(build("mlp2"), "fp6_e3m2", warn=warned.append)
    check("codebook chain lowers (LUT books travel on every bundle), no warning today",
          m2.kind == "fused" and all("c_lut" in b for b in m2.cb["mx_operands"]) and warned == [])

    print("\n[3] the compiler does not import merlin ------------------------------")
    text = (REPO / "compiler" / "lower.py").read_text()
    check("compiler/lower.py has no merlin import",
          "import merlin" not in text and "from merlin" not in text)
    check("merlin not loaded in this process", not any(m.startswith("merlin") for m in sys.modules))

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nALL CHECKS PASSED")
    return 1 if FAILURES else 0


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
