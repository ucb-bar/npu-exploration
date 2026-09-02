"""Self-test for the grade/ chain WITHOUT any hardware or toolchain.

Exercises exactly the path a real run takes after the simulator returns —
Telemetry buffering -> attach -> log.jsonl, metrics.compare verdict logic, and
report.write_report's results/ layout. Needs only torch + numpy, so it proves
the telemetry and reporting plumbing works before spike is ever involved.

    .venv/bin/python tests/selftest_grade.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from grade.metrics import accuracy_metrics, bit_exact_diff, compare  # noqa: E402
from grade.report import make_run_id, write_report  # noqa: E402
from grade.telemetry import Telemetry  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="grade_selftest_"))
    try:
        print("\n[1] metrics -----------------------------------------------------")
        ref = torch.randn(16, 16)
        exact = ref.clone()
        noisy = ref + 0.01 * torch.randn(16, 16)

        m_exact = accuracy_metrics(exact, ref)
        check("identical tensors give zero error", m_exact["rel_fro"] == 0.0)
        m_noisy = accuracy_metrics(noisy, ref)
        check("noisy tensor gives small non-zero error",
              0.0 < m_noisy["rel_fro"] < 0.2, f"rel_fro={m_noisy['rel_fro']:.4%}")
        check("bit_exact_diff detects equality", bit_exact_diff(exact, ref)["bit_exact"])
        check("bit_exact_diff detects difference", not bit_exact_diff(noisy, ref)["bit_exact"])

        print("\n[2] verdict logic -----------------------------------------------")
        g_pass = compare(noisy, ref, None, tol_rel_fro=0.15)
        check("fp32 tier passes inside tolerance", g_pass["pass"] is True)
        check("tier is reported as fp32", g_pass["tier"] == "fp32")
        check("golden slot is empty in phase 1", g_pass["correctness_vs_golden_model"] is None)

        g_fail = compare(noisy, ref, None, tol_rel_fro=0.0001)
        check("fp32 tier fails outside tolerance", g_fail["pass"] is False)

        nan = ref.clone()
        nan[0, 0] = float("nan")
        g_nan = compare(nan, ref, None, tol_rel_fro=1.0)
        check("non-finite output fails regardless of tolerance", g_nan["pass"] is False,
              f"finite {g_nan['finite']['n_finite']}/{g_nan['finite']['total']}")

        g_gold = compare(exact, ref, exact, tol_rel_fro=0.15)
        check("golden tier reports bit-exact pass", g_gold["pass"] is True)
        check("tier switches to golden when supplied", g_gold["tier"] == "golden")

        print("\n[3] telemetry buffering -----------------------------------------")
        tel = Telemetry()
        tel.log("setup", "logged BEFORE attach -- must still reach the file")
        tel.log("model", "second buffered event", m=16, k=16, n=16)
        # Reaches into Telemetry._records on purpose: buffering before attach() is
        # the behaviour under test, and it has no public accessor.
        check("events buffer before attach", len(tel._records) == 2)

        print("\n[4] report + log.jsonl ------------------------------------------")
        run_id = make_run_id("selftest", "16x16x16")
        run_dir = write_report(
            results_dir=tmp, run_id=run_id,
            run_config={"kernel": "selftest", "m": 16, "k": 16, "n": 16},
            provenance={"simulator": "none", "note": "self-test, no hardware"},
            hardware_output=noisy, fp32_reference=ref, golden_model_output=None,
            metrics=g_pass, artifacts={"elf": "n/a"}, telemetry=tel)

        for fn in ("config.json", "metrics.json", "log.jsonl",
                   "hardware_output.npy", "fp32_reference.npy"):
            check(f"{fn} written", (run_dir / fn).exists())
        check("golden npy absent in phase 1", not (run_dir / "golden_model.npy").exists())

        lines = [json.loads(x) for x in
                 (run_dir / "log.jsonl").read_text().strip().splitlines()]
        stages = [r["stage"] for r in lines]
        check("buffered events were flushed to log.jsonl",
              "setup" in stages and "model" in stages, f"stages={stages}")
        check("post-attach event also landed", "report" in stages)
        check("structured fields survive round-trip",
              any(r.get("k") == 16 for r in lines))
        check("every record has ts + stage + message",
              all({"ts", "stage", "message"} <= set(r) for r in lines))

        cfg = json.loads((run_dir / "config.json").read_text())
        check("config.json snapshots run_config", cfg["run_config"]["m"] == 16)
        check("config.json snapshots provenance", "simulator" in cfg["provenance"])
        met = json.loads((run_dir / "metrics.json").read_text())
        check("metrics.json round-trips the verdict", met["pass"] is True)

        print("\n" + "=" * 66)
        if FAILURES:
            print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
            return 1
        print("ALL CHECKS PASSED -- telemetry, metrics and reporting are wired correctly.")
        print("(This proves the grade/ chain only; spike is exercised by run_matmul.py.)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
