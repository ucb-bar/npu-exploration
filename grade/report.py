"""Writes results/<run_id>/. Separate from pipeline.py because it is about
persistence format, not decision-making.

A results folder must stay interpretable months later, so it snapshots the
resolved run configuration (shapes, seed, quantization settings, geometry the
backend actually planned with, toolchain provenance) alongside the arrays —
not just the metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .telemetry import Telemetry


def make_run_id(kernel: str, shape: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{kernel}_{shape}"


def write_report(
    *,
    results_dir: Path,
    run_id: str,
    run_config: dict,
    provenance: dict,
    hardware_output: torch.Tensor,
    fp32_reference: torch.Tensor,
    golden_model_output: torch.Tensor | None = None,
    metrics: dict,
    artifacts: dict | None = None,
    telemetry: Telemetry | None = None,
) -> Path:
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_config": run_config,
            "provenance": provenance,
            "artifacts": artifacts or {},
        }, f, indent=2)

    np.save(run_dir / "hardware_output.npy", hardware_output.detach().cpu().numpy())
    np.save(run_dir / "fp32_reference.npy", fp32_reference.detach().cpu().numpy())
    if golden_model_output is not None:
        np.save(run_dir / "golden_model.npy", golden_model_output.detach().cpu().numpy())

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if telemetry is not None:
        telemetry.attach(run_dir / "log.jsonl")
        verdict = "PASS" if metrics["pass"] else "FAIL"
        telemetry.log("report", f"{verdict} (tier={metrics['tier']}) -- saved to {run_dir}")

    return run_dir
