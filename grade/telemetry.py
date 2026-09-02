"""Shared step-logger used by every stage (recipes, kernels, codegen,
backends, compare) so run output is consistent instead of each module
inventing its own print format.

Each event both prints a human-readable line to stdout and appends a
structured record to the run's log.jsonl (written by compare/report.py once
a run_id/output directory exists). Before a run_id exists (e.g. while just
validating a recipe on its own), events are still printed; they're simply
not persisted to a log file until attach() is called.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


class Telemetry:
    def __init__(self) -> None:
        self._log_path: Path | None = None
        self._records: list[dict] = []

    def attach(self, log_path: Path) -> None:
        """Point future (and previously buffered) events at a log.jsonl file."""
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._records:
            with open(self._log_path, "a") as f:
                for record in self._records:
                    f.write(json.dumps(record) + "\n")
            self._records.clear()

    def log(self, stage: str, message: str, **fields) -> None:
        record = {
            "ts": time.time(),
            "stage": stage,
            "message": message,
            **fields,
        }
        print(f"[{stage:8s}] {message}", file=sys.stderr)
        if self._log_path is not None:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        else:
            self._records.append(record)
