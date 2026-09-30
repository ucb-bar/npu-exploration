"""A copy of the workspace with a SYNTHETIC SRAM table, for plumbing checks only.

The real tables (tech/sram_qrt/qrt_table.csv) are PDK data kept out of the workspace repo. This fixture invents
one macro per SRAM family so memory_model.py can run end to end: it checks that the adapters call it, parse it and
keep it out of the existing totals. None of its numbers mean anything.
"""
from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path

FAMILIES = ("tsn16ffclluhd2prf", "tsn16ffcllhdspsbsram", "tsn16ffcllshdspmbsram", "tsn16ffcll1prf", "tsn16ffclldpsram")
COLUMNS = ("family", "type", "corner", "vt", "vdd", "word", "io", "mux", "area_um2", "taa_ns", "tcyc_ns",
           "readc_uA_MHz", "writec_uA_MHz", "leakage_uA")


def synthetic_workspace(root: Path) -> Path:
    """Copy the workspace's ppa/ into a temporary folder and add an invented SRAM table. Caller removes it."""
    dst = Path(tempfile.mkdtemp(prefix="ppa_memfixture_")) / "ppa"
    shutil.copytree(root, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "plots", "report"))
    table = dst / "tech" / "sram_qrt" / "qrt_table.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for fam in FAMILIES:
            w.writerow((fam, "SYNTHETIC", "tt0p8v25c", "svt", 0.8, 8192, 256, 4, 100000.0, 0.5, 0.8, 10.0, 12.0, 5.0))
    return dst
