#!/usr/bin/env python3
"""Verify a completed native Spike full-layer replay, never promote FPGA timing."""
import argparse
import hashlib
import json
from pathlib import Path

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(build, log, library):
    b = json.loads(Path(build).read_text())
    text = Path(log).read_text()
    elf = Path(build).parent / "prefill.elf"
    if b["target"] != "spike" or b.get("timing_qualified") is not False:
        raise ValueError("Only explicit Spike functional builds")
    if sha(elf) != b["elf_sha256"]:
        raise ValueError("ELF changed")
    for key in ("source_dependencies", "data", "objects"):
        if any(sha(p) != h for p, h in b[key].items()):
            raise ValueError("Build dependency changed: " + key)

    def records(prefix):
        return [json.loads(l[len(prefix) :]) for l in text.splitlines() if l.startswith(prefix)]

    gates = records("MXMM_NUMERIC_GATE ")
    regions = records("IRIS_REGION ")
    trials = records("MXMM_TRIAL_RESULT ")
    if not regions or len(gates) != 18 * len(regions) or len(trials) != len(regions):
        raise ValueError("Incomplete full-vector replay")
    stages = {"q", "k", "v", "o", "gate", "up", "down", "hout", "update"}
    for region, trial in zip(regions, trials):
        case = region["case_id"]
        current = [g for g in gates if g["case_id"] == case]
        if (
            region["pass"] is not True
            or trial["pass"] is not True
            or trial["source"] != "spike"
            or trial["case_id"] != case
        ):
            raise ValueError("Failed/non-Spike trial")
        if {(g["comparison"], g["stage"]) for g in current} != {
            (c, s) for c in ("spike", "hardware_baseline") for s in stages
        }:
            raise ValueError("Wrong gate roster")
        if len(current) != 18 or any(
            g["pass"] is not True
            or g["finite"] is not True
            or g["relative_l2_ppm"] != 0
            or g["bit_mismatches"] != 0
            for g in current
        ):
            raise ValueError("Port requires exact complete output/reference agreement")
    if text.count("MICROMIX_PREFILL_DONE ") != 1:
        raise ValueError("No completed layer sequence")
    return dict(
        pass_=True,
        model=b["model"],
        mix=b["mix"],
        full_layer=True,
        independent_and_same_mix_exact=True,
        numeric_gates=len(gates),
        regions=len(regions),
        elf_sha256=sha(elf),
        build_sha256=sha(build),
        log_sha256=sha(log),
        library_sha256=sha(library),
        timing_usable=False,
        hardware_actions=0,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = verify(a.build, a.log, a.library)
    result["pass"] = result.pop("pass_")
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
