#!/usr/bin/env python3
"""Render explicit mixed-only caller profiles from reviewed qualification rows.
This is a data renderer, not a replacement for numerical/terminal qualification.
"""
import argparse
import json
from pathlib import Path

MODELS = {"tinyllama": 0, "qwen25_7b": 1, "llama2_7b": 2}
# Advanced selectors are admitted only for independently terminal-qualified rows.
ADVANCED_EVIDENCE = {
    (
        "tinyllama",
        "decode",
        "mix50",
        4,
    ): "80a71de67f3cd82bff0925314141756774de68eb1e8dcf357b72658a23691a81",
    (
        "qwen25_7b",
        "decode",
        "mix50",
        5,
    ): "9bc4c1b8d1a90af3a826ea04d8068a94ac5bed28bf996af8c11a5e86623cbde8",
    (
        "llama2_7b",
        "decode",
        "mix50",
        6,
    ): "fc3646b73bab7aa16c963bc7db2ccf4791ee8dfa3883c5eb50761f714a299f96",
}


def render(data):
    rows = data["rows"]
    expected = {
        (m, r, x) for m in MODELS for r in ("prefill", "decode") for x in ("mix50", "mix625")
    }
    if data.get("schema_version") != 1 or data.get("image") != "dim16_bf16_shared_mx":
        raise ValueError("Unsupported image/profile schema")
    if len(rows) != len(expected):
        raise ValueError("Complete reviewed model/regime/mix roster required")
    found = set()
    table = {}
    for row in rows:
        key = tuple(row[k] for k in ("model", "regime", "mix"))
        if key not in expected or key in found:
            raise ValueError("Duplicate/unsupported row")
        if (
            row.get("source") != "fpga"
            or row.get("qualified") is not True
            or not row.get("evidence")
        ):
            raise ValueError("Measured qualification evidence required")
        evidence = row["evidence"]
        if not isinstance(evidence, dict) or any(
            not isinstance(evidence.get(k), str)
            or len(evidence[k]) != 64
            or any(c not in "0123456789abcdef" for c in evidence[k])
            for k in (
                "evidence_sha256",
                "elf_sha256",
                "image_sha256",
                "uart_sha256",
                "archive_sha256",
            )
        ):
            raise ValueError("Complete measured evidence SHA anchors required")
        if (
            type(evidence.get("median_cycles")) is not int
            or evidence["median_cycles"] <= 0
            or not isinstance(evidence.get("run"), str)
        ):
            raise ValueError("Measured run and positive cycles required")
        for k in (
            "nc",
            "run",
            "fence_groups",
            "integer_pack",
            "prepared_reuse",
            "attention_mode",
            "cache_lut",
            "physical_padding",
        ):
            if type(row.get(k)) is not int:
                raise ValueError("Integer caller settings required")
        if (
            row["integer_pack"] != 1
            or row["prepared_reuse"] not in (0, 1)
            or row["fence_groups"] not in (0, 1)
            or row["attention_mode"] not in (0, 1, 2, 3, 4, 5, 6)
            or row["cache_lut"] not in (0, 1)
            or row["physical_padding"] not in (0, 1)
        ):
            raise ValueError("Unsupported flags")
        if row["nc"] not in (128, 256) or row["run"] not in (4, 8):
            raise ValueError("Unsupported measured schedule")
        if (
            row["attention_mode"] >= 4
            and ADVANCED_EVIDENCE.get((*key, row["attention_mode"])) != evidence["evidence_sha256"]
        ):
            raise ValueError("Unqualified advanced attention selector")
        if type(row.get("cache_time_tile", 0)) is not int or row.get("cache_time_tile", 0) not in (
            0,
            16,
        ):
            raise ValueError("Unsupported cache tile")
        if row.get("cache_time_tile", 0) and not (
            key[0] == "llama2_7b"
            and key[1] == "decode"
            and row["attention_mode"] == 6
            and row["cache_lut"] == 1
            and row["physical_padding"] == 1
        ):
            raise ValueError("Cache tile requires explicit padded MHA geometry")
        if (row["attention_mode"] == 6) != (row.get("cache_time_tile", 0) == 16):
            raise ValueError("Measured MHA tiled profile requires paired cache tile16")
        if type(row.get("decode_issue_mode", 0)) is not int or row.get("decode_issue_mode", 0) != 0:
            raise ValueError("Issue1 has no qualified profile; use explicit opt-in")
        if row["regime"] == "decode" and row["prepared_reuse"]:
            raise ValueError("Prepared reuse is full-row prefill only")
        if row["regime"] == "prefill" and row["attention_mode"]:
            raise ValueError("Scalar decode mode cannot select VPU prefill")
        found.add(key)
        table[key] = row
    lines = [
        "/* Explicit reviewed mixed-format profile; defaults remain unchanged. */",
        "#ifndef MX_MICROMIX_PROFILES_H",
        "#define MX_MICROMIX_PROFILES_H",
        '#include "mx_llm_geometry.h"',
        '#include "mx_micromix_prepared.h"',
        "typedef enum {MXMM_PROFILE_MIX50=1,MXMM_PROFILE_MIX625=2} mxmm_profile_mix;",
        "typedef enum {MXMM_PROFILE_PREFILL=0,MXMM_PROFILE_DECODE=1} mxmm_profile_regime;",
        "typedef struct {mxmm_schedule schedule;unsigned char integer_pack,prepared_reuse,attention_mode,cache_lut,physical_padding,cache_time_tile,decode_issue_mode;} mxmm_profile;",
        "/* attention_mode:0 scalar,1 GQA8 FMA,2 GQA7 FMA,3 MHA4 FMA,4 GQA8 tiled FMA,5 GQA7 tiled FMA,6 MHA QKT8/PVH1J16 FMA.",
        "# * Values are bytes so prefill never implicitly includes strict-FP helpers. */".replace(
            "# *", " *"
        ),
        "static const mxmm_profile mxmm_profiles[3][2][2]={",
    ]
    for m in MODELS:
        lines.append(" { /* " + m + " */")
        for regime in ("prefill", "decode"):
            entries = []
            for mix in ("mix50", "mix625"):
                r = table[m, regime, mix]
                entries.append(
                    "{{%d,%d,%d},%d,%d,%d,%d,%d,%d,%d}"
                    % (
                        tuple(
                            r[k]
                            for k in (
                                "nc",
                                "run",
                                "fence_groups",
                                "integer_pack",
                                "prepared_reuse",
                                "attention_mode",
                                "cache_lut",
                                "physical_padding",
                            )
                        )
                        + (r.get("cache_time_tile", 0), r.get("decode_issue_mode", 0))
                    )
                )
            lines.append("  {" + ",".join(entries) + "}, /* " + regime + " */")
        lines.append(" },")
    lines += [
        "};",
        "static inline const mxmm_profile *mxmm_profile_get(mx_llm_model model,mxmm_profile_regime regime,mxmm_profile_mix mix) {",
        " if((unsigned)model>=3 || (unsigned)regime>=2 || (mix!=MXMM_PROFILE_MIX50 && mix!=MXMM_PROFILE_MIX625))return NULL;",
        " return &mxmm_profiles[model][regime][mix-1];",
        "}",
        "/* Apply in the SAME translation unit as projection calls. An outstanding lease",
        " * prevents all state changes. Caller separately applies returned attention/cache",
        "# * settings and explicitly chooses prepared vs canonical entrypoints. */".replace(
            "# *", " *"
        ),
        "static inline int mxmm_profile_apply(mx_llm_model model,mxmm_profile_regime regime,mxmm_profile_mix mix,mxmm_schedule *schedule) {",
        " const mxmm_profile *p=mxmm_profile_get(model,regime,mix);",
        " if(!p || !schedule || mxmp_live)return 1;",
        " *schedule=p->schedule;mxmm_fast_pack=1;mxmm_integer_pack=p->integer_pack;return 0;",
        "}",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path(__file__).with_name("micromix_profiles.json"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    text = render(json.loads(a.input.read_text()))
    a.output.write_text(text)


if __name__ == "__main__":
    main()
