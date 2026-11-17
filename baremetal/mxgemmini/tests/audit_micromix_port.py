#!/usr/bin/env python3
"""Compare imported hot function bodies with caller-supplied qualified sources.
Only comments/whitespace are ignored; known generic log-label prefixes are
normalized explicitly. Assembly strings, C numeric tokens/operators and
loop structure are retained. This audit is additional to actual numerical tests.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

K = Path(__file__).resolve().parent.parent
TOKEN = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\s+|[A-Za-z_]\w*|\d+(?:\.\d*)?(?:[eEpP][+-]?\d+)?[uUlLfF]*|.',
    re.S,
)


def tokens(text):
    result = []
    for m in TOKEN.finditer(text):
        t = m.group()
        if t.isspace() or t.startswith(("//", "/*")):
            continue
        result.append(t)
    return result


def functions(text):
    ts = tokens(text)
    found = {}
    depth = 0
    for i, t in enumerate(ts):
        if t != "{" or depth:
            depth += (t == "{") - (t == "}")
            continue
        depth = 1
        if i and ts[i - 1] == ")":
            nest = 1
            j = i - 2
            while j >= 0 and nest:
                nest += (ts[j] == ")") - (ts[j] == "(")
                j -= 1
            if j >= 0 and re.fullmatch(r"[A-Za-z_]\w*", ts[j]):
                end = i + 1
                level = 1
                while end < len(ts) and level:
                    level += (ts[end] == "{") - (ts[end] == "}")
                    end += 1
                found[ts[j]] = ts[i:end]
    return found


def function_body_source(text, name):
    """Return an audited function body without assuming its source formatting."""
    expected = functions(text).get(name)
    if expected is None:
        raise ValueError("Missing function body: " + name)
    spans = [
        match
        for match in TOKEN.finditer(text)
        if not match.group().isspace() and not match.group().startswith(("//", "/*"))
    ]
    values = [match.group() for match in spans]
    matches = [
        index
        for index, value in enumerate(values)
        if value == "{" and values[index : index + len(expected)] == expected
    ]
    if len(matches) != 1:
        raise ValueError("Function body must be unique: " + name)
    start = matches[0]
    return text[spans[start].start() : spans[start + len(expected) - 1].end()]


def compare(old, new, excluded=(), generic_labels=False):
    old_text = Path(old).read_text()
    if generic_labels:
        old_text = old_text.replace("llama7b_", "micromix_").replace("LLAMA7B_", "MICROMIX_")
    a = functions(old_text)
    b = functions(Path(new).read_text())
    missing = a.keys() - b.keys() - set(excluded)
    if missing:
        raise ValueError("Missing function body: " + ",".join(sorted(missing)))
    names = sorted((a.keys() & b.keys()) - set(excluded))
    if not names:
        raise ValueError("No shared function definitions")
    differences = [name for name in names if a[name] != b[name]]
    if differences:
        raise ValueError("Function body changed: " + ",".join(differences))
    return dict(
        functions=names,
        count=len(names),
        pass_=True,
        old_sha256=hashlib.sha256(Path(old).read_bytes()).hexdigest(),
        new_sha256=hashlib.sha256(Path(new).read_bytes()).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--tiny", type=Path, required=True)
    p.add_argument("--generic", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    results = dict(
        api=compare(a.api, K / "include/mx_micromix.h", ("mxmm_decode", "mxmm_encode")),
        formats=compare(
            a.api,
            K / "include/mx_micromix_formats.h",
            tuple(functions(a.api.read_text()).keys() - {"mxmm_decode", "mxmm_encode"}),
        ),
        prepared=compare(a.prepared, K / "include/mx_micromix_prepared.h"),
        tiny=compare(
            a.tiny, K / "src/micromix_prefill_tinyllama.c", ("main", "mxmm_layer_project")
        ),
        generic=compare(
            a.generic,
            K / "src/micromix_prefill_generic.c",
            ("main", "mxmm_layer_project"),
            generic_labels=True,
        ),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "pass": True,
                "scope": "shared hot C function token bodies; asm strings retained; only generic labels normalized",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(a.out)


if __name__ == "__main__":
    main()
