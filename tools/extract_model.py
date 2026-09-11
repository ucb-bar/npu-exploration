"""Lift a closed set of functions out of a module, verbatim.

Several things this repo needs live inside large single-file models in `gemmini-rocc-tests` — the
datapath's exact-dyadic primitives, the bit-exact mesh, the element encoders — mixed in with header
writers and CLI drivers we do not want. Transcribing them by hand would create a second source of
truth that drifts silently; importing across trees makes every graded run depend on that tree.

So: **extract**. Walk the AST, take the transitive closure of a few entry points over module-level
definitions, and emit them in source order. The result is byte-for-byte the upstream code, which
makes "is our copy still correct?" a diff rather than a judgement — see
`tests/selftest_extracted.py`, which re-runs this and requires a textual match plus elementwise
agreement.

    python tools/extract_model.py <source.py> <out.py> <entry> [<entry> ...] [--header TEXT]
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def closure(src_text: str, entries: list[str]) -> tuple[str, list[str]]:
    """``(source, names)`` for ``entries`` plus every module-level name they reach, in file order."""
    lines = src_text.splitlines()
    tree = ast.parse(src_text)

    defs: dict[str, ast.AST] = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[n.name] = n
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    defs[t.id] = n

    missing = [e for e in entries if e not in defs]
    if missing:
        raise KeyError(f"entry point(s) not found at module level: {missing}")

    need, seen, chosen = list(entries), set(), []
    while need:
        name = need.pop(0)
        if name in seen or name not in defs:
            continue
        seen.add(name)
        node = defs[name]
        chosen.append(node)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in defs:
                need.append(sub.id)

    chosen.sort(key=lambda n: n.lineno)
    body = "\n\n".join("\n".join(lines[n.lineno - 1:n.end_lineno]) for n in chosen)
    names = sorted(seen & set(defs))
    return body, names


#: Everything the extracted bodies may reference from outside the closure. Kept explicit: an
#: extracted function that needs something not here fails at import, loudly, rather than at runtime.
PREAMBLE = """from __future__ import annotations

import math
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

Tensor = torch.Tensor
QuantFn = Optional[Callable[[Tensor], Tensor]]
"""


def render(src: Path, entries: list[str], *, header: str = "") -> str:
    body, names = closure(src.read_text(), entries)
    doc = (f'"""{header.rstrip()}\n\n'
           f"EXTRACTED VERBATIM from ``{src.name}`` by ``tools/extract_model.py`` — do not edit.\n"
           f"Entry points: {', '.join(entries)}\n"
           f"Closure: {len(names)} definitions.\n"
           f'"""\n')
    return doc + PREAMBLE + "\n\n" + body + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("entries", nargs="+")
    ap.add_argument("--header", default="")
    a = ap.parse_args()
    text = render(a.source, a.entries, header=a.header)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    print(f"{a.out}: {len(text.splitlines())} lines from {a.source.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
