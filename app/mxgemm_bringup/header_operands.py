"""Read MX operand codes + E8M0 scales out of a gemmini-rocc-tests data header.

App-layer, operand-specific: this knows about a particular C header shipped with the gemmini tests.
The compiler backend knows nothing about it — it receives the parsed arrays on the command buffer as
``mx_operands`` and never reads a file.

Parsing is structural (brace matching + split), not pattern matching: a regex that assumes one
spelling of a C array declaration silently returns nothing when the spelling changes, and a silently
empty operand set produces a kernel that runs and computes zeros.
"""
from __future__ import annotations

from pathlib import Path


class HeaderParseError(RuntimeError):
    """The header did not contain a declaration in the expected shape. Raised rather than returning
    an empty array — see the module docstring."""


def _find_initializer(text: str, name: str) -> str:
    """Return the brace-balanced initializer body for ``<type> <name>[...] = { ... };``."""
    for marker in (f" {name}[", f"*{name}[", f" {name} ["):
        idx = text.find(marker)
        if idx != -1:
            break
    else:
        raise HeaderParseError(f"no declaration of {name!r} found")
    eq = text.find("=", idx)
    open_brace = text.find("{", eq)
    if eq == -1 or open_brace == -1:
        raise HeaderParseError(f"{name!r} has no initializer")
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
    raise HeaderParseError(f"{name!r} initializer is not brace-balanced")


def _parse_2d(text: str, name: str) -> list[list[int]]:
    """Parse a 2-D integer initializer into rows. Accepts C integer literals in any base."""
    body = _find_initializer(text, name)
    rows: list[list[int]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(body):
        if ch == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                rows.append([int(tok.strip(), 0) for tok in body[start:i].split(",") if tok.strip()])
    if not rows:
        raise HeaderParseError(f"{name!r} parsed to zero rows")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise HeaderParseError(f"{name!r} has ragged rows: widths {sorted(widths)}")
    return rows


def load_mx_operands(header: str | Path, *, with_golden: bool = True) -> dict[str, list[list[int]]]:
    """Parse a header into the ``mx_operands`` side-channel bundle the backend expects."""
    text = Path(header).read_text(encoding="utf-8")
    ops = {
        "a_codes": _parse_2d(text, "A_in"),
        "b_codes": _parse_2d(text, "B_in"),
        "a_scales": _parse_2d(text, "A_scales_row"),
        "b_scales": _parse_2d(text, "B_scales_col"),
    }
    if with_golden:
        ops["golden_bf16"] = _parse_2d(text, "C_out_bf16")
    return ops
