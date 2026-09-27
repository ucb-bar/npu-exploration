"""`app/mxarith.py` must stay identical to the model it was extracted from.

Those five primitives — the PE's truncating product, the per-lane RNE quantizer, the exact add, and
the bf16 pair used for cross-tile accumulation — used to be IMPORTED from
`gemmini-rocc-tests/fp8_matmul_model.py`, so there was one implementation and it could not drift.
Extracting them into this repo removed the last runtime dependency the graded path had on that tree,
but it also created a second copy.

This test is what buys the property back. Two checks:

1. **Textual.** Re-run the extraction against the current upstream file and diff. Anything but an
   exact match means the model changed and `app/mxarith.py` is stale.
2. **Behavioural.** Run both implementations over random and edge-case input — subnormals, zeros,
   infinities, NaN, exact ties — and require elementwise identity, NaN patterns included.

Skips cleanly when the reference tree is absent, because the extracted copy is self-sufficient; the
point of the test is to catch upstream moving, not to reintroduce the dependency.

    .venv/bin/python tests/selftest_mxarith.py            # check
    .venv/bin/python tests/selftest_mxarith.py --update   # re-extract after an upstream change
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The chipyard tree ($MERLIN_CHIPYARD) is the canonical reference; the old sibling
# software/ layout stays as a fallback for checkouts that used it.
from models.spike.build_spike import gemmini_root  # noqa: E402

ROCC = gemmini_root() / "software" / "gemmini-rocc-tests"
if not ROCC.is_dir():
    ROCC = REPO.parent / "software" / "gemmini-rocc-tests"

#: every extracted module: (upstream file, our copy, entry points). Keep in step with the
#: `tools/extract_model.py` invocations recorded in planning/merlin_glue_port_plan.md section 4.16.
MODULES = [
    (ROCC / "fp8_matmul_model.py", REPO / "app" / "mxmesh" / "fp8.py",
     ["tiled_matmul_hwlike", "matrix_mx_requantize", "tensor_to_custom_fp_codes",
      "make_fp_quantizer", "parse_fp_spec", "mx_product_quantize_trunc", "fp_quantize_rne",
      "fp_add_exact", "q_bf16_rne", "bf16_accum_add"]),
    (ROCC / "fp4_matmul_model.py", REPO / "app" / "mxmesh" / "fp4.py",
     ["tiled_matmul_hwlike", "matrix_mx_requantize", "tensor_to_custom_fp_codes"]),
]


sys.path.insert(0, str(REPO / "tools"))
from extract_model import closure                      # noqa: E402


def body_of(text: str) -> str:
    """Everything after the generated preamble, i.e. the extracted region."""
    marker = "QuantFn = Optional[Callable[[Tensor], Tensor]]"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    marker = "Tensor = torch.Tensor"
    return text.split(marker, 1)[1].strip() if marker in text else text.strip()


def edge_cases() -> torch.Tensor:
    """Values that separate a faithful primitive from an approximate one."""
    vals = [0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.0, 255.0, 256.0, 448.0, 480.0,
            2.0 ** -9, 2.0 ** -10, 2.0 ** -23, 2.0 ** 8, 2.0 ** 9,
            1.0 + 2.0 ** -8, 3.0 / 8.0, 1e-38, 1e38,
            float("inf"), float("-inf"), float("nan")]
    rng = np.random.default_rng(0)
    rand = rng.standard_normal(4096).astype(np.float32) * rng.choice([1e-6, 1.0, 1e3], 4096)
    return torch.tensor(vals + rand.tolist(), dtype=torch.float32)


def same(a: torch.Tensor, b: torch.Tensor) -> int:
    """Count of differing elements, treating NaN as equal to NaN."""
    an, bn = torch.isnan(a), torch.isnan(b)
    return int(((a != b) & ~(an & bn)).sum())


def main(argv: list[str]) -> int:
    if not ROCC.is_dir():
        print(f"SKIP: reference tree absent ({ROCC}); the extracted copies are self-sufficient")
        return 0

    fails = checks = 0

    print("[1] textual: every extraction still matches its upstream ---------")
    for up, ours, entries in MODULES:
        if not up.exists():
            print(f"  SKIP  {ours.name}: {up.name} absent")
            continue
        fresh, _ = closure(up.read_text(), entries)
        ok = body_of(ours.read_text()) == fresh.strip()
        checks += 1
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {ours.relative_to(REPO)} == extract({up.name})")
        if not ok:
            print("        upstream changed; re-extract with tools/extract_model.py, "
                  "then re-check the numbers below")

    print("\n[2] behavioural: elementwise identity on edge cases + random ----")
    sys.path.insert(0, str(ROCC))
    import fp8_matmul_model as UP                        # noqa: E402
    from app.mxmesh import fp8 as OURS                   # noqa: E402

    x = edge_cases()
    y = torch.roll(x, 7)
    cases = [
        ("q_bf16_rne(x)", lambda M: M.q_bf16_rne(x)),
        ("bf16_accum_add(x, y)", lambda M: M.bf16_accum_add(x, y)),
        ("mx_product_quantize_trunc(x, 4, 3)", lambda M: M.mx_product_quantize_trunc(x, 4, 3)),
        ("fp_quantize_rne(x, 8, 7)", lambda M: M.fp_quantize_rne(x, 8, 7)),
        ("fp_quantize_rne(x, 4, 4)", lambda M: M.fp_quantize_rne(x, 4, 4)),
        ("fp_quantize_rne(x, 4, 6)", lambda M: M.fp_quantize_rne(x, 4, 6)),
        ("fp_add_exact(x, y, 4, 4)", lambda M: M.fp_add_exact(x, y, 4, 4)),
        ("fp_add_exact(x, y, 8, 7)", lambda M: M.fp_add_exact(x, y, 8, 7)),
    ]
    for label, f in cases:
        try:
            d = same(f(OURS), f(UP))
        except Exception as exc:
            print(f"  FAIL  {label:38s} {type(exc).__name__}: {exc}")
            fails += 1
            continue
        checks += 1
        fails += d != 0
        print(f"  {'PASS' if d == 0 else 'FAIL'}  {label:38s} {d}/{x.numel()} differ")

    print("\n[3] the MESH agrees, not just the primitives ---------------------")
    import numpy as _np
    rng = _np.random.default_rng(0)
    A = torch.tensor(rng.standard_normal((32, 64)), dtype=torch.float32)
    B = torch.tensor(rng.standard_normal((64, 32)), dtype=torch.float32)
    sa = torch.ones(32, 2, dtype=torch.float32)
    sb = torch.ones(2, 32, dtype=torch.float32)
    prod = [(4, 3)] * 16
    acc = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)] * 1
    kw = dict(verbose=False, prod_precision_list=prod, acc_precision_list=acc)
    d = same(OURS.tiled_matmul_hwlike(A, B, sa, sb, **kw),
             UP.tiled_matmul_hwlike(A, B, sa, sb, **kw))
    checks += 1
    fails += d != 0
    print(f"  {'PASS' if d == 0 else 'FAIL'}  tiled_matmul_hwlike(32x64x32)          {d}/1024 differ")

    print(f"\n{checks} checks, {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
