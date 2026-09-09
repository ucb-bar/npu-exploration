"""config/recipes/baseline.json must keep describing the hardware we actually have.

`baseline` is the claim "this is the stock MX-Gemmini". That claim is written down in
five places that nothing keeps in sync:

  1. Chisel      ConfigsFP.scala          meshRows / meshProd… / meshAcc…   (the design)
  2. spike       libgemmini/gemmini.cc    prod_e, prod_m, acc_e[], acc_m[]  (what runs)
  3. spike       libgemmini/gemmini_params.h              DIM
  4. compile     gemmini-rocc-tests/include/gemmini_params.h  DIM
  5. planner     mxgemm_emit.DEFAULT_GEOMETRY / BLOCK_SCALE_GROUP

They have already drifted once -- ACC_ROWS is 1024 in (3) and 512 in (4) -- so this is
a live failure mode, not a hypothetical one. It happens to be harmless (the MX path
does not touch the accumulator memory), which is exactly why it went unnoticed.

If this test fails, the golden model is describing a machine nobody built.

Run:  .venv/bin/python tests/test_recipe_drift.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config.recipe import load  # noqa: E402


def gemmini_root() -> Path:
    """The pinned hw/gemmini submodule — the recipe is held to THESE sources, which are
    exactly what config/build_spike.py compiles. The chipyard tree no longer speaks here."""
    return REPO / "hw" / "gemmini"


def parse_gemmini_cc(p: Path) -> dict:
    t = p.read_text(encoding="utf-8")

    def one(pat: str, what: str):
        m = re.findall(pat, t)
        if len(m) != 1:
            raise AssertionError(f"{what}: expected 1 match in {p.name}, got {len(m)}")
        return m[0]

    prod = one(r"const int prod_e = (\d+), prod_m = (\d+);", "prod precision")
    acc_e = one(r"const int8_t acc_e\[16\] = \{([^}]*)\};", "acc_e")
    acc_m = one(r"const int8_t acc_m\[16\] = \{([^}]*)\};", "acc_m")
    group = one(r"const int GROUP = (\d+);", "GROUP")
    return {"prod_e": int(prod[0]), "prod_m": int(prod[1]),
            "acc_e": tuple(int(x) for x in acc_e.split(",")),
            "acc_m": tuple(int(x) for x in acc_m.split(",")),
            "block": int(group)}


def parse_scala(p: Path) -> dict:
    """Read the precision lists out of GemminiMxFPConfigs.defaultMxFPConfig.

    ``MxFloat(expWidth, sigWidth, ...)`` counts the implicit bit, so the C's mantissa
    field is ``sigWidth - 1``. That identity is the whole reason the two agree, so the
    test asserts it rather than assuming it.
    """
    t = p.read_text(encoding="utf-8")
    body = t[t.index("val defaultMxFPConfig"):]
    body = body[:body.index("val testMxFPConfig")]

    def fills(field: str) -> tuple[tuple[int, int], ...]:
        """Expand ``Seq.fill(n){MxFloat(e,s,..)} ++ ...`` into 16 (exp, sig) pairs.

        The field's value ends at the next top-level ``name =`` assignment, so the
        two lists cannot bleed into each other.
        """
        seg = body[body.index(field) + len(field):]
        end = re.search(r"\n\s{4}\w+\s*=", seg)
        seg = seg[:end.start()] if end else seg
        out: list[tuple[int, int]] = []
        for n, e, sig in re.findall(r"Seq\.fill\((\d+)\)\s*\{\s*MxFloat\((\d+),\s*(\d+)", seg):
            out.extend([(int(e), int(sig))] * int(n))
        if len(out) != 16:
            raise AssertionError(f"{field}: expanded to {len(out)} entries, expected 16")
        return tuple(out)

    mesh = re.search(r"meshRows\s*=\s*(\d+),\s*\n\s*meshColumns\s*=\s*(\d+)", body)
    prod, acc = fills("meshProdPrecisionList"), fills("meshAccPrecisionList")
    return {"dim": (int(mesh.group(1)), int(mesh.group(2))) if mesh else None,
            "prod": prod, "acc": acc}


def parse_define(p: Path, name: str) -> int | None:
    m = re.search(rf"#define {name}\s+(\d+)", p.read_text(encoding="utf-8"))
    return int(m.group(1)) if m else None


def main() -> int:
    r = load("baseline")
    g = gemmini_root()
    lg = g / "software/libgemmini"
    rt = g / "software/gemmini-rocc-tests/include"
    scala = g / "src/main/scala/gemmini/ConfigsFP.scala"

    if not (lg / "gemmini.cc").exists():
        print(f"SKIP: hw/gemmini submodule not initialized at {g} "
              "(git submodule update --init --recursive hw/gemmini)")
        return 0

    fails = []

    def check(label: str, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {label:44s} {got}")
        if not ok:
            fails.append(f"{label}: recipe says {want}, source says {got}")

    print("baseline.json vs libgemmini/gemmini.cc (what spike runs)")
    cc = parse_gemmini_cc(lg / "gemmini.cc")
    check("prod_e", cc["prod_e"], r.prod_e)
    check("prod_m", cc["prod_m"], r.prod_m)
    check("acc_e[16]", cc["acc_e"], r.acc_e)
    check("acc_m[16]", cc["acc_m"], r.acc_m)
    check("GROUP", cc["block"], r.block)

    print("\nbaseline.json vs ConfigsFP.scala (the design the RTL elaborates)")
    if scala.exists():
        sc = parse_scala(scala)
        check("meshRows == meshColumns == dim", sc["dim"], (r.dim, r.dim))
        check("meshProdPrecisionList -> (e, m=sig-1)", tuple((e, s - 1) for e, s in sc["prod"]),
              tuple((r.prod_e, r.prod_m) for _ in range(r.dim)))
        check("meshAccPrecisionList  -> (e, m=sig-1)", tuple((e, s - 1) for e, s in sc["acc"]),
              tuple(zip(r.acc_e, r.acc_m)))
    else:
        print(f"  skip  ConfigsFP.scala not found at {scala}")

    print("\nbaseline.json vs the two gemmini_params.h (spike's and the compiler's)")
    check("libgemmini DIM", parse_define(lg / "gemmini_params.h", "DIM"), r.dim)
    check("rocc-tests DIM", parse_define(rt / "gemmini_params.h", "DIM"), r.dim)

    print("\nbaseline.json vs the backend planner")
    sys.path.insert(0, str(REPO / "compiler/targets/mx_gemmini_rocket"))
    from backend import mxgemm_emit
    check("DEFAULT_GEOMETRY['dim']", mxgemm_emit.DEFAULT_GEOMETRY["dim"], r.dim)
    check("BLOCK_SCALE_GROUP", mxgemm_emit.BLOCK_SCALE_GROUP, r.block)

    print("\nschema v2: derived per-format geometry vs the reference models")
    from config.recipe import RecipeError, derive_format
    from config.recipe import parse as parse_recipe
    f8 = derive_format("fp8", r.dim)
    check("fp8 tile", f8.tile, (16, 16, 16))
    check("fp8 (codes_per_byte, prod_frac_bits)", (f8.codes_per_byte, f8.prod_frac_bits), (1, 7))
    fp4_ref = rt.parent / "fp4_matmul_model.py"
    if fp4_ref.exists():
        t4 = fp4_ref.read_text(encoding="utf-8")
        want4 = {k: int(re.search(rf"^{k}\s*=\s*(\d+)", t4, re.M).group(1))
                 for k in ("TILE_M", "TILE_N", "TILE_K", "PROD_MANT_BITS")}
        f4 = derive_format("fp4", r.dim)
        check("fp4 tile vs fp4_matmul_model.py", f4.tile,
              (want4["TILE_M"], want4["TILE_N"], want4["TILE_K"]))
        check("fp4 prod_frac_bits vs fp4_matmul_model.py", f4.prod_frac_bits,
              want4["PROD_MANT_BITS"])
        check("fp4 codes_per_byte", f4.codes_per_byte, 2)
    else:
        print(f"  skip  fp4_matmul_model.py not found at {fp4_ref}")

    print("\nschema v2: formats stays out of build_id; illegal blocks fail closed")
    import copy
    raw2 = copy.deepcopy(r.raw)
    raw2["formats"] = {"fp8": {"tile": [16, 16, 16]}}
    check("build_id unchanged by an explicit formats block",
          parse_recipe(raw2).build_id(), r.build_id())

    def rejects(label, formats_block):
        raw3 = copy.deepcopy(r.raw)
        raw3["formats"] = formats_block
        try:
            parse_recipe(raw3)
            check(label, "accepted", "RecipeError")
        except RecipeError:
            check(label, "RecipeError", "RecipeError")

    rejects("contradictory formats.fp8.tile rejected", {"fp8": {"tile": [8, 8, 8]}})
    rejects("formats.fp6 rejected (unpinned)", {"fp6": {"via_lut": True}})
    rejects("unknown key under formats.fp8 rejected", {"fp8": {"bogus": 1}})

    print()
    if fails:
        print(f"{len(fails)} DRIFT(S) — the baseline recipe no longer describes the hardware:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("NO DRIFT — baseline.json agrees with Chisel, spike, both headers and the planner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
