"""config/hardware/baseline.json must keep describing the hardware we actually have.

`baseline` is the claim "this is the stock MX-Gemmini". That claim is written down in
several places that nothing else keeps in sync:

  1. Chisel      ConfigsFP.scala          meshRows / meshProd… / meshAcc…   (the design)
  2. spike       libgemmini/gemmini.cc    prod_e, prod_m, acc_e[], acc_m[]  (what runs)
  3. spike       libgemmini/gemmini_params.h              DIM
  4. compile     gemmini-rocc-tests/include/gemmini_params.h  DIM
  5. planner     mxgemm_emit.DEFAULT_GEOMETRY / BLOCK_SCALE_GROUP
  6. spike       libgemmini/gemmini_params.h              BANK_NUM, BANK_ROWS (the scratchpad)
  7. rtl_exact   mxgemmini_rtl.json, acc_schedule.csv, rtl_datapath.PROD_FLOOR
  8. mxquant     models/mxquant/block.SCALE_FLOOR vs config.recipe's kernel-path constants

They have already drifted once -- ACC_ROWS is 1024 in (3) and 512 in (4) -- so this is
a live failure mode, not a hypothetical one. It happens to be harmless (the MX path
does not touch the accumulator memory), which is exactly why it went unnoticed.

If this test fails, the models are describing a machine nobody built.

Run:  .venv/bin/python tests/test_recipe_drift.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config.recipe import (KERNEL_BLOCK, KERNEL_DIM, KERNEL_SCALE_FLOOR, KERNEL_SCRATCHPAD,  # noqa: E402
                           load_hardware)


def gemmini_root() -> Path:
    """The gemmini sources, from the chipyard tree. The hw/gemmini pin is gone."""
    import os
    cy = os.environ.get("MERLIN_CHIPYARD")
    root = Path(cy) if cy else Path(__file__).resolve().parents[4]
    return root / "generators" / "gemmini"


def parse_gemmini_cc(p: Path) -> dict:
    t = p.read_text(encoding="utf-8")

    def one(pat: str, what: str):
        m = re.findall(pat, t)
        if len(m) != 1:
            raise AssertionError(f"{what}: expected 1 match in {p.name}, got {len(m)}")
        return m[0]

    prod = one(r"const int prod_e = (\d+), prod_m = (\d+);", "prod precision")
    group = one(r"const int GROUP = (\d+);", "GROUP")
    lit_e = re.findall(r"const int8_t acc_e\[16\] = \{([^}]*)\};", t)
    if lit_e:                                   # libgemmini before 9f10afe: two 16-entry literals
        acc_e = tuple(int(x) for x in lit_e[0].split(","))
        acc_m = tuple(int(x) for x in one(r"const int8_t acc_m\[16\] = \{([^}]*)\};", "acc_m").split(","))
    else:                                       # since: a per-lane ramp `for (kk < DIM)` with an #if on GEMMINI_DIM
        body = one(r"(?s)int8_t acc_e\[DIM\], acc_m\[DIM\];\s*for \(int kk = 0; kk < DIM; kk\+\+\) \{(.*?)#endif\s*\}",
                   "acc ladder loop") if "#if GEMMINI_DIM" in t else \
            one(r"(?s)int8_t acc_e\[DIM\], acc_m\[DIM\];\s*for \(int kk = 0; kk < DIM; kk\+\+\) \{(.*?)\n\s*\}", "acc ladder loop")
        ramp = body.split("#else")[-1]           # the stock (DIM=16) branch
        steps = [(int(n), int(e), int(m)) for n, e, m in
                 re.findall(r"if\s*\(kk < (\d+)\)\s*\{\s*acc_e\[kk\] = (\d+);\s*acc_m\[kk\] = (\d+);", ramp)]
        tail = re.search(r"else\s*\{\s*acc_e\[kk\] = (\d+);\s*acc_m\[kk\] = (\d+);", ramp)
        if not steps or not tail:
            raise AssertionError(f"acc ladder loop in {p.name}: could not read the ramp")
        lanes = []
        for kk in range(16):
            e, m = next(((e, m) for n, e, m in steps if kk < n), (int(tail.group(1)), int(tail.group(2))))
            lanes.append((e, m))
        acc_e, acc_m = tuple(e for e, _ in lanes), tuple(m for _, m in lanes)
    return {"prod_e": int(prod[0]), "prod_m": int(prod[1]),
            "acc_e": acc_e, "acc_m": acc_m, "block": int(group)}


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
    """``#define NAME <int>``; or, since libgemmini 9f10afe, ``#define DIM GEMMINI_DIM`` with the
    default ``#define GEMMINI_DIM <int>`` under an ``#ifndef`` (build_spike passes -DGEMMINI_DIM)."""
    t = p.read_text(encoding="utf-8")
    m = re.search(rf"#define {name}\s+(\d+)", t)
    if m:
        return int(m.group(1))
    alias = re.search(rf"#define {name}\s+\(?\s*(\w+)\s*\)?\s*$", t, re.M)
    if alias:
        m = re.search(rf"#define {alias.group(1)}\s+(\d+)", t)
        return int(m.group(1)) if m else None
    return None


def main() -> int:
    r = load_hardware("baseline")
    g = gemmini_root()
    lg = g / "software/libgemmini"
    rt = g / "software/gemmini-rocc-tests/include"
    scala = g / "src/main/scala/gemmini/ConfigsFP.scala"

    if not (lg / "gemmini.cc").exists():
        print(f"SKIP: gemmini sources not found at {g}; set MERLIN_CHIPYARD "
              "(see scripts/env.sh)")
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

    print("\nbaseline.json vs libgemmini/gemmini_params.h (spike's scratchpad at DIM 16)")
    gp = (lg / "gemmini_params.h").read_text(encoding="utf-8")
    bank_num = int(re.search(r"#define BANK_NUM\s+(\d+)", gp).group(1))
    rows = re.search(r"(?s)#if GEMMINI_DIM == 8.*?#else\s*#define BANK_ROWS\s+(\d+)", gp) \
        or re.search(r"#define BANK_ROWS\s+(\d+)", gp)
    check("BANK_NUM", bank_num, r.banks)
    check("BANK_ROWS", int(rows.group(1)), r.rows)

    print("\nbaseline.json vs rtl_exact (the extracted hardware model)")
    import json
    rj = json.loads((REPO / "rtl_exact" / "mxgemmini_rtl.json").read_text(encoding="utf-8"))
    check("mxgemmini_rtl.json mesh.dim", rj["mesh"]["dim"], r.dim)
    check("mxgemmini_rtl.json mesh.block", rj["mesh"]["block"], r.block)
    check("mxgemmini_rtl.json product (e, m)", (rj["formats"]["product"]["exp"], rj["formats"]["product"]["man"]),
          (r.prod_e, r.prod_m))
    check("mxgemmini_rtl.json acc_schedule", tuple(tuple(x) for x in rj["acc_schedule"]), tuple(zip(r.acc_e, r.acc_m)))
    csv = (REPO / "rtl_exact" / "acc_schedule.csv").read_text(encoding="utf-8").split()[1:]
    check("acc_schedule.csv", tuple(tuple(int(v) for v in ln.split(",")[1:]) for ln in csv), tuple(zip(r.acc_e, r.acc_m)))
    from rtl_exact import rtl_datapath
    check("rtl_datapath.PROD_FLOOR == types.prodFloor", rtl_datapath.PROD_FLOOR, r.prod_floor)

    print("\nkernel-path constants (config.recipe) vs baseline and the code they describe")
    check("KERNEL_DIM == baseline dim", KERNEL_DIM, r.dim)
    check("KERNEL_BLOCK == baseline block", KERNEL_BLOCK, r.block)
    check("KERNEL_SCRATCHPAD == baseline scratchpad", KERNEL_SCRATCHPAD, (r.banks, r.rows))
    from models.mxquant import block as mxblock
    check("KERNEL_SCALE_FLOOR == models/mxquant/block.SCALE_FLOOR", KERNEL_SCALE_FLOOR, mxblock.SCALE_FLOOR)

    print()
    if fails:
        print(f"{len(fails)} DRIFT(S) — the baseline recipe no longer describes the hardware:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("NO DRIFT — baseline.json agrees with Chisel, spike, both headers, the planner and rtl_exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
