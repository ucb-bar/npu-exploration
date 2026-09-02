"""Two chained MX GEMMs — a reproducible case for RTL evaluation.

GEMM1 (`A @ W1`) commits through the **requantizer** to FP8 codes + E8M0 scales; GEMM2 consumes that
intermediate directly and commits to BF16. This is the composition the V1 spad-resident path exists
to make on-device, and it is the first thing in this repo that chains two mesh operations.

The seam needs care. The requantizer normalizes each output block's max to the element format's full
range (`MxRequantizer.scala`: `log2_pmax_floor = 8` for FP8 -> codes peak at 448), while the mesh
accumulates a 16-deep column at exponent width 4 for 15 of its 16 rows
(`ConfigsFP.scala: meshAccPrecisionList`), saturating near 2**8. Feeding one straight into the other
overflows on spike: 4096/4096 NaN. Two ways to close it, both here so RTL can judge both:

* ``weight``  — leave the intermediate untouched and absorb the shift in W2's block scales. Nothing
  extra happens between the GEMMs, so the intermediate could stay resident on device.
* ``rescale`` — re-split the intermediate's (code, E8M0 scale) pair on the host: divide codes by
  2**6, add 6 to the scale. Value-identical and lossless, but the intermediate makes a host trip.

Both are pure *scale-factor* choices — nothing is added to the representation, the same value is
just split differently between the fp8 code and its block exponent.

Everything needed to replay this on Verilator is written to the artifact directory: both interface
MLIR modules, every operand array, the emitted C, and spike's outputs.

    .venv/bin/python app/chain_2gemm/run_chain.py
    .venv/bin/python app/chain_2gemm/run_chain.py --variant rescale --m 64 --k 64 --n1 64 --n2 64
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "app"))
sys.path.insert(0, str(REPO / "compiler" / "targets" / "mx_gemmini_rocket"))

import backend as mx          # noqa: E402
import mxiface                # noqa: E402
import mxquant as q           # noqa: E402

#: W2 block-scale target for the `weight` variant. Chosen by measurement, not derivation: with the
#: intermediate pinned at peak 448 the safe exponent is where 16-deep accumulation stays under 2**8.
#: Measured NaN counts, A untouched: exp 2 -> 4096, 0 -> 3972, -2 -> 341, -4 -> 0, -6 -> 0.
#: -4 is the largest (most precision-preserving) value that measured clean; -6 adds margin.
W2_TARGET_EXP = -4


def nan_count(bits: np.ndarray) -> int:
    return int(((bits & 0x7F80) == 0x7F80).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--k", type=int, default=64, help="GEMM1 contraction")
    ap.add_argument("--n1", type=int, default=64, help="hidden width = GEMM2's contraction")
    ap.add_argument("--n2", type=int, default=64, help="output width")
    ap.add_argument("--variant", choices=("weight", "rescale"), default="weight")
    ap.add_argument("--w2-exp", type=int, default=W2_TARGET_EXP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--artifacts", type=Path, default=REPO / "out" / "artifacts" / "chain_2gemm")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    A = rng.standard_normal((a.m, a.k)).astype(np.float32)
    W1 = rng.standard_normal((a.k, a.n1)).astype(np.float32)
    W2 = rng.standard_normal((a.n1, a.n2)).astype(np.float32)
    art = a.artifacts / a.variant
    art.mkdir(parents=True, exist_ok=True)

    # ---- GEMM 1: A @ W1 -> FP8 via the requantizer ----------------------------------------------
    iface1 = mxiface.matmul_interface_mlir(a.m, a.n1, a.k, out_dtype="f8E4M3FN", out="T0")
    ops1 = q.quantize_matmul_operands(A, W1)
    cb1 = mxiface.to_command_buffer(iface1, ops1)
    r1 = mx.run_command_buffer(cb1, workdir=art / "gemm1")
    T0_codes = np.array(r1["outputs"]["T0"], dtype=np.uint8)
    T0_scales = np.array(r1["outputs"]["T0_scales"], dtype=np.uint8)
    print(f"GEMM1  {a.m}x{a.n1}x{a.k} -> fp8   "
          f"peak code {np.abs(q.fp8_e4m3_decode(T0_codes)).max():.0f}, "
          f"E8M0 {T0_scales.min()}..{T0_scales.max()}, cycles {r1['metrics'].get('cycles')}")

    # ---- the seam -------------------------------------------------------------------------------
    if a.variant == "rescale":
        a2_codes, a2_scales = q.rescale_for_next_gemm(T0_codes, T0_scales)
        w2_exp = 2
        note = f"host re-split of (code, scale) by 2**{q.CHAIN_EXP_SHIFT}; value-identical"
    else:
        a2_codes, a2_scales = T0_codes, np.ascontiguousarray(T0_scales.T)
        w2_exp = a.w2_exp
        note = f"intermediate untouched; W2 block scales target 2**{w2_exp}"
    print(f"seam   {note}\n       A2 peak code {np.abs(q.fp8_e4m3_decode(a2_codes)).max():.4g}")

    # ---- GEMM 2: T0 @ W2 -> BF16 ----------------------------------------------------------------
    w2c, w2s = q.quantize_rows(np.ascontiguousarray(W2.T), target_exp=w2_exp)
    ops2 = {"a_codes": a2_codes, "a_scales": a2_scales,
            "b_codes": np.ascontiguousarray(w2c.T), "b_scales": w2s}
    iface2 = mxiface.matmul_interface_mlir(a.m, a.n2, a.n1, lhs="T0", weight="W2")
    cb2 = mxiface.to_command_buffer(iface2, ops2)
    r2 = mx.run_command_buffer(cb2, workdir=art / "gemm2")
    bits = np.array(r2["outputs"]["Y0"], dtype=np.uint16)
    bad = nan_count(bits)
    got = q.bf16_bits_to_float(bits)
    print(f"GEMM2  {a.m}x{a.n2}x{a.n1} -> bf16  NaN/inf {bad}/{bits.size}, "
          f"cycles {r2['metrics'].get('cycles')}")
    cos = float("nan")
    if bad == 0:
        # Golden-free sanity: the fp32 chain on the ORIGINAL tensors. Not a bit-exact oracle (the
        # device is fp8 operands / bf16 accumulate), but a wrong chain does not land near it.
        ref = (A @ W1) @ W2
        cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref)))
        print(f"       range [{got.min():.4g}, {got.max():.4g}]  "
              f"|Y|mean {np.abs(got).mean():.4g}   cos vs fp32 chain {cos:.6f}")

    # ---- artifacts for the RTL replay -----------------------------------------------------------
    (art / "gemm1.interface.mlir").write_text(iface1, encoding="utf-8")
    (art / "gemm2.interface.mlir").write_text(iface2, encoding="utf-8")
    for name, src in (("gemm1", art / "gemm1" / "main.c"), ("gemm2", art / "gemm2" / "main.c")):
        if src.exists():
            shutil.copy(src, art / f"{name}.c")
    np.savez_compressed(
        art / "operands.npz",
        A=A, W1=W1, W2=W2,
        g1_a_codes=ops1["a_codes"], g1_a_scales=ops1["a_scales"],
        g1_b_codes=ops1["b_codes"], g1_b_scales=ops1["b_scales"],
        T0_codes=T0_codes, T0_scales=T0_scales,
        g2_a_codes=a2_codes, g2_a_scales=a2_scales,
        g2_b_codes=ops2["b_codes"], g2_b_scales=ops2["b_scales"],
        Y0_bits=bits)
    (art / "spike_result.json").write_text(json.dumps({
        "shapes": {"m": a.m, "k": a.k, "n1": a.n1, "n2": a.n2},
        "variant": a.variant, "seam": note, "w2_target_exp": w2_exp, "seed": a.seed,
        "gemm1": {"peak_code": float(np.abs(q.fp8_e4m3_decode(T0_codes)).max()),
                  "e8m0_min": int(T0_scales.min()), "e8m0_max": int(T0_scales.max()),
                  "metrics": r1["metrics"]},
        "gemm2": {"nan_or_inf": bad, "total": int(bits.size), "metrics": r2["metrics"],
                  "cos_vs_fp32_chain": cos},
        "oracle": r1["oracle"],
    }, indent=2), encoding="utf-8")
    print(f"\nartifacts -> {art}")
    print("  gemm{1,2}.interface.mlir, gemm{1,2}.c, operands.npz, spike_result.json")
    print("  replay on RTL: same ELFs, compare Y0_bits (spike is derived_from_rtl=False)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
