#!/usr/bin/env python3
"""Compare an RTL output C (BF16, [M][N] row-major) of lutk_proj.c on TinyLlama data against
  1. the bit-exact mesh golden in data/lutk_<tag>.bin (should match exactly; reports mismatches and max ULP), and
  2. the MXQuant study's fp32 output of the same variant (y_study_<i> in data/lutk_<tag>_ref.npz): the RTL error
     relative to the study must be the same size as the golden's own (BF16 accumulation), far below the LUT's
     effect (study vs all-MXFP8), which is what the perplexity numbers measure.

RTL output formats: --rtl c.bin (raw little-endian uint16 BF16), c.npy, or a text log holding the BF16 words as
hex (0x3f80 ... ; every 4-hex-digit token after --hex-prefix, M*N of them, row-major).

    python gen/compare_lutk_tinyllama.py --tag tl_down_g1 --var b6 --dim 16 --rtl out/c_tl_down_g1_lut1.bin
    python gen/compare_lutk_tinyllama.py --tag tl_down_g1 --var b6 --dim 16 --self-test   # golden vs itself
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

DATA = Path("/tools/scratch/nicorakela/radiance-cy-dev/generators/gemmini/npu-exploration/baremetal/mxgemmini/data")


def offsets(h):
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"#define LK_OFF_(\w+) (\d+)", h.read_text())}


def f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_rtl(path, M, N, prefix):
    p = Path(path)
    if p.suffix == ".npy":
        x = np.load(p)
        return x.view(np.uint16) if x.dtype != np.uint16 and x.dtype.itemsize == 2 else x.astype(np.uint16)
    raw = p.read_bytes()
    if p.suffix == ".bin":
        return np.frombuffer(raw, np.uint16)[:M * N].reshape(M, N)
    toks = re.findall(re.escape(prefix) + r"([0-9a-fA-F]{4})\b", raw.decode(errors="replace"))
    assert len(toks) >= M * N, f"{p}: {len(toks)} hex words, want {M * N}"
    return np.array([int(t, 16) for t in toks[:M * N]], np.uint16).reshape(M, N)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--var", required=True, help="variant name (b5, b6, b7, all) or index")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--golden", default="CML", help="golden of the kernel's schedule: CML (mixed-K, one accumulation), "
                                                  "CSL (shared acc, LUT K-blocks first), CLUT (two partials + CPU add)")
    ap.add_argument("--rtl", help="RTL output file (.bin / .npy / text log)")
    ap.add_argument("--hex-prefix", default="0x")
    ap.add_argument("--self-test", action="store_true", help="compare the golden with itself (checks the plumbing)")
    ap.add_argument("--data", default=str(DATA))
    a = ap.parse_args()

    data = Path(a.data)
    z = np.load(data / f"lutk_{a.tag}_ref.npz")
    meta = json.loads(str(z["meta"]))
    M, N = meta["m"], meta["n"]
    names = meta["variants"]
    vi = int(a.var) if a.var.isdigit() else names.index(a.var)
    blob = (data / f"lutk_{a.tag}.bin").read_bytes()
    off = offsets(data / f"lutk_{a.tag}.h")
    key = f"{a.golden}{vi}_D{a.dim}"
    gold = np.frombuffer(blob, np.uint16, M * N, off[key]).reshape(M, N)
    fp8 = np.frombuffer(blob, np.uint16, M * N, off[f"FP8_D{a.dim}"]).reshape(M, N)
    rtl = gold.copy() if a.self_test else load_rtl(a.rtl, M, N, a.hex_prefix)

    ys, y8, yr = z[f"y_study_{vi}"], z["y_fp8"], z["y_real"]
    rel = lambda u, v: float(np.linalg.norm(u - v) / np.linalg.norm(v))
    ulp = np.abs(rtl.astype(np.int32) - gold.astype(np.int32))      # same-sign BF16 words: integer distance = ULPs
    sign = (rtl ^ gold) & 0x8000
    print(f"{meta['layer']}  M={M} K={meta['k']} N={N} G={meta['g']}  variant {names[vi]} "
          f"({sum(meta['masks'][names[vi]])}/{len(meta['masks'][names[vi]])} LUT K-blocks)  dim{a.dim}  golden {key}")
    print(f"  bit-exact vs golden : {int((rtl == gold).sum())}/{M * N} equal, {int((rtl != gold).sum())} differ, "
          f"max {int(ulp[sign == 0].max(initial=0))} ULP, {int(sign.astype(bool).sum())} sign flips")
    print(f"  rel err vs study fp32 output   : RTL {rel(f32(rtl), ys):.3e}   golden {rel(f32(gold), ys):.3e}"
          f"   (BF16 accumulation; should be equal)")
    print(f"  LUT effect, study vs all-MXFP8 : {rel(ys, y8):.3e}   RTL vs all-MXFP8 golden {rel(f32(rtl), f32(fp8)):.3e}")
    print(f"  study vs unquantized fp32      : {rel(ys, yr):.3e}")
    ok = bool((rtl == gold).all())
    print("  PASS (bit-exact)" if ok else "  FAIL: not bit-exact; check the error lines above against the LUT effect")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
