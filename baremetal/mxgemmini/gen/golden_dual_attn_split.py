#!/usr/bin/env python3
"""Bit-exact goldens for src/dual_attn_llama.c in key-split mode (DA_KSPLIT): every pass (query head) of a
gen_attn_vpu.py --layer header runs as two partial flash passes -- keys [0, SKA) on the 16x16 (DIM 16 mesh, 8-lane RSUM),
keys [SKA, SK) on the 8x8 (DIM 8, 4-lane RSUM) -- each in src/attn_flash.c's order with no final 1/l (ATTN_PARTIAL), then
merged on the 16x16's VPU:
    m = max(m1, m2);  a_i = exp(m_i - m);  l = l1*a1 + l2*a2;  r = 1/l;  c_i = a_i*r;  O = O1*c1 + O2*c2
Writes data/<tag>_split<SKA>_expect.h: the merged O hash per pass, and the partials' O / m / l hashes (kernel layouts:
O flat [SQ][D] BF16, m and l [SQ][8] BF16 replicated).

    ../../../.venv/bin/python3 golden_dual_attn_split.py ../data/attn_vpu_dual_l8b.h --ska 1664 --bk 128
"""
from __future__ import annotations

import argparse
import os
import re
from multiprocessing import get_context
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

import golden_attn_flash as GF  # noqa: E402
import golden_llama_layer_e2e as GL  # noqa: E402
import golden_llama_layer_mx as GM  # noqa: E402

_A = {}


def partial(args):
    """One key-range pass in attn_flash.c's order without the final 1/l: (p, O, m, l)."""
    p, k0, k1 = args
    a = _A
    qc, qs = a["LQ"][p], a["LQS"][p].astype(np.int64) - a["fold"]
    kt, kts = a["KT"][:, k0:k1], a["KTS"][:, k0:k1]
    vc, vs = a["V"][k0:k1], a["VS"][k0 // 32:k1 // 32]
    off, m, L, O, alpha = 0, None, None, None, None
    for j, b in enumerate(GF.blocks(k1 - k0, a["bk"], a["bkf"], a["bkl"])):
        S = GM.mesh(GL.e4m3_decode(qc), qs.T, GL.e4m3_decode(kt[:, off:off + b]), kts[:, off:off + b], pool=False)
        mt = GL.v_rmax(S)
        if j == 0:
            m = mt
        else:
            mn = GL.v_max(m, mt); alpha = GL.v_exp(GL.v_sub(m, mn)); m = mn
        P = GL.v_exp(GL.v_sub(S, m[:, None]))
        lt = GM.v_rsum(P)
        L = lt if j == 0 else GL.v_add(GL.v_mul(L, alpha), lt)
        pc, ps = GL.spad_requant(P)
        Oj = GM.mesh(GL.e4m3_decode(pc), ps, GL.e4m3_decode(vc[off:off + b]), vs[off // 32:(off + b) // 32], pool=False)
        O = Oj if j == 0 else GL.v_add(GL.v_mul(O, alpha[:, None]), Oj)
        off += b
    return p, O, m, L


def merge(O1, m1, l1, O2, m2, l2):
    m = GL.v_max(m1, m2)
    a1 = GL.v_exp(GL.v_sub(m1, m)); a2 = GL.v_exp(GL.v_sub(m2, m))
    l = GL.v_add(GL.v_mul(l1, a1), GL.v_mul(l2, a2))
    r = GL.v_rcp(l)
    c1 = GL.v_mul(a1, r); c2 = GL.v_mul(a2, r)
    return GL.v_add(GL.v_mul(O1, c1[:, None]), GL.v_mul(O2, c2[:, None]))


def run_dim(dim, jobs):
    GM._DIM, GM._VL = dim, dim // 2          # read by the forked workers
    with get_context("fork").Pool(len(jobs)) as pool:
        return {p: (O, m, l) for p, O, m, l in pool.imap_unordered(partial, jobs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("header", type=Path)
    ap.add_argument("--ska", type=int, required=True, help="keys [0, ska) on the 16x16, the rest on the 8x8")
    ap.add_argument("--bk", type=int, default=128)
    ap.add_argument("--bk-first", type=int, default=None, help="first key block of each range (attn_flash BK_FIRST)")
    ap.add_argument("--bk-last", type=int, default=None, help="last key block of each range (attn_flash BK_LAST)")
    ap.add_argument("--suffix", default="", help="appended to the expect header's name")
    a = ap.parse_args()
    t = a.header.read_text()
    d = {k: int(v) for k, v in re.findall(r"#define ATTN_(SQ|SK|D|NPASS|NKV|SCALE_IN_Q)\s+(\d+)", t)}
    SQ, SK, D, NP = d["SQ"], d["SK"], d["D"], d["NPASS"]
    bf, bl = a.bk_first or a.bk, a.bk_last or a.bk
    assert d["NKV"] == 1 and 0 < a.ska <= SK   # ska == SK: the 16x16 alone over every key (reference; no 8x8 / merge)
    for n in ((a.ska, SK - a.ska) if a.ska < SK else (a.ska,)):
        assert (n - bf - bl) % a.bk == 0 and n - bf - bl >= 0, f"range {n} does not fit blocks {bf} + k x {a.bk} + {bl}"
    _A.update(
        LQ=GF.c_array(t, "LQ_IN", np.uint8).reshape(NP, SQ, D),
        LQS=GF.c_array(t, "LQ_SCALES", np.uint8).reshape(NP, D // 32, SQ),
        KT=GF.c_array(t, "LKT_IN", np.uint8).reshape(D, SK), KTS=GF.c_array(t, "LKT_SCALES", np.uint8).reshape(D // 32, SK),
        V=GF.c_array(t, "LV_IN", np.uint8).reshape(SK, D), VS=GF.c_array(t, "LV_SCALES", np.uint8).reshape(SK // 32, D),
        fold=0 if d.get("SCALE_IN_Q") else {16: 2, 64: 3, 256: 4}[D], bk=a.bk, bkf=bf, bkl=bl)
    p16 = run_dim(16, [(p, 0, a.ska) for p in range(NP)])
    p8 = run_dim(8, [(p, a.ska, SK) for p in range(NP)]) if a.ska < SK else None
    rep = lambda x: np.repeat(x[:, None], 8, axis=1)
    rows = {k: [] for k in ("O", "O16", "M16", "L16", "O8", "M8", "L8")}
    for p in range(NP):
        (O1, m1, l1) = p16[p]
        if p8 is None:   # reference: no second range; the "8x8" entries and O are the 16x16 partial's
            O2, m2, l2, O = O1, m1, l1, O1
        else:
            (O2, m2, l2) = p8[p]
            O = merge(O1, m1, l1, O2, m2, l2)
        for k, v in (("O", O), ("O16", O1), ("M16", rep(m1)), ("L16", rep(l1)), ("O8", O2), ("M8", rep(m2)),
                     ("L8", rep(l2))):
            rows[k].append(GM.fnv(np.ascontiguousarray(v, np.uint16)))
        print(f"  pass {p}: O {rows['O'][-1]:016x}  O16 {rows['O16'][-1]:016x}  O8 {rows['O8'][-1]:016x}", flush=True)
    tag = a.header.stem.replace("attn_vpu_", "attn_")
    out = a.header.parent / f"{tag}_split{a.ska}{a.suffix}_expect.h"
    body = [f"// GENERATED by gen/golden_dual_attn_split.py {a.header.name} --ska {a.ska} --bk {a.bk} --bk-first {bf} --bk-last {bl}"]
    for k, v in rows.items():
        body.append(f"static const uint64_t EXP_SPLIT_{k}[{NP}] = {{ " + ", ".join(f"0x{h:016x}ULL" for h in v) + " };")
    out.write_text("\n".join(body) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
