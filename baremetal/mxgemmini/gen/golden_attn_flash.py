#!/usr/bin/env python3
"""Bit-exact Python golden of one src/attn_flash.c pass, independent of Spike (golden_llama_layer_e2e's rtl_exact mesh,
VPU and SPAD_REQUANT ports), in the kernel's order: per key block S = Q K^T (Q's E8M0 minus the 1/sqrt(d) fold),
causal mask on the last block, rmax, m / alpha, exp(S - m) and its row sums, l = l * alpha + lt, P = SPAD_REQUANT(S),
O_j = P V, O = O * alpha + O_j; finally O *= 1/l. Returns O (BF16 bits) and its fnv hash (the kernel's EXP_FLASH_HASH_O).

    ../../../.venv/bin/python3 golden_attn_flash.py ../data/attn_vpu_llama_2h.h --bk 128 \\
        --expect ../data/attn_flash_llama_2h_expect.h                 # FP8: check against the recorded Spike hash
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import golden_llama_layer_e2e as G


def blocks(sk, bk, bk_first=None, bk_last=None):
    bf, bl = bk_first or bk, bk_last or bk
    return [bf] + [bk] * ((sk - bf - bl) // bk) + [bl]


def flash(qc, qs, ktc, kts, vc, vs, bl, fold_shift=0, mask=None, fp4=False):
    """qc [SQ][D] codes, qs [D/32][SQ], ktc [D][SK], kts [D/32][SK], vc [SK][D], vs [SK/32][D]; fp4: E2M1 operands and
    an FP4 SPAD_REQUANT of P (pmax 0). mask [SQ][CH] BF16 bits added to the last CH columns of the last block."""
    dec = G.fp4_decode if fp4 else G.e4m3_decode
    rq = (lambda x: G.spad_requant_fp4(x, 0)) if fp4 else G.spad_requant
    qs_f = (qs.astype(np.int64) - fold_shift).T          # [SQ][D/32]
    off, m, L, O, alpha = 0, None, None, None, None
    for j, b in enumerate(bl):
        S = G.mesh(qc, qs_f, ktc[:, off:off + b], kts[:, off:off + b], decode=dec)
        if mask is not None and j == len(bl) - 1:
            ch = mask.shape[1]
            S[:, b - ch:] = G.v_add(S[:, b - ch:], mask)
        mt = G.v_rmax(S)
        if j == 0:
            m = mt
        else:
            mn = G.v_max(m, mt)
            alpha = G.v_exp(G.v_sub(m, mn))
            m = mn
        P = G.v_exp(G.v_sub(S, m[:, None]))
        lt = G.v_rsum(P)
        L = lt if j == 0 else G.v_add(G.v_mul(L, alpha), lt)
        pc, ps = rq(P)
        Oj = G.mesh(pc, ps, vc[off:off + b], vs[off // 32:(off + b) // 32], decode=dec)
        O = Oj if j == 0 else G.v_add(G.v_mul(O, alpha[:, None]), Oj)
        off += b
    O = G.v_mul(O, G.v_rcp(L)[:, None])
    return O, G.fnv(O)


def c_array(text, name, dtype):
    m = re.search(r"\b" + name + r"\s*((?:\[[^\]]+\])+)\s*(?:__attribute__\(\(aligned\(\d+\)\)\))?\s*=\s*\{", text)
    end = text.index("};", m.end())
    return np.array([int(x, 0) for x in re.findall(r"0x[0-9a-fA-F]+|\d+", text[m.end():end])], dtype=dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("header", type=Path)
    ap.add_argument("--bk", type=int, required=True)
    ap.add_argument("--bk-first", type=int, default=None)
    ap.add_argument("--bk-last", type=int, default=None)
    ap.add_argument("--expect", type=Path, default=None)
    a = ap.parse_args()
    t = a.header.read_text()
    d = {k: int(v) for k, v in re.findall(r"#define ATTN_(SQ|SK|D|CAUSAL|HEADS|SCALE_IN_Q)\s+(\d+)", t)}
    SQ, SK, D = d["SQ"], d["SK"], d["D"]
    fold = 0 if d.get("SCALE_IN_Q") else {16: 2, 64: 3, 256: 4}.get(D, 0)
    assert d.get("SCALE_IN_Q") or D in (16, 64, 256), "a VPU 1/sqrt(d) multiply is not modelled"
    qc = c_array(t, "Q_IN", np.uint8).reshape(SQ, D)
    qs = c_array(t, "Q_SCALES", np.uint8).reshape(D // 32, SQ)
    ktc = c_array(t, "KT_IN", np.uint8).reshape(D, SK)
    kts = c_array(t, "KT_SCALES", np.uint8).reshape(D // 32, SK)
    vc = c_array(t, "V_IN", np.uint8).reshape(SK, D)
    vs = c_array(t, "V_SCALES", np.uint8).reshape(SK // 32, D)
    mask = None
    if d.get("CAUSAL"):
        ch = SQ // d.get("HEADS", 1)
        mask = np.tile(c_array(t, "MASK_BF16", np.uint16).reshape(ch, ch), (d.get("HEADS", 1), 1))
    bl = blocks(SK, a.bk, a.bk_first, a.bk_last)
    O, h = flash(qc, qs, ktc, kts, vc, vs, bl, fold, mask)
    msg = f"O hash {h:016x} (Sq {SQ}, Sk {SK}, d {D}, blocks {bl[0]} + {len(bl) - 2} x {a.bk} + {bl[-1]})"
    if a.expect:
        e = int(re.search(r"EXP_FLASH_HASH_O\s+0x([0-9a-f]+)", a.expect.read_text()).group(1), 16)
        msg += " == Spike" if h == e else f" DIFFERS from Spike {e:016x}"
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
