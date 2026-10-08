#!/usr/bin/env python3
"""Data for src/attn_vpu.c: one dense attention block, softmax on the Gemmini VPU.

    S = Q @ K^T            mesh, BF16 into the scratchpad (golden: the bit-exact mesh model)
    P = softmax(S / sqrt(d))   VPU (bit-exact reference: Spike / include/vpu_ref.h)
    O = requant(P) @ V     SPAD_REQUANT + mesh

Random Q, K, V ~ N(0, 1), MX-quantized (E4M3, E8M0 per 32 along d for Q/K, along Sk for V) with the
gen_matmul_llama quantizer. Emits the operands in the layouts the kernel mvins, S_GOLDEN (exact) and
two fp32 references for grading O:  O_REF_S (softmax over the mesh's own S, then @ dequantized V --
the error of VPU softmax + P requant + PV) and O_REF_F (fp64 attention on the dequantized inputs).

    ../../../.venv/bin/python3 gen_attn_vpu.py                         # Sq 32, Sk 64, d 64
    ../../../.venv/bin/python3 gen_attn_vpu.py --sq 64 --sk 256 --d 128 --tag _fa
    ../../../.venv/bin/python3 gen_attn_vpu.py --capture ../../../out/layer_capture/attn_qkv_layer5_h0_s2112.npz \
        --sq 64 --sk 2048 --tag _llama --no-s-golden          # real TinyLlama head (capture_attn_qkv.py)

--capture: Q = the capture's LAST sq tokens, K/V = its first sk tokens (sq + sk <= seq), d = the head dim. Every key
precedes every query, so no causal mask is needed: 64 new tokens attending to a 2048-token KV cache.
--capture a.npz b.npz ...: GQA head packing. Captures of query heads sharing one kv head (same K/V, checked) stack
their queries as extra rows: Sq = sq x heads (rows h*sq.. are head h). Softmax is per row, so this is exact.

    ../../../.venv/bin/python3 gen_attn_vpu.py --capture ../../../out/layer_capture/attn_qkv_layer5_h{0,1}_s2112.npz \
        --sq 64 --sk 2048 --tag _llama_2h --no-s-golden
--layer TEMPLATE: a whole layer's attention as GQA-packed passes. TEMPLATE has {h} for the query head; heads that
share a kv head are packed --pack at a time (consecutive heads of a group), each pass Sq = sq x pack. One header
holds every pass's Q, each kv head's K/V once, the fp64-attention O reference per pass, and the pass -> kv map.

    ../../../.venv/bin/python3 gen_attn_vpu.py --layer ../../../out/layer_capture/attn_qkv_layer5_h{h}_s2112.npz \
        --heads 0-31 --pack 2 --sq 64 --sk 2048 --tag _layer5_p2
--scale-in-q: Q is pre-scaled by 1/sqrt(d) (the scale folded into Wq), so a d that is not a power of 4 (Llama-7B's 128)
needs neither an E8M0 fold nor a VPU multiply.

    ../../../.venv/bin/python3 gen_attn_vpu.py --sq 64 --sk 2048 --d 128 --scale-in-q --tag _llama7b --no-s-golden
--causal: Q = the last sq of the sk key tokens (chunked prefill: the chunk attends to the cache and to itself), with
the causal mask (key position > query position -> -inf) on the last sq keys; emits MASK_BF16 [sq][sq] (0 / -inf).

    ../../../.venv/bin/python3 gen_attn_vpu.py --capture ../../../out/layer_capture/attn_qkv_layer5_h0_s2112.npz \
        --sq 64 --sk 2048 --tag _llama_causal --no-s-golden --causal
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import gen_llama_layer as L

G, FMT = L.G, L.FMT
DATA = Path(__file__).resolve().parent.parent / "data"


def deq(P: np.ndarray, scales: np.ndarray, axis: str) -> np.ndarray:
    """Code values * 2^(E8M0 - 127), blocks of 32 along K (axis row: [M][K/32], col: [K/32][N])."""
    s = np.exp2(scales.astype(np.float64) - 127.0)
    s = np.repeat(s, 32, axis=1 if axis == "row" else 0)
    return P.astype(np.float64) * s


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sq", type=int, default=32)
    ap.add_argument("--sk", type=int, default=64)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--capture", type=Path, nargs="+", default=None,
                    help="npz from capture_attn_qkv.py (real Q/K/V); several = query heads of one kv head, packed")
    ap.add_argument("--no-s-golden", action="store_true", help="omit S_GOLDEN (dense-kernel check only)")
    ap.add_argument("--causal", action="store_true", help="queries are the last sq keys; causal mask on them")
    ap.add_argument("--scale-in-q", action="store_true",
                    help="Q pre-scaled by 1/sqrt(d) (as with the scale folded into Wq): for d not a power of 4")
    ap.add_argument("--layer", default=None, help="capture path template with {h}: whole-layer multi-pass header")
    ap.add_argument("--heads", default="0-31", help="--layer: query heads, 'a-b' or comma list")
    ap.add_argument("--pack", type=int, default=2, help="--layer: query heads per pass (sharing a kv head)")
    a = ap.parse_args()
    if a.layer is not None:
        return main_layer(a)
    Sq, Sk, d = a.sq, a.sk, a.d
    src = f"random N(0,1), seed {a.seed}"
    H = 1
    if a.capture is not None:
        zs = [np.load(c) for c in a.capture]
        z, H = zs[0], len(zs)
        assert not (a.causal and H > 1), "causal head packing needs a per-head mask (not implemented)"
        for o in zs[1:]:
            assert int(o["kv_head"]) == int(z["kv_head"]) and np.array_equal(o["K"], z["K"]) and np.array_equal(o["V"], z["V"]), \
                "packed heads must share one kv head"
        T = int(z["seq"]); d = int(z["Q"].shape[1])
        q0 = Sk - Sq if a.causal else T - Sq
        assert a.causal and Sk <= T or Sq + Sk <= T, f"capture has {T} tokens, need sq + sk = {Sq + Sk}"
        heads = ",".join(str(int(o["head"])) for o in zs)
        src = (f"TinyLlama layer {int(z['layer'])} head{'s' if H > 1 else ''} {heads} (kv head {int(z['kv_head'])}), "
               f"queries = tokens {q0}..{q0 + Sq - 1}, keys/values = tokens 0..{Sk - 1}" + (", causal" if a.causal else ""))
    # tile shape only; the dense kernel additionally needs Sq*Sk/32 <= 2048 (one SPAD_REQUANT), flash per block
    assert Sq % 16 == 0 and Sk % 32 == 0 and d % 32 == 0
    if a.capture is not None:
        Q = np.ascontiguousarray(np.vstack([o["Q"][q0:q0 + Sq] for o in zs]))   # head h -> rows h*Sq ..
        K = np.ascontiguousarray(z["K"][:Sk]); V = np.ascontiguousarray(z["V"][:Sk])
        Sq *= H
    else:
        rng = np.random.default_rng(a.seed)
        Q = rng.standard_normal((Sq, d)).astype(np.float32)
        K = rng.standard_normal((Sk, d)).astype(np.float32)
        V = rng.standard_normal((Sk, d)).astype(np.float32)

    if a.scale_in_q:
        Q = (Q / np.sqrt(d)).astype(np.float32)
        src += ", Q pre-scaled by 1/sqrt(d)"
    q_codes, q_scales, q_P = G.quantize(Q, axis="row", f=FMT)                         # A [Sq][d], [Sq][d/32]
    kt_codes, kt_scales, kt_P = G.quantize(np.ascontiguousarray(K.T), axis="col", f=FMT)  # B [d][Sk], [d/32][Sk]
    v_codes, v_scales, v_P = G.quantize(V, axis="col", f=FMT)                         # B [Sk][d], [Sk/32][d]
    S = L.mesh(q_P, q_scales, kt_P, kt_scales)                                        # exact BF16 values
    sc = 1.0 if a.scale_in_q else 1.0 / np.sqrt(d)
    # causal: query i sits at position Sk - Sq + i and sees keys <= its position (mask adds -inf beyond)
    neg = np.zeros((Sq, Sk))
    if a.causal:
        assert Sq <= Sk
        neg[np.arange(Sk)[None, :] > (Sk - Sq + np.arange(Sq))[:, None]] = -np.inf
    Vd = deq(v_P, v_scales, "col")
    O_ref_s = softmax(S.astype(np.float64) * sc + neg) @ Vd
    Qd, Kd = deq(q_P, q_scales, "row"), deq(kt_P, kt_scales, "col").T
    O_ref_f = softmax((Qd @ Kd.T) * sc + neg) @ Vd
    # the same MX pipeline with an exact softmax: fp64 softmax of the mesh S -> MX (rows) -> mesh @ V.
    # The kernel's distance from this is the VPU softmax's own error; from O_REF_S it adds P's E4M3.
    p_codes, p_scales, p_P = G.quantize(softmax(S.astype(np.float64) * sc + neg).astype(np.float32), axis="row", f=FMT)
    O_ref_q = L.mesh(p_P, p_scales, v_P, v_scales)
    print(f"  Sq={Sq} Sk={Sk} d={d}: |S| max {np.abs(S).max():.3g}, rel(O_ref_s, O_ref_f) = "
          f"{np.linalg.norm(O_ref_s - O_ref_f) / np.linalg.norm(O_ref_f):.3e}, "
          f"rel(O_ref_q, O_ref_s) = {np.linalg.norm(O_ref_q - O_ref_s) / np.linalg.norm(O_ref_s):.3e}")

    r, b = G._rows, G.bf16_bits
    f32 = lambda x: ",\n".join("    { " + ", ".join("0x%08x" % int(v) for v in row) + " }"
                               for row in np.ascontiguousarray(x, dtype=np.float32).view(np.uint32))
    path = DATA / f"attn_vpu{a.tag}.h"
    guard = f"INCLUDE_ATTN_VPU{a.tag.upper()}_H"
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by gen/gen_attn_vpu.py --sq {Sq} --sk {Sk} --d {d} -- do not edit.
// Attention data for src/attn_vpu.c / attn_flash.c. Q/K/V: {src}. See the generator for the layouts.
#ifndef {guard}
#define {guard}

#include <stdint.h>

#define ATTN_SQ {Sq}
#define ATTN_SK {Sk}
#define ATTN_D  {d}
#define ATTN_CAUSAL {int(a.causal)}
#define ATTN_HEADS  {H}   // packed query heads sharing K/V (rows h*ATTN_SQ/ATTN_HEADS ..)
#define ATTN_SCALE_IN_Q {int(a.scale_in_q)}   // 1: Q already carries 1/sqrt(d); no scaling in the kernel

static const uint8_t Q_IN[ATTN_SQ][ATTN_D] __attribute__((aligned(64))) = {{
{r(q_codes, 2)}
}};
static const uint8_t Q_SCALES[ATTN_D / 32][ATTN_SQ] __attribute__((aligned(64))) = {{
{r(q_scales.T, 2)}
}};
static const uint8_t KT_IN[ATTN_D][ATTN_SK] __attribute__((aligned(64))) = {{
{r(kt_codes, 2)}
}};
static const uint8_t KT_SCALES[ATTN_D / 32][ATTN_SK] __attribute__((aligned(64))) = {{
{r(kt_scales, 2)}
}};
static const uint8_t V_IN[ATTN_SK][ATTN_D] __attribute__((aligned(64))) = {{
{r(v_codes, 2)}
}};
static const uint8_t V_SCALES[ATTN_SK / 32][ATTN_D] __attribute__((aligned(64))) = {{
{r(v_scales, 2)}
}};
{"" if not a.causal else "// causal mask on the last Sq keys: 0 where key <= query position, -inf (0xff80) beyond" + chr(10) +
 "static const uint16_t MASK_BF16[ATTN_SQ][ATTN_SQ] __attribute__((aligned(64))) = {" + chr(10) +
 r(np.where(np.arange(Sq)[None, :] > np.arange(Sq)[:, None], 0xFF80, 0).astype(np.uint16), 4) + chr(10) + "};"}
{"" if a.no_s_golden else "// S = Q @ K^T from the bit-exact mesh model (before 1/sqrt(d))" + chr(10) +
 "static const uint16_t S_GOLDEN[ATTN_SQ][ATTN_SK] __attribute__((aligned(64))) = {" + chr(10) + r(b(S), 4) + chr(10) + "};"}
// fp32 references for O: softmax over the mesh's own S, and fp64 attention on the dequantized inputs
static const uint32_t O_REF_S_F32[ATTN_SQ][ATTN_D] = {{
{f32(O_ref_s)}
}};
static const uint32_t O_REF_F_F32[ATTN_SQ][ATTN_D] = {{
{f32(O_ref_f)}
}};
// exact softmax, then the kernel's own MX steps (P -> E4M3 rows, mesh @ V): the floor for the VPU kernel
static const uint32_t O_REF_Q_F32[ATTN_SQ][ATTN_D] = {{
{f32(O_ref_q)}
}};

#endif
""")
    print(f"  wrote {path}")
    return 0


def main_layer(a) -> int:
    """Multi-pass header for a whole layer (non-causal): passes of a.pack GQA heads, K/V once per kv head."""
    assert not a.causal, "--layer: non-causal only"
    heads = (list(range(int(a.heads.split("-")[0]), int(a.heads.split("-")[1]) + 1)) if "-" in a.heads
             else [int(x) for x in a.heads.split(",")])
    zs = {h: np.load(a.layer.format(h=h)) for h in heads}
    kv_of = {h: int(zs[h]["kv_head"]) for h in heads}
    kvs = sorted(set(kv_of.values()))
    passes = []                                    # (kv head, [query heads])
    for kv in kvs:
        hs = [h for h in heads if kv_of[h] == kv]
        assert len(hs) % a.pack == 0, f"kv head {kv}: {len(hs)} heads not a multiple of --pack {a.pack}"
        passes += [(kv, hs[i:i + a.pack]) for i in range(0, len(hs), a.pack)]
    z0 = zs[heads[0]]
    T = int(z0["seq"]); d = int(z0["Q"].shape[1]); sq, Sk = a.sq, a.sk
    assert sq + Sk <= T and sq % 16 == 0 and Sk % 32 == 0 and d % 32 == 0
    q0 = T - sq
    Sq = sq * a.pack
    sc = 1.0 / np.sqrt(d)
    kvi = {kv: i for i, kv in enumerate(kvs)}
    kt_c, kt_s, v_c, v_s, kd, vd = {}, {}, {}, {}, {}, {}
    for kv in kvs:
        src = next(zs[h] for h in heads if kv_of[h] == kv)
        for h in heads:
            if kv_of[h] == kv:
                assert np.array_equal(zs[h]["K"], src["K"]) and np.array_equal(zs[h]["V"], src["V"]), "kv mismatch"
        K = np.ascontiguousarray(src["K"][:Sk]); V = np.ascontiguousarray(src["V"][:Sk])
        kt_c[kv], kt_s[kv], kt_P = G.quantize(np.ascontiguousarray(K.T), axis="col", f=FMT)
        v_c[kv], v_s[kv], v_P = G.quantize(V, axis="col", f=FMT)
        kd[kv] = deq(kt_P, kt_s[kv], "col").T; vd[kv] = deq(v_P, v_s[kv], "col")
    q_c, q_s, o_ref = [], [], []
    for kv, hs in passes:
        Q = np.ascontiguousarray(np.vstack([zs[h]["Q"][q0:q0 + sq] for h in hs]))
        qc, qs, qP = G.quantize(Q, axis="row", f=FMT)
        q_c.append(qc); q_s.append(qs)
        o_ref.append(softmax((deq(qP, qs, "row") @ kd[kv].T) * sc) @ vd[kv])
    r = G._rows
    nest = lambda arrs, w: ",\n".join("  {\n" + r(x, w) + "\n  }" for x in arrs)
    f32 = lambda x: ",\n".join("    { " + ", ".join("0x%08x" % int(v) for v in row) + " }"
                               for row in np.ascontiguousarray(x, dtype=np.float32).view(np.uint32))
    nestf = lambda arrs: ",\n".join("  {\n" + f32(x) + "\n  }" for x in arrs)
    P = len(passes); NKV = len(kvs)
    path = DATA / f"attn_vpu{a.tag}.h"
    guard = f"INCLUDE_ATTN_VPU{a.tag.upper()}_H"
    with open(path, "w") as fh:
        fh.write(f"""// GENERATED by gen/gen_attn_vpu.py --layer --heads {a.heads} --pack {a.pack} --sq {sq} --sk {Sk} -- do not edit.
// Whole-layer attention for src/attn_flash_layer.c: TinyLlama layer {int(z0['layer'])}, {len(heads)} query heads over {NKV} kv
// heads, packed {a.pack} per pass ({P} passes). Queries = tokens {q0}..{T - 1}, keys/values = tokens 0..{Sk - 1}, non-causal.
#ifndef {guard}
#define {guard}

#include <stdint.h>

#define ATTN_SQ {Sq}
#define ATTN_SK {Sk}
#define ATTN_D  {d}
#define ATTN_CAUSAL 0
#define ATTN_HEADS  {a.pack}
#define ATTN_NPASS  {P}
#define ATTN_NKV    {NKV}

static const uint8_t LPASS_KV[ATTN_NPASS] = {{ {", ".join(str(kvi[kv]) for kv, _ in passes)} }};
static const uint8_t LPASS_HEAD0[ATTN_NPASS] = {{ {", ".join(str(hs[0]) for _, hs in passes)} }};
static const uint8_t LQ_IN[ATTN_NPASS][ATTN_SQ][ATTN_D] __attribute__((aligned(64))) = {{
{nest(q_c, 2)}
}};
static const uint8_t LQ_SCALES[ATTN_NPASS][ATTN_D / 32][ATTN_SQ] __attribute__((aligned(64))) = {{
{nest([x.T for x in q_s], 2)}
}};
static const uint8_t LKT_IN[ATTN_NKV][ATTN_D][ATTN_SK] __attribute__((aligned(64))) = {{
{nest([kt_c[kv] for kv in kvs], 2)}
}};
static const uint8_t LKT_SCALES[ATTN_NKV][ATTN_D / 32][ATTN_SK] __attribute__((aligned(64))) = {{
{nest([kt_s[kv] for kv in kvs], 2)}
}};
static const uint8_t LV_IN[ATTN_NKV][ATTN_SK][ATTN_D] __attribute__((aligned(64))) = {{
{nest([v_c[kv] for kv in kvs], 2)}
}};
static const uint8_t LV_SCALES[ATTN_NKV][ATTN_SK / 32][ATTN_D] __attribute__((aligned(64))) = {{
{nest([v_s[kv] for kv in kvs], 2)}
}};
// fp64 attention on the dequantized inputs, per pass (rows h*{sq}.. = the pass's h-th head)
static const uint32_t LO_REF_F_F32[ATTN_NPASS][ATTN_SQ][ATTN_D] = {{
{nestf(o_ref)}
}};

#endif
""")
    print(f"  {len(heads)} heads, {NKV} kv heads, {P} passes of Sq {Sq}: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
