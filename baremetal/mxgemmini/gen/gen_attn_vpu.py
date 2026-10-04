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
    ap.add_argument("--capture", type=Path, default=None, help="npz from capture_attn_qkv.py (real Q/K/V)")
    ap.add_argument("--no-s-golden", action="store_true", help="omit S_GOLDEN (dense-kernel check only)")
    ap.add_argument("--causal", action="store_true", help="queries are the last sq keys; causal mask on them")
    a = ap.parse_args()
    Sq, Sk, d = a.sq, a.sk, a.d
    src = f"random N(0,1), seed {a.seed}"
    if a.capture is not None:
        z = np.load(a.capture)
        T = int(z["seq"]); d = int(z["Q"].shape[1])
        q0 = Sk - Sq if a.causal else T - Sq
        assert a.causal and Sk <= T or Sq + Sk <= T, f"capture has {T} tokens, need sq + sk = {Sq + Sk}"
        src = (f"TinyLlama layer {int(z['layer'])} head {int(z['head'])} (kv head {int(z['kv_head'])}), "
               f"queries = tokens {q0}..{q0 + Sq - 1}, keys/values = tokens 0..{Sk - 1}" + (", causal" if a.causal else ""))
    # tile shape only; the dense kernel additionally needs Sq*Sk/32 <= 2048 (one SPAD_REQUANT), flash per block
    assert Sq % 16 == 0 and Sk % 32 == 0 and d % 32 == 0
    if a.capture is not None:
        Q = np.ascontiguousarray(z["Q"][q0:q0 + Sq]); K = np.ascontiguousarray(z["K"][:Sk]); V = np.ascontiguousarray(z["V"][:Sk])
    else:
        rng = np.random.default_rng(a.seed)
        Q = rng.standard_normal((Sq, d)).astype(np.float32)
        K = rng.standard_normal((Sk, d)).astype(np.float32)
        V = rng.standard_normal((Sk, d)).astype(np.float32)

    q_codes, q_scales, q_P = G.quantize(Q, axis="row", f=FMT)                         # A [Sq][d], [Sq][d/32]
    kt_codes, kt_scales, kt_P = G.quantize(np.ascontiguousarray(K.T), axis="col", f=FMT)  # B [d][Sk], [d/32][Sk]
    v_codes, v_scales, v_P = G.quantize(V, axis="col", f=FMT)                         # B [Sk][d], [Sk/32][d]
    S = L.mesh(q_P, q_scales, kt_P, kt_scales)                                        # exact BF16 values
    sc = 1.0 / np.sqrt(d)
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


if __name__ == "__main__":
    raise SystemExit(main())
