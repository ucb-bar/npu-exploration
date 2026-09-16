#!/usr/bin/env python3
"""Generate the data for `bareMetalC/llama_attention_full.c` -- a COMPLETE TinyLlama attention
sub-layer: all 32 query heads, all 4 GQA kv heads, the full 2048x2048 q_proj and o_proj.

Why this one is different from `llama_attn.h`. That kernel runs ONE head, so `o_proj` is a partial
sum over 1 of 32 heads and the only available reference is a numpy reimplementation of that slice.
With every head present there is no truncated reduction anywhere, so the result is the layer's REAL
attention output -- and the capture carries `attn_torch`, the tensor TinyLlama's own `self_attn`
module produced on the same tokens. That is a reference this repo could not previously write down.

Data goes out as a BINARY BLOB, not C initializers. The weights alone are 9.1 MiB, which as
`0x%02x, ` text would be ~56 MB of C source -- minutes of GCC and gigabytes of RAM. The blob is
linked in with `objcopy -I binary` and addressed through offsets emitted in the header, which also
makes the build fast enough to iterate on.

    PATH=../../npu-exploration/.venv/bin:$PATH \\
      ../../npu-exploration/.venv/bin/python3 gen_llama_attn_full.py

Needs an --all-heads capture:
    cd ../../npu-exploration && .venv/bin/python3 -m app.capture_llama_layer --all-heads
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

#: gen/ -> mxgemmini/ -> baremetal/ -> the repo root.
HERE = Path(__file__).resolve().parent
NPU = HERE.parents[2]
#: Generated data lands next to the kernels that include it, not next to the generator.
DATA = HERE.parent / "data"
#: gen_matmul_llama.py stays in gemmini-rocc-tests: it generates the matmul_*.h data for the ISA
#: tests themselves, so it belongs with them. We borrow its FORMATS/quantize/mesh-model helpers.
ROCC = NPU.parent / "software" / "gemmini-rocc-tests"
if not (NPU / "app" / "mxq_golden.py").exists():
    raise SystemExit(f"npu-exploration not found at {NPU}")
if not (ROCC / "gen_matmul_llama.py").exists():
    raise SystemExit(f"gen_matmul_llama.py not found in {ROCC}")
sys.path.insert(0, str(NPU))
sys.path.insert(0, str(ROCC))
DATA.mkdir(parents=True, exist_ok=True)

import gen_matmul_llama as G      # noqa: E402  -- quantize(), _requant(), _run_mesh(), bf16_bits()
import gen_llama_layer as GL      # noqa: E402  -- load_capture(), FMT
from app.capture_llama_layer import rmsnorm, rope, softmax_causal  # noqa: E402

FMT = GL.FMT
BLOCK = 32
CAPTURE = NPU / "out" / "layer_capture"


def load_all_heads() -> dict:
    """The newest --all-heads capture. A single-head one cannot be used: the weights are sliced."""
    cands = sorted(CAPTURE.glob("layer*_allheads_*.npz"))
    if not cands:
        raise SystemExit(
            f"no --all-heads capture in {CAPTURE}\nRun it first:\n"
            f"    cd {NPU} && .venv/bin/python3 -m app.capture_llama_layer --all-heads")
    with np.load(cands[-1]) as z:
        d = {k: z[k] for k in z.files}
    d["_path"] = cands[-1]
    return d


# --- the blob ------------------------------------------------------------------------------------

class Blob:
    """Append-only binary image. Every tensor is 64-byte aligned so the DMA never straddles oddly.

    `add` records the offset under a name; the emitted header turns each into
    `#define LLAMA_OFF_<NAME>`, and the C reads it as `LLAMA_BLOB + LLAMA_OFF_<NAME>`.
    """

    def __init__(self) -> None:
        self.buf = bytearray()
        self.off: dict[str, int] = {}
        self.desc: list[tuple[str, int, int, str]] = []

    def add(self, name: str, arr: np.ndarray, dtype) -> int:
        a = np.ascontiguousarray(arr, dtype=dtype)
        pad = (-len(self.buf)) % 64
        self.buf.extend(b"\0" * pad)
        o = len(self.buf)
        self.off[name] = o
        self.buf.extend(a.tobytes())
        self.desc.append((name, o, a.nbytes, str(a.shape)))
        return o


def mesh(A_P, A_s, B_P, B_s):
    return G._run_mesh(A_P, A_s, B_P, B_s, FMT)


def build(cap: dict) -> tuple[Blob, dict]:
    M = int(cap["meta_seq"])
    D = int(cap["meta_d_model"])
    H = int(cap["meta_head_dim"])
    NH = int(cap["meta_n_heads"])
    NKV = int(cap["meta_n_kv_heads"])
    PER = NH // NKV
    QD, KVD = NH * H, NKV * H
    eps = float(cap["meta_rms_eps"])
    cos, sin = cap["rope_cos"], cap["rope_sin"]
    print(f"  shape  M={M} D={D} H={H} heads={NH} kv_heads={NKV} (q head h -> kv head h//{PER})")
    assert cap["Wq"].shape == (D, QD) and cap["Wo"].shape == (QD, D), \
        f"capture is not --all-heads: Wq {cap['Wq'].shape}, Wo {cap['Wo'].shape}"

    b = Blob()
    b.add("H_PRE", GL.bf16_exact(cap["h_pre"], "h_pre"), np.uint16)
    b.add("W_IN_LN", GL.bf16_exact(cap["w_in_ln"], "w_in_ln"), np.uint16)
    b.add("ROPE_COS", cos, np.float32)
    b.add("ROPE_SIN", sin, np.float32)

    # --- host: RMSNorm over the FULL D, then the mesh's A operand ---
    xn = rmsnorm(cap["h_pre"], cap["w_in_ln"], eps)
    xn_codes, xn_scales, xn_P = G.quantize(xn, axis="row", f=FMT)
    b.add("XN_CODES", xn_codes, np.uint8)
    b.add("XN_SCALES", xn_scales.T, np.uint8)             # A-side window: [D/32][M]

    # --- mesh: the three projections, whole. Each is N-chunked in the C, but an output column
    #     depends only on its own column of B, so the unchunked model is the same golden. ---
    proj = {}
    for name, W in (("WQ", cap["Wq"]), ("WK", cap["Wk"]), ("WV", cap["Wv"])):
        t0 = time.time()
        w_codes, w_scales, w_P = G.quantize(W, axis="col", f=FMT)
        b.add(f"{name}_CODES", w_codes, np.uint8)
        b.add(f"{name}_SCALES", w_scales, np.uint8)
        out = mesh(xn_P, xn_scales, w_P, w_scales)
        proj[name] = out
        b.add(f"{name[1]}_OUT", G.bf16_bits(out), np.uint16)
        print(f"  mesh   {name[1]} = Xn @ {name}  {out.shape}  |max|={np.abs(out).max():.4g}"
              f"   ({time.time() - t0:.1f}s)")

    Q, K, V = proj["WQ"], proj["WK"], proj["WV"]

    # --- host: RoPE per head, the K transpose, and V as a B operand ---
    q_codes = np.zeros((NH, M, H), np.uint8)
    q_scales = np.zeros((NH, H // BLOCK, M), np.uint8)
    for h in range(NH):
        qr = rope(Q[:, h * H:(h + 1) * H], cos, sin)
        c, s, _ = G.quantize(qr, axis="row", f=FMT)
        q_codes[h], q_scales[h] = c, s.T
    kt_codes = np.zeros((NKV, H, M), np.uint8)
    kt_scales = np.zeros((NKV, H // BLOCK, M), np.uint8)
    v_codes = np.zeros((NKV, M, H), np.uint8)
    v_scales = np.zeros((NKV, M // BLOCK, H), np.uint8)
    for kv in range(NKV):
        kr = rope(K[:, kv * H:(kv + 1) * H], cos, sin)
        c, s, _ = G.quantize(np.ascontiguousarray(kr.T), axis="col", f=FMT)
        kt_codes[kv], kt_scales[kv] = c, s
        c, s, _ = G.quantize(np.ascontiguousarray(V[:, kv * H:(kv + 1) * H]), axis="col", f=FMT)
        v_codes[kv], v_scales[kv] = c, s
    for n, a in (("Q_CODES", q_codes), ("Q_SCALES", q_scales), ("KT_CODES", kt_codes),
                 ("KT_SCALES", kt_scales), ("V_CODES", v_codes), ("V_SCALES", v_scales)):
        b.add(n, a, np.uint8)

    # --- mesh + host, per head: scores, causal softmax, and O = P @ V requantized to FP8 ---
    s_out = np.zeros((NH, M, M), np.uint16)
    p_codes = np.zeros((NH, M, M), np.uint8)
    p_scales = np.zeros((NH, M // BLOCK, M), np.uint8)
    o_codes = np.zeros((NH, M, H), np.uint8)
    o_scales = np.zeros((NH, M, H // BLOCK), np.uint8)
    O_val = np.zeros((M, QD), np.float32)
    t0 = time.time()
    for h in range(NH):
        kv = h // PER
        # Re-derive the mesh operands from the codes actually emitted, so the golden is what the
        # device is given rather than what the host happened to compute in higher precision.
        qh_P = _p_from(q_codes[h])
        kt_P = _p_from(kt_codes[kv])
        S = mesh(qh_P, q_scales[h].T, kt_P, kt_scales[kv])
        s_out[h] = G.bf16_bits(S)
        P = softmax_causal(S.astype(np.float32) / np.sqrt(np.float32(H)))
        pc, ps, pP = G.quantize(P, axis="row", f=FMT)
        p_codes[h], p_scales[h] = pc, ps.T
        O = mesh(pP, ps, _p_from(v_codes[kv]), v_scales[kv])
        oc, os_, oP = G._requant(O, FMT)
        o_codes[h], o_scales[h] = oc, os_
        O_val[:, h * H:(h + 1) * H] = oP * _e8(os_).repeat(BLOCK, axis=1)
    print(f"  mesh   {NH} heads: S = Q@K^T then O = P@V (requant)   ({time.time() - t0:.1f}s)")
    b.add("S_OUT", s_out, np.uint16)
    for n, a in (("P_CODES", p_codes), ("P_SCALES", p_scales),
                 ("O_CODES", o_codes), ("O_SCALES", o_scales)):
        b.add(n, a, np.uint8)

    # --- mesh: o_proj, accumulated across heads. Y = sum_h O_h @ Wo[64h:64h+64, :], which is one
    #     [M,QD]x[QD,D] matmul -- the device reaches the same value by adding each head's
    #     contribution into the same smem region. ---
    wo_codes, wo_scales, wo_P = G.quantize(cap["Wo"], axis="col", f=FMT)
    b.add("WO_CODES", wo_codes, np.uint8)
    b.add("WO_SCALES", wo_scales, np.uint8)
    o_all = np.concatenate([_p_from(o_codes[h]) for h in range(NH)], axis=1)
    o_all_scales = np.concatenate([o_scales[h] for h in range(NH)], axis=1)   # [M][QD/32]
    t0 = time.time()
    Y = mesh(o_all, o_all_scales, wo_P, wo_scales)
    print(f"  mesh   Y = O @ Wo  {Y.shape}  |max|={np.abs(Y).max():.4g}   ({time.time() - t0:.1f}s)")
    b.add("Y_OUT", G.bf16_bits(Y), np.uint16)

    ref = cap["ref_attn"]
    at = cap["attn_torch"] if "attn_torch" in cap else ref
    b.add("REF_ATTN", G.bf16_bits(ref), np.uint16)
    b.add("ATTN_TORCH", G.bf16_bits(at), np.uint16)
    rel = float(np.linalg.norm(Y - ref) / np.linalg.norm(ref))
    rel_t = float(np.linalg.norm(Y - at) / np.linalg.norm(at))
    print(f"  grade  MX chain vs fp32 reference        : rel_fro = {100 * rel:.4f}%")
    print(f"  grade  MX chain vs the MODEL's own output: rel_fro = {100 * rel_t:.4f}%")

    dims = dict(M=M, D=D, H=H, NH=NH, NKV=NKV, PER=PER, QD=QD, KVD=KVD, eps=eps,
                rel=rel, rel_torch=rel_t, layer=int(cap["meta_layer"]))
    return b, dims


def _e8(codes: np.ndarray) -> np.ndarray:
    from app.mxwire import e8m0_decode
    return e8m0_decode(codes).astype(np.float32)


_E4M3 = None


def _p_from(codes: np.ndarray) -> np.ndarray:
    """Decode E4M3 codes back to their block-normalized values P -- what the mesh multiplies."""
    global _E4M3
    if _E4M3 is None:
        c = np.arange(256, dtype=np.uint8)
        e = ((c >> 3) & 0xF).astype(np.int32)
        m = (c & 0x7).astype(np.int32)
        mag = np.where(e == 0, m * 2.0 ** -9, (1.0 + m / 8.0) * 2.0 ** (e - 7)).astype(np.float32)
        _E4M3 = np.where((c & 0x80) != 0, -mag, mag).astype(np.float32)
    return _E4M3[codes]


def emit(b: Blob, d: dict) -> tuple[Path, Path]:
    bin_path = DATA / "llama_attn_full.bin"
    hdr_path = DATA / "llama_attn_full.h"
    bin_path.write_bytes(bytes(b.buf))

    offs = "\n".join(f"#define LLAMA_OFF_{n:<12s} {o}u" for n, o in b.off.items())
    table = "\n".join(f"//   {n:<12s} @ {o:>9d}  {sz:>9d} B  {sh}" for n, o, sz, sh in b.desc)
    with open(hdr_path, "w") as fh:
        fh.write(f"""// GENERATED by gen_llama_attn_full.py -- do not edit by hand.
//
// A COMPLETE TinyLlama attention sub-layer: all {d['NH']} query heads, all {d['NKV']} GQA kv heads,
// the full {d['D']}x{d['QD']} q_proj and {d['QD']}x{d['D']} o_proj. Layer {d['layer']},
// {d['M']} real tokens of wikitext2, captured by npu-exploration/app/capture_llama_layer.py
// --all-heads. Quantized by MXQuant (block 32, {FMT.name}); mesh goldens from
// fp8_matmul_model.tiled_matmul_hwlike at the datapath's own precision schedule.
//
// NOTHING IS TRUNCATED. Unlike llama_attn.h -- one head, so o_proj is a partial sum over 1 of
// {d['NH']} -- every head is present and the hidden size is full, so the result IS this layer's real
// attention output. ATTN_TORCH is what TinyLlama's own self_attn module produced on these tokens,
// which makes this the first kernel here gradeable against the model rather than against a numpy
// reimplementation of a slice.
//
//   host   xn = rmsnorm(h_pre, w_in_ln)                       -> MX
//   mesh   Q = Xn @ Wq   [{d['M']},{d['D']}]x[{d['D']},{d['QD']}]   ({d['NH']} chunks of {d['H']})
//   mesh   K,V = Xn @ Wk/Wv  [{d['M']},{d['D']}]x[{d['D']},{d['KVD']}]  ({d['NKV']} chunks each)
//   host   RoPE per head; K transposed per kv head
//   mesh   per head  S_h = Q_h @ K_kv^T   [{d['M']},{d['H']}]x[{d['H']},{d['M']}]
//   host   per head  softmax(causal(S_h/sqrt({d['H']})))      -> MX
//   mesh   per head  O_h = P_h @ V_kv     -> FP8 requant into the scratchpad
//   mesh   Y = sum_h O_h @ Wo[{d['H']}h:{d['H']}h+{d['H']}, :]   accumulated in smem across heads
//   host   out = h_pre + Y
//
// The MX chain lands at rel_fro {100 * d['rel']:.4f}% of the fp32 reference, and
// {100 * d['rel_torch']:.4f}% of the model's own bf16 attention output.
//
// DATA LIVES IN llama_attn_full.bin, linked as a binary section -- {len(b.buf) / 1e6:.1f} MB, which as C
// initializers would be ~{6 * len(b.buf) / 1e6:.0f} MB of source. Offsets are into that blob:
//
{table}
#ifndef INCLUDE_LLAMA_ATTN_FULL_H
#define INCLUDE_LLAMA_ATTN_FULL_H

#include <stdint.h>

#define LLAMA_M   {d['M']}      // tokens (also the causal context)
#define LLAMA_D   {d['D']}      // hidden size, FULL
#define LLAMA_H   {d['H']}      // head dim
#define LLAMA_NH  {d['NH']}      // query heads -- ALL of them
#define LLAMA_NKV {d['NKV']}      // GQA kv heads
#define LLAMA_PER {d['PER']}      // query heads per kv head
#define LLAMA_QD  {d['QD']}      // NH * H, = D
#define LLAMA_KVD {d['KVD']}      // NKV * H
#define LLAMA_GD  {d['D'] // BLOCK}      // D / 32
#define LLAMA_GH  {d['H'] // BLOCK}      // H / 32
#define LLAMA_GM  {d['M'] // BLOCK}      // M / 32
#define LLAMA_RMS_EPS {d['eps']:.10g}f

// The blob, linked by objcopy -I binary (see bareMetalC/Makefile).
extern const uint8_t _binary_llama_attn_full_bin_start[];
#define LLAMA_BLOB _binary_llama_attn_full_bin_start
#define LLAMA_AT(off, type) ((const type *) (LLAMA_BLOB + (off)))

{offs}

#endif // INCLUDE_LLAMA_ATTN_FULL_H
""")
    return bin_path, hdr_path


def main() -> int:
    cap = load_all_heads()
    print(f"llama_attn_full  from {cap['_path'].name}")
    b, d = build(cap)
    bp, hp = emit(b, d)
    print(f"  wrote {bp.relative_to(DATA.parent)}  ({bp.stat().st_size / 1e6:.1f} MB)")
    print(f"  wrote {hp.relative_to(DATA.parent)}  ({hp.stat().st_size / 1e3:.1f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
