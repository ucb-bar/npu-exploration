"""A full single-head attention layer on the MX Gemmini.

Six GEMMs on the mesh, everything else on the Rocket host:

    Q = X @ Wq      K = X @ Wk      V = X @ Wv        (3 GEMMs)
    S = Q @ K^T                                        (1 GEMM)
    P = softmax(S / sqrt(d))                           HOST
    O = P @ V                                          (1 GEMM)
    Y = O @ Wo                                         (1 GEMM)

**Why softmax is on the host, not a compiler op.** The `merlin_iface` grammar has exactly five ops
(tensor / resident_pack / matmul / commit / evict) and a commit epilogue limited to
`["bias_add", "requant", "acc_scale", "relu"]` — there is no softmax and no reduction. That is not a
gap in the grammar, it matches the hardware: the target contract declares one compute unit,
`mx_systolic_mesh`, `ops: [matmul]`, with elementwise only `composed_with: [contraction]`. **There
is no reduction hardware.** Softmax is a row max, a row sum and a divide, so it runs on the scalar
host — merlin's own routing model (contractions to the mesh, everything else to the scalar/vector
lane), and why merlin ships `MF1_softmax_bf16_pt` as a bf16 PyTorch slice rather than an ISA capsule.

**Why every GEMM commits to bf16.** Attention needs host work between GEMMs regardless (the softmax,
the K transpose, the 1/sqrt(d) scale), so each intermediate comes back anyway. Taking bf16 out and
re-quantizing on the host with a chosen block-scale target avoids the requant seam documented in
`planning/npu_exploration_bridge_plan.md` section 13.3b — where the requantizer's full-range output
(peak 448) meets a mesh accumulator that saturates near 2**8. Nothing here needs that path.

    .venv/bin/python app/attention/run_attention.py
    .venv/bin/python app/attention/run_attention.py --seq 32 --d-model 64 --d-head 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "app"))
sys.path.insert(0, str(REPO / "compiler" / "targets" / "mx_gemmini_rocket"))

import backend as mx          # noqa: E402
import mxiface                # noqa: E402
import mxquant as q           # noqa: E402

_gemm_count = 0
_cycles = 0


def mx_gemm(a: np.ndarray, b: np.ndarray, workdir: Path, label: str) -> np.ndarray:
    """One `A[m][k] @ B[k][n]` on the mesh: quantize -> interface MLIR -> command buffer -> spike.

    Returns the bf16 result decoded to float32. Operands are quantized fresh each time, which is
    what keeps every mesh input inside the accumulator's usable range.
    """
    global _gemm_count, _cycles
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"{label}: contraction mismatch {a.shape} @ {b.shape}"
    cb = mxiface.to_command_buffer(
        mxiface.matmul_interface_mlir(m, n, k),
        q.quantize_matmul_operands(a, b))
    res = mx.run_command_buffer(cb, workdir=workdir / label)
    bits = np.array(res["outputs"]["Y0"], dtype=np.uint16)
    bad = int(((bits & 0x7F80) == 0x7F80).sum())
    cyc = res["metrics"].get("cycles", 0)
    _gemm_count += 1
    _cycles += cyc
    print(f"  mesh  {label:<10} {m}x{n}x{k}  cycles {cyc:>5}"
          + (f"   !! {bad} NaN/inf" if bad else ""))
    return q.bf16_bits_to_float(bits)


def host_softmax(s: np.ndarray) -> np.ndarray:
    """Row-wise softmax on the host. Max-subtracted for stability — the scalar core has the
    reduction the mesh does not."""
    z = s - s.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=64, help="sequence length (M)")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--d-head", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workdir", type=Path, default=REPO / "out" / "build" / "attention")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    # ---- the model ------------------------------------------------------------------------------
    Wq, Wk, Wv = (nn.Linear(a.d_model, a.d_head, bias=False) for _ in range(3))
    Wo = nn.Linear(a.d_head, a.d_model, bias=False)
    X = torch.randn(a.seq, a.d_model)
    scale = 1.0 / np.sqrt(a.d_head)
    print(f"model  single-head attention  seq={a.seq} d_model={a.d_model} d_head={a.d_head}")

    with torch.no_grad():
        q_, k_, v_ = Wq(X), Wk(X), Wv(X)
        ref = Wo(torch.softmax(q_ @ k_.T * scale, dim=-1) @ v_).numpy().astype(np.float32)

    # ---- on the device --------------------------------------------------------------------------
    npx = X.numpy().astype(np.float32)
    wq, wk, wv, wo = (w.weight.detach().numpy().T.astype(np.float32) for w in (Wq, Wk, Wv, Wo))

    Q = mx_gemm(npx, wq, a.workdir, "Q=X@Wq")
    K = mx_gemm(npx, wk, a.workdir, "K=X@Wk")
    V = mx_gemm(npx, wv, a.workdir, "V=X@Wv")
    S = mx_gemm(Q, np.ascontiguousarray(K.T), a.workdir, "S=Q@K^T")   # transpose on the host
    P = host_softmax(S * scale)                                        # reduction on the host
    print(f"  host  softmax     rows sum to {P.sum(axis=-1).min():.6f}..{P.sum(axis=-1).max():.6f}")
    O = mx_gemm(P, V, a.workdir, "O=P@V")
    Y = mx_gemm(O, wo, a.workdir, "Y=O@Wo")

    # ---- compare against the PyTorch layer -------------------------------------------------------
    cos = float((Y * ref).sum() / (np.linalg.norm(Y) * np.linalg.norm(ref)))
    print(f"\n{_gemm_count} GEMMs on the mesh, {_cycles} accelerator cycles total")
    print(f"vs PyTorch fp32 attention:  cos={cos:.6f}  "
          f"max|d|={np.abs(Y - ref).max():.4g}  |Y|max={np.abs(ref).max():.4g}")
    ok = np.isfinite(Y).all() and cos > 0.99
    print("verdict  " + ("PASS" if ok else "FAIL") + "  (finite, cos > 0.99)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
