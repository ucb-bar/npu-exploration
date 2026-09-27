#!/usr/bin/env python3
"""Run the bit-exact mesh golden across many cores, by splitting the OUTPUT COLUMNS.

`fp8_matmul_model.tiled_matmul_hwlike` is the reference the kernels are graded against, and it is
slow -- MEASURED at **0.42 MMAC/s** on this machine, single threaded and essentially flat in N:

    M=32 K=2048 N=64       4.2 MMAC    11.22s
    M=32 K=2048 N=256     16.8 MMAC    39.54s
    M=32 K=2048 N=1024    67.1 MMAC   160.86s

One full MLP sub-layer is ~1.1 GMAC, a whole decoder layer ~1.4 G, and all 22 layers plus lm_head
~33 G -- 22 hours of goldens on one core. That is the only thing standing between this repo and a
whole-model kernel, and it is removable: **an output column depends only on its own column of B**
(`llama_layer_hw_plan.md` section 10.2, already load-bearing for the N-chunked kernels), so the
columns are independent and the work is embarrassingly parallel.

THE SPLIT IS ON N ONLY, and that is not an arbitrary choice. The model accumulates along K across
tiles in bf16, so splitting K would change the accumulation order and therefore the VALUE -- the
resulting golden would no longer be the datapath's. Columns carry no such coupling. `selftest()`
checks the identity bit-for-bit rather than trusting the argument, and `mesh_par` runs it on import
of its CLI. M is left whole: it is 32 here, and splitting it would buy nothing.

    from mesh_par import mesh_parallel
    C = mesh_parallel(A_P, A_scales, B_P, B_scales, FMT)      # == G._run_mesh(...), bit for bit

    python3 mesh_par.py                                       # self-test + a throughput report
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Each worker must stay single-threaded: torch would otherwise open its own pool per process and
# 256 processes x 256 threads thrashes into a standstill. Set BEFORE torch is imported anywhere.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
NPU = HERE.parents[2]
ROCC = NPU.parent / "software" / "gemmini-rocc-tests"
for _p in (str(NPU), str(ROCC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: Columns per work item. A multiple of 32 keeps every E8M0 block and every 16-wide mesh tile whole
#: inside one chunk, so a chunk boundary can never fall inside a block. Small enough that 256 cores
#: stay busy on a 2048-wide projection (64 items), large enough that process startup is amortized.
CHUNK_N = 32

_FMT = None       # set in the worker initializer; a Format is not cheap to pickle per item


def _init(fmt):
    global _FMT
    import torch
    torch.set_num_threads(1)
    _FMT = fmt


def _chunk(arg):
    """One work item: columns [n0, n1) of the product. Runs in a worker process."""
    n0, n1, A_P, A_scales, B_P_c, B_scales_c = arg
    import gen_matmul_llama as G
    return n0, n1, G._run_mesh(A_P, A_scales, B_P_c, B_scales_c, _FMT)


def mesh_parallel(A_P: np.ndarray, A_scales: np.ndarray,
                  B_P: np.ndarray, B_scales: np.ndarray, fmt,
                  workers: int | None = None, chunk_n: int = CHUNK_N,
                  label: str = "") -> np.ndarray:
    """`gen_matmul_llama._run_mesh` over a process pool, split on output columns. Bit-identical.

    Falls back to a direct call when the product is small enough that process startup dominates,
    so a caller never has to decide.
    """
    import gen_matmul_llama as G
    M, K = A_P.shape
    N = B_P.shape[1]
    if N % chunk_n or N <= chunk_n:
        return G._run_mesh(A_P, A_scales, B_P, B_scales, fmt)

    workers = workers or min(len(os.sched_getaffinity(0)), N // chunk_n)
    items = [(n0, min(n0 + chunk_n, N), A_P, A_scales,
              np.ascontiguousarray(B_P[:, n0:n0 + chunk_n]),
              np.ascontiguousarray(B_scales[:, n0:n0 + chunk_n]))
             for n0 in range(0, N, chunk_n)]

    import multiprocessing as mp
    t0 = time.time()
    C = np.empty((M, N), dtype=np.float32)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(fmt,)) as pool:
        for n0, n1, c in pool.imap_unordered(_chunk, items, chunksize=1):
            C[:, n0:n1] = c
    if label:
        dt = time.time() - t0
        macs = M * K * N
        print(f"  mesh   {label}  [{M},{K}]x[{K},{N}]  {macs / 1e6:.0f} MMAC  "
              f"{dt:.1f}s on {workers} cores  ({macs / dt / 1e6:.1f} MMAC/s)", flush=True)
    return C


def selftest() -> int:
    """The column-independence claim, checked bit-for-bit rather than argued.

    This is the whole basis of the parallel path: if a chunk boundary perturbed even one element,
    every golden generated here would be subtly wrong and every kernel would 'fail' against it.
    """
    import gen_matmul_llama as G
    import gen_llama_layer as GL
    fmt = GL.FMT
    rng = np.random.default_rng(7)
    M, K, N = 32, 256, 128
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    _, a_s, a_P = G.quantize(A, axis="row", f=fmt)
    _, b_s, b_P = G.quantize(B, axis="col", f=fmt)

    t0 = time.time()
    ref = G._run_mesh(a_P, a_s, b_P, b_s, fmt)
    t_ref = time.time() - t0
    t0 = time.time()
    par = mesh_parallel(a_P, a_s, b_P, b_s, fmt, chunk_n=32)
    t_par = time.time() - t0

    same = np.array_equal(ref.view(np.uint32), par.view(np.uint32))   # BITS, not np.allclose
    print(f"[selftest] M={M} K={K} N={N}  serial {t_ref:.1f}s  parallel {t_par:.1f}s  "
          f"({t_ref / max(t_par, 1e-9):.1f}x)")
    print(f"[selftest] bit-identical to the serial golden: {same}")
    if not same:
        bad = int((ref.view(np.uint32) != par.view(np.uint32)).sum())
        print(f"[selftest] FAILED -- {bad}/{ref.size} elements differ. The column-independence "
              f"assumption is wrong for this model; do NOT use the parallel path.")
        return 1
    print(f"[selftest] PASSED -- splitting on output columns is exact, "
          f"{len(os.sched_getaffinity(0))} cores available")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
