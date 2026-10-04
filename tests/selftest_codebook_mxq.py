"""mxq.lut == the compiler's codebook rule, bit for bit (LUT_integration.md, link L3).

The reference is the numpy rule the LUT kernels were proved bit-exact on spike with (tests/fixtures/
codebook_numpy.py, the compiler/codebook.py of e0470f6). For each of the four LUT formats, groups G = 0, 1, 2,
and both sides (A groups rows, B groups columns), on random operands and on TinyLlama weights:

    tables  mxq.lut.tables  == build_codebooks      (torch.equal on the values)
    pick    mxq.lut.pick    == assign_indices
    finder  mxq.lut.finder  == finder_indices        (requantized-output element codes)
    values  mxq.lut.values  == codebook_values;  mxq.lut.decode == compiler.wire.DECODERS on every code

on the CPU, and again on the GPU when there is one (same bits). Also the properties of LUT_integration.md
link L6: indices < 16, 16 distinct finder-safe entries per table, n >> G tables, a group with 16 or fewer
distinct codes is reproduced exactly.

    .venv/bin/python tests/selftest_codebook_mxq.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models  # noqa: F401,E402  -- mxq on sys.path
from compiler import formats  # noqa: E402
from compiler.operands import quantize_mx_block32, ROUND_MODE  # noqa: E402
from compiler.wire import DECODERS, encode_requant  # noqa: E402
from mxq import lut  # noqa: E402
from tests.fixtures import codebook_numpy as ref  # noqa: E402

FORMATS = ("fp6_e3m2", "fp6_e2m3", "fp8_e5m2", "fp8_e4m3_quad")
GROUPS = (0, 1, 2)
MAX_ITERS = 50
FAILS: list[str] = []


def check(ok: bool, what: str) -> None:
    if not ok:
        FAILS.append(what)
        print("  FAIL", what)


def operands(seed: int):
    """(name, V) pairs: random shapes and scales, a heavy-tailed one, and a few-distinct-codes one."""
    g = torch.Generator().manual_seed(seed)
    yield "randn 64x128", torch.randn(64, 128, generator=g)
    yield "laplace 96x64", torch.distributions.Laplace(0, 1).sample((96, 64)) * 3
    yield "outliers 64x64", torch.randn(64, 64, generator=g) * torch.where(torch.rand(64, 64, generator=g) < 0.01,
                                                                          torch.tensor(50.0), torch.tensor(1.0))
    yield "few codes 64x32", torch.randint(-2, 3, (64, 32), generator=g).float()


def llama_tensors():
    """Up to three TinyLlama weight slices, if the checkpoint is cached; [] otherwise (said so)."""
    try:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32,
                                                 local_files_only=True)
    except Exception as e:                                    # noqa: BLE001
        print(f"  (TinyLlama not cached: {type(e).__name__}; random operands only)")
        return []
    layer = m.model.layers[3]
    return [("q_proj", layer.self_attn.q_proj.weight[:128, :256].detach().clone()),
            ("gate_proj", layer.mlp.gate_proj.weight[:128, :256].detach().clone()),
            ("down_proj", layer.mlp.down_proj.weight[:128, :256].detach().clone())]


def one(name: str, V: torch.Tensor, dtype: str, g: int, device: str) -> None:
    f = formats.get(dtype, where="selftest")
    mfmt = f.mxq
    settings = ref.Settings(group=g, max_iters=MAX_ITERS)
    for side, axis in (("a", "row"), ("b", "col")):
        P = quantize_mx_block32(V, fmt=mfmt, axis=axis, round_mode=ROUND_MODE).P.numpy().astype(np.float32)
        span = P.shape[0] if axis == "row" else P.shape[1]
        if span % (1 << g):
            continue
        books = ref.build_codebooks(P, axis=axis, fmt=f, lut=settings)
        idx = ref.assign_indices(P, books, axis=axis, g=g)

        Pk = torch.from_numpy(P.T.copy() if axis == "row" else P).to(device)   # K×n: A's rows become columns
        T = lut.tables(Pk, mfmt, group=g, max_iters=MAX_ITERS)
        I = lut.pick(Pk, T, group=g)
        I_np = I.cpu().numpy().T if axis == "row" else I.cpu().numpy()
        tag = f"{dtype} G={g} {side} {name} [{device}]"
        check(np.array_equal(T.cpu().numpy(), books), f"tables {tag}")
        check(np.array_equal(I_np.astype(np.uint8), idx), f"pick {tag}")

        # L6 properties
        Tc = T.cpu()
        check(Tc.shape == (Pk.shape[1] >> g, 16), f"table count {tag}")
        check(bool((I >= 0).all() and (I < 16).all()), f"index range {tag}")
        safe = set(lut.values(mfmt).tolist())
        check(all(len(set(r)) == 16 and set(r) <= safe for r in Tc.tolist()), f"16 distinct safe entries {tag}")
        P_lut = lut.lookup(I, T, group=g).cpu()
        Pkc = Pk.cpu()
        for j in range(Pkc.shape[1] >> g):
            cols = slice(j << g, (j + 1) << g)
            vals = set(Pkc[:, cols].reshape(-1).tolist())
            if len(vals) <= 16 and vals <= safe:
                check(torch.equal(P_lut[:, cols], Pkc[:, cols]), f"lossless group {j} {tag}")

    # the finder, on requantized-output element codes of a C tile (rows grouped, as the requantizer does)
    C = (V[:, : V.shape[1] // 32 * 32] * 0.37).numpy().astype(np.float32)
    if C.shape[1] == 0 or C.shape[0] % (1 << g):
        return
    M, N = C.shape
    blocks = C.reshape(M, N // 32, 32)
    mx = np.abs(blocks).max(axis=2)
    with np.errstate(divide="ignore"):
        X = np.exp2(np.floor(np.log2(np.where(mx == 0, 1, mx))))
    elem = encode_requant((blocks / X[:, :, None]).reshape(M, N), dtype=f.name)
    Pq = quantize_mx_block32(torch.from_numpy(C), fmt=mfmt, axis="row", round_mode=ROUND_MODE).P.numpy()
    books = ref.build_codebooks(Pq.astype(np.float32), axis="row", fmt=f, lut=settings)
    want = ref.finder_indices(elem, books, fmt=f, axis="row", g=g)
    got = lut.finder(torch.from_numpy(elem.T.copy()).to(device), torch.from_numpy(books).to(device), mfmt, group=g)
    check(np.array_equal(got.cpu().numpy().T.astype(np.uint8), want), f"finder {dtype} G={g} {name} [{device}]")


def main() -> int:
    print("selftest_codebook_mxq: mxq.lut == compiler codebook rule")
    for dtype in FORMATS:
        f = formats.get(dtype, where="selftest")
        codes = np.arange(1 << f.entry_bits, dtype=np.uint8)
        got = lut.decode(torch.from_numpy(codes.astype(np.int64)), f.mxq).numpy()
        want = DECODERS[f.name](codes)
        check(np.array_equal(got, want, equal_nan=True), f"decode {dtype}")
        check(np.array_equal(lut.values(f.mxq).numpy(), ref.codebook_values(f)), f"values {dtype}")

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    cases = [(n, V) for seed in (0, 1) for n, V in operands(seed)] + llama_tensors()
    for device in devices:
        for dtype in FORMATS:
            for g in GROUPS:
                for name, V in cases:
                    one(name, V, dtype, g, device)
        print(f"  {device}: {len(FORMATS)} formats x G {GROUPS} x {len(cases)} operands")
    print("FAIL" if FAILS else "PASS", f"({len(FAILS)} failures)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
