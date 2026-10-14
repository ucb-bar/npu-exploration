"""mxq's attention core (mxq.nn.attend, the function the perplexity path runs) against spike, bit for bit.

The perplexity path computes S = Q·Kᵀ, softmax and O = P·V inside ``mxq.nn.attend``, with a KV group's heads
side by side, masked key blocks skipped and the reducer compiled. Every other check holds those against mxq
itself; this one holds them against the chip. The kernel is the attention core alone, one head on the mesh:

    S = Q @ Kᵀ  (mesh)  ->  P = softmax(S·scale, causal)  (host, mx_host.h)  ->  O = P @ V  (mesh)

Q is the kernel's input, Kᵀ and V its two resident operands. Two heads share K and V, so ``attend`` runs them as
one grouped-query KV group while spike runs each head as its own kernel.

The claims:
  1. spike == the mxquant kernel model on every run (the existing verdict, here as a precondition);
  2. DATAPATH: attend with the host's softmax (kernels.host_ops.softmax, the Python twin of mx_softmax_rows)
     equals spike's O bit for bit: both heads, chunks None/16/48 (small chunks skip key blocks), eager and,
     on a GPU, compiled. Everything attend owns -- quantizers, reducers, head batching, key skipping, the
     causal mask -- is then the chip's;
  3. AS SHIPPED: attend as the perplexity path runs it (torch.softmax) against spike: reported, not required.
     Its only difference from 2 is the softmax, so any mismatch is the softmax's last bits flipping a code of P.

Needs the toolchain (scripts/env.sh) and mxq; skipped cleanly without them. Exit 0 on pass, 1 on failure.

    source scripts/env.sh && .venv/bin/python tests/selftest_attend_spike.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FAILURES: list[str] = []
T, D, SEED = 128, 64, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def same(a, b) -> tuple[int, int]:
    """(identical elements, total), comparing float32 bit patterns."""
    import numpy as np
    a = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    b = np.ascontiguousarray(b, dtype=np.float32).view(np.uint32)
    return int((a == b).sum()), a.size


def main() -> int:
    import models
    if not models.paths():
        print(f"SKIP: {models.mxq_missing()}")
        return 0
    import shutil
    if shutil.which("spike") is None:
        print("SKIP: spike not on PATH -- source scripts/env.sh")
        return 0
    import numpy as np
    import torch
    from config import scheme as _scheme
    from config.recipe import load_hardware, load_run
    from grade.pipeline import run
    from grade.telemetry import Telemetry
    from kernels import host_ops
    from kernels.spec import HostStage, KernelSpec, Stage
    import mxq.nn._attention as A

    recipe, run_recipe = load_hardware("baseline"), load_run("default")
    scale = 1.0 / float(np.sqrt(D))
    g = torch.Generator().manual_seed(SEED)
    q = torch.randn(2, T, D, generator=g)                          # two query heads ...
    q[1, 7] *= 30                                                  # ... one with a large row: wide scores
    k = torch.randn(T, D, generator=g)                             # ... on one KV head
    v = torch.randn(T, D, generator=g)

    # --- spike: one kernel per head ---------------------------------------------------------------
    hw = []
    for h in range(2):
        spec = KernelSpec(name=f"attention_core_h{h}", x=q[h].clone(), stages=[
            Stage("S", weight=k.t().contiguous(), lhs="x"),
            HostStage("P", op="softmax", src="S", params={"scale": scale, "causal": True},
                      note="causal mask + softmax on the host"),
            Stage("O", weight=v.clone(), lhs="P"),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            res = run(spec, recipe=recipe, run_recipe=run_recipe, models=("mxquant", "spike"),
                      workdir=Path(tmp) / "work", results_dir=Path(tmp) / "results", telemetry=Telemetry())
            c = res["metrics"].get("correctness_vs_mxquant") or {}
            run_dir = Path(res["run_dir"])
            lowering = json.loads((run_dir / "config.json").read_text())["run_config"].get("lowering")
            hw.append(np.load(run_dir / "hardware_output.npy"))
        check(f"1. head {h}: spike == the mxquant kernel model", bool(c.get("bit_exact")),
              f"{c.get('n_mismatch')} of {c.get('total_elements')} differ, lowering {lowering}")
    hw = np.stack(hw)                                              # 2 x T x D, bf16 values as float32

    # --- attend ------------------------------------------------------------------------------------
    def twin(S, precision):
        """The host's softmax on attend's scaled, masked scores (the mask is already in S: scale 1, no causal)."""
        if precision is not None:
            raise ValueError("the host softmax is fp32")
        return torch.from_numpy(host_ops.softmax(S.detach().cpu().numpy())).to(S.device)

    lo = torch.finfo(torch.float32).min
    mask = torch.full((T, T), lo).triu(1)[None, None]
    arms = [("eager", "cpu", False)] + ([("compiled", "cuda", True)] if torch.cuda.is_available() else [])
    for arm, dev, compiled in arms:
        sch = _scheme.scheme(recipe, run_recipe, compiled=compiled)
        qq, kk, vv, mm = q[None].to(dev), k[None, None].to(dev), v[None, None].to(dev), mask.to(dev)
        for chunk in (None, 16, 48):
            torch_softmax = A.softmax
            A.softmax = twin
            try:
                o = A.attend(qq, kk, vv, mm, scale, sch, sch, chunk=chunk)       # 1 x T x 2 x D
            finally:
                A.softmax = torch_softmax
            o = o[0].transpose(0, 1).to(torch.bfloat16).float().cpu().numpy()
            n, tot = same(o, hw)
            check(f"2. {arm}, chunk {chunk}: attend with the host softmax == spike", n == tot, f"{n}/{tot}")
        o = A.attend(qq, kk, vv, mm, scale, sch, sch)[0].transpose(0, 1).to(torch.bfloat16).float().cpu().numpy()
        n, tot = same(o, hw)
        print(f"  INFO  3. {arm}: attend as shipped (torch.softmax) == spike on {n}/{tot} "
              f"({tot - n} differ, max |d| {float(np.abs(o - hw).max()):.3g})")

    print("\nALL CHECKS PASSED -- mxq.nn.attend is the chip's attention core." if not FAILURES
          else f"\n{len(FAILURES)} FAILED: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
