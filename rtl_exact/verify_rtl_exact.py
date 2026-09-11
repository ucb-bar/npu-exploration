"""The gate: MXQuant's simulated matmul, under this config, must equal the HARDWARE bit for bit.

Runs MXQuant's own `MXLinearSim` over the real llama MLP in `fixture_llama_mlp.npz` twice -- as
shipped, and with `rtl_datapath.install()` -- and compares both to `Y_hw`, which spike reproduces
element-for-element (`bareMetalC/llama_mlp.c`: 0/65536 mismatches).

Expected:

    as shipped        : 10.63% vs fp32, 17.90% vs hardware      -- correlated, not equal
    rtl_exact installed: 11.59% vs fp32,  0.00% vs hardware      -- 65536/65536 identical

A FAIL means one of three things, in order of likelihood: MXQuant's `_simulate_atw` was
restructured and `rtl_datapath.install()`'s copy of it needs re-syncing; the hardware's arithmetic
changed (`fp8_matmul_model` / `gemmini.cc`); or the config drifted. Each is worth knowing loudly,
which is the point of keeping this runnable.

    cd generators/gemmini/npu-exploration
    .venv/bin/python3 rtl_exact/verify_rtl_exact.py

`eval_complete.py` lives on MXQuant's `chloe-branch-all`; if it is not importable this script
extracts it from that branch into a temp dir rather than asking you to switch branches.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPU = HERE.parent
MXQ = NPU / "MXQuant"
PRODACC_FILES = ("eval_complete.py", "mx_quantization.py", "lut_quantization.py")
BRANCH = "origin/chloe-branch-all"


def _prodacc_dir(explicit: Path | None) -> Path:
    """MXQuant's prod/acc bundle, extracted from its branch if it is not already on disk."""
    if explicit:
        return explicit
    for cand in (MXQ / "prodacc_bundle", MXQ / "FP4_complete_integration_e2e"):
        if (cand / "eval_complete.py").exists():
            return cand
    out = Path(tempfile.mkdtemp(prefix="mxq_prodacc_"))
    for f in PRODACC_FILES:
        blob = subprocess.run(["git", "show", f"{BRANCH}:prodacc_bundle/{f}"],
                              cwd=MXQ, capture_output=True, text=True)
        if blob.returncode != 0:
            raise SystemExit(f"cannot read prodacc_bundle/{f} from {BRANCH} in {MXQ}:\n{blob.stderr}")
        (out / f).write_text(blob.stdout)
    print(f"[setup] extracted MXQuant's prod/acc bundle from {BRANCH} -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prodacc", type=Path, default=None,
                    help="directory holding MXQuant's eval_complete.py (default: extract from "
                         f"{BRANCH})")
    ap.add_argument("--fixture", type=Path, default=HERE / "fixture_llama_mlp.npz")
    args = ap.parse_args()

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(_prodacc_dir(args.prodacc)))

    import torch
    import torch.nn as nn
    import eval_complete as EC
    import rtl_datapath

    cfg = rtl_datapath.load_config()
    fx = np.load(args.fixture, allow_pickle=False)
    print(f"[fixture] {args.fixture.name}  {' '.join(str(s) for s in fx['meta'])}")
    eps = float(fx["rms_eps"])

    def rmsnorm(h, w):
        h = h.astype(np.float32)
        inv = 1.0 / np.sqrt((h * h).mean(axis=-1, keepdims=True) + eps)
        return (h * inv * w.astype(np.float32)).astype(np.float32)

    def silu(x):
        return (x / (1.0 + np.exp(-x))).astype(np.float32)

    def linear(W, x):
        K, N = W.shape
        orig = nn.Linear(K, N, bias=False)
        with torch.no_grad():
            orig.weight.copy_(torch.from_numpy(np.ascontiguousarray(W.T)))
        sim = EC.MXLinearSim(orig, cfg.mx_fmt, False, cfg.product[0], cfg.product[1],
                             cfg.acc_schedule, 0, 0, window=cfg.window)
        with torch.no_grad():
            return sim(torch.from_numpy(np.ascontiguousarray(x))).numpy().astype(np.float32)

    def run_mlp():
        xn = rmsnorm(fx["h_mid"], fx["w_post_ln"])
        g, u = linear(fx["Wg"], xn), linear(fx["Wu"], xn)
        return linear(fx["Wd"], silu(g) * u)

    hw, ref = fx["Y_hw"], fx["ref_mlp"]
    rel = lambda a, b: 100.0 * float(np.linalg.norm(a - b) / np.linalg.norm(b))

    print("[run] MXQuant as shipped ...")
    y_ship = run_mlp()
    print("[run] MXQuant with rtl_exact installed ...")
    rtl_datapath.install(EC, cfg)
    assert rtl_datapath.is_installed(EC)
    y_rtl = run_mlp()

    same = int((y_rtl == hw).sum())
    print()
    print(f"  as shipped         : {rel(y_ship, ref):6.2f}% vs fp32   {rel(y_ship, hw):6.2f}% vs hardware")
    print(f"  rtl_exact installed: {rel(y_rtl, ref):6.2f}% vs fp32   {rel(y_rtl, hw):6.2f}% vs hardware")
    print(f"  identical elements : {same}/{y_rtl.size}   max abs diff {float(np.abs(y_rtl - hw).max()):g}")
    print()

    if same == y_rtl.size:
        print("PASS -- MXQuant under rtl_exact/mxgemmini_rtl.json is BIT-IDENTICAL to the hardware.")
        return 0
    print("FAIL -- see this file's docstring for the three things that cause it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
