"""Build `fixture_llama_mlp.npz`: the inputs of one real llama MLP plus the HARDWARE's output.

The fixture is what makes `verify_rtl_exact.py` self-contained -- it pins the RTL-exact config to a
known hardware result without needing the gemmini tree, spike, or a model download at verify time.

`Y_hw` is `fp8_matmul_model.tiled_matmul_hwlike` run over the chain, which spike reproduces
BIT-EXACTLY: `bareMetalC/llama_mlp.c` reports 0/65536 element mismatches against it on the same
operands (and the same ELF builds for the MxGemminiRocketConfig RTL path). So "matches the fixture"
means "matches the hardware", not "matches another python model".

Run from the gemmini tree, which owns the golden model and the capture:

    cd generators/gemmini/npu-exploration
    PATH=.venv/bin:$PATH .venv/bin/python3 rtl_exact/make_fixture.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GEMMINI = (HERE / ".." / ".." / "software" / "gemmini-rocc-tests").resolve()
NPU = (HERE / "..").resolve()
for p in (str(GEMMINI), str(NPU)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gen_llama_layer as L  # noqa: E402


def main() -> int:
    cap = L.load_capture()
    print(f"capture {cap['_path'].name}")
    d = L.build_mlp(cap)
    out = HERE / "fixture_llama_mlp.npz"
    np.savez_compressed(
        out,
        h_mid=cap["h_mid"].astype(np.float32),
        w_post_ln=cap["w_post_ln"].astype(np.float32),
        Wg=cap["Wg"].astype(np.float32),
        Wu=cap["Wu"].astype(np.float32),
        Wd=cap["Wd"].astype(np.float32),
        Y_hw=d["Y_bf16"].astype(np.float32),
        ref_mlp=cap["ref_mlp"].astype(np.float32),
        rms_eps=np.float32(d["eps"]),
        meta=np.array([f"layer{int(cap['meta_layer'])}", f"head{int(cap['meta_head'])}",
                       f"neurons{int(cap['meta_neuron0'])}-{int(cap['meta_neuron0']) + d['F']}",
                       f"M{d['M']}", f"D{d['D']}", f"F{d['F']}",
                       str(cap["meta_model_id"])]),
    )
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"  Y_hw {d['Y_bf16'].shape}, rel_fro vs fp32 = {100 * d['rel']:.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
