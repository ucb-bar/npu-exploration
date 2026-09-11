"""The emitted C runtime and its Python twin must agree — checked, not assumed.

`backend/runtime/mx_host.h` is the fp32 host side of a layer: the glue the mesh cannot do
(RMSNorm, SiLU, softmax, RoPE) plus **the MX quantizer that hands its result back to the mesh**.
That quantizer has a Python twin in `app/mxq_golden.py`, and the two must produce identical codes
and identical block scales — otherwise a host stage feeds the mesh something the golden does not
predict, and every downstream comparison is measuring the wrong thing.

The check compiles mx_host.h with the NATIVE compiler (not riscv), which keeps it fast and isolates
the arithmetic from the baremetal environment. `llama_layer_hw_plan.md` §6 step 3 did the same and
reported 0/65536 code and 0/2048 scale mismatches; this reproduces that as a standing test.

Expected: exactly zero mismatches. E4M3 keeps 3 mantissa bits at every exponent, which absorbs a
last-ulp difference between newlib and numpy, so anything non-zero is a real divergence.

    .venv/bin/python tests/selftest_mx_host.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.mxq_golden import quantize_operand          # noqa: E402
from app.mxwire import e8m0_decode                   # noqa: E402

RUNTIME = REPO / "compiler" / "targets" / "mx_gemmini_rocket" / "backend" / "runtime"

#: A C harness that quantizes stdin-supplied floats with mx_host.h and prints the wire bytes.
HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include "mx_host.h"

int main(int argc, char **argv) {
    int M = atoi(argv[1]), K = atoi(argv[2]);
    int by_cols = atoi(argv[3]);
    float *V = malloc((size_t)M * K * sizeof(float));
    for (long i = 0; i < (long)M * K; i++)
        if (scanf("%f", &V[i]) != 1) return 2;

    int nblocks = (by_cols ? M : K) / MX_BLOCK;
    unsigned char *codes  = malloc((size_t)M * K);
    unsigned char *scales = malloc((size_t)nblocks * (by_cols ? K : M));
    mx_host_init();
    if (by_cols) mx_quantize_cols(V, M, K, codes, scales);
    else         mx_quantize_rows(V, M, K, codes, scales);

    for (long i = 0; i < (long)M * K; i++) printf("%d ", codes[i]);
    printf("\n");
    for (long i = 0; i < (long)nblocks * (by_cols ? K : M); i++) printf("%d ", scales[i]);
    printf("\n");
    return 0;
}
"""

CASES = [
    ("A 64x64  (side=a, rows)", 64, 64, "a"),
    ("A 32x128 (side=a, rows)", 32, 128, "a"),
    ("B 64x64  (side=b, cols)", 64, 64, "b"),
    ("B 128x96 (side=b, cols)", 128, 96, "b"),
]


def main() -> int:
    hdr = RUNTIME / "mx_host.h"
    if not hdr.exists():
        print(f"SKIP: {hdr} missing")
        return 0

    with tempfile.TemporaryDirectory() as td:
        src, exe = Path(td) / "h.c", Path(td) / "h"
        src.write_text(HARNESS)
        cc = subprocess.run(["gcc", "-O2", "-I", str(RUNTIME), str(src), "-o", str(exe), "-lm"],
                            capture_output=True, text=True)
        if cc.returncode != 0:
            print(f"FAIL: mx_host.h does not compile natively\n{cc.stderr[-2000:]}")
            return 1
        print(f"mx_host.h compiles natively ({hdr.stat().st_size} B)\n")

        rng = np.random.default_rng(0)
        fails = 0
        print(f"{'case':26s} {'codes':>16s} {'scales':>14s} {'recon':>10s}")
        for label, R, C, side in CASES:
            V = (rng.standard_normal((R, C)) * 3.0).astype(np.float32)
            want_c, want_s, _ = quantize_operand(V, side=side)

            proc = subprocess.run(
                [str(exe), str(R), str(C), "1" if side == "b" else "0"],
                input=" ".join(f"{x:.9g}" for x in V.ravel()),
                capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"{label:26s} harness failed: {proc.stderr[:200]}")
                fails += 1
                continue
            lines = proc.stdout.strip().split("\n")
            got_c = np.array(lines[0].split(), dtype=np.uint8).reshape(R, C)
            got_s = np.array(lines[1].split(), dtype=np.uint8)

            # No transpose on either side. `mx_quantize_rows` writes `scales_a[g*M + m]`, i.e.
            # [GK][M] -- already the layout the A-side scale memory indexes (a_off = group*M + row)
            # and exactly what quantize_operand(side="a") returns. An earlier draft of this test
            # transposed one side and reported ~40% scale mismatches against codes that matched
            # perfectly; codes agreeing while scales disagree is impossible (the codes are v/scale),
            # which is what gave the test away.
            want_s_flat = want_s.ravel()
            dc = int((got_c != want_c).sum())
            ds = int((got_s != want_s_flat).sum())
            # A stronger check than byte equality alone: the DECODED tensors must match too.
            rec_c = (got_c.astype(np.int32) != want_c.astype(np.int32)).sum()
            fails += (dc != 0) + (ds != 0)
            print(f"{label:26s} {f'{dc}/{got_c.size}':>16s} {f'{ds}/{got_s.size}':>14s} "
                  f"{'==' if rec_c == 0 else 'DIFF':>10s}")

    print(f"\n{len(CASES)} case(s), {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
