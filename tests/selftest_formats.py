"""Per-format gate: our compiler must reproduce the shipped baremetal test's golden.

Every MX format already has a PASSing hand-written test in
``gemmini-rocc-tests/bareMetalC``, whose header carries both the exact operand bytes and the exact
expected output (``C_out_bf16`` for the non-requant path). So for each format the check is:

    take that test's operands verbatim -> emit OUR C for them -> run on spike
                                       -> require our output == its ``C_out_bf16``

That is a stronger statement than "it runs": it says our codegen computes what the reference test
computes, on the reference's own data, bit for bit. It also needs no golden of our own, which is why
it is the gate for the formats MXQuant's ``rtl_exact`` does not yet cover — ``rtl_exact`` was
established on FP8 only (``planning/merlin_glue_port_plan.md`` §4.3), and pretending otherwise would
be claiming a correctness tier we have not earned.

    .venv/bin/python tests/selftest_formats.py            # every proven format
    .venv/bin/python tests/selftest_formats.py fp4_e2m1   # just one
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "app", REPO / "compiler" / "targets" / "mx_gemmini_rocket",
           REPO / "merlin" / "merlin" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mxiface                                   # noqa: E402
from app import mxformats                        # noqa: E402
from backend import runner                       # noqa: E402

ROCC = REPO.parent / "software" / "gemmini-rocc-tests"

#: format -> (header, the baremetal test it backs). Each entry is one shipped, PASSing test.
CASES = {
    "fp8_e4m3": ("matmul_fp8_64x64.h", "matmul_tiled_fp8_64x64"),
    "fp4_e2m1": ("matmul_fp4_64x64.h", "matmul_tiled_fp4_64x64"),
    # Non-square, and the reason it is here: the 64x64x64 case cannot distinguish an A row stride
    # of K from one of M, which is exactly the class of bug that returns plausible wrong numbers
    # (npu_exploration_bridge_plan.md section 10.2). A K != M shape is what actually tests the
    # packing derived from gemmini.cc:1536.
    "fp4_e2m1@128x128x512": ("matmul_fp4_128x128x512.h", "matmul_tiled_fp4_128x128x512"),
    "fp4_e2m1@128x128": ("matmul_fp4_128x128.h", "matmul_tiled_fp4_128x128"),
    # The LUT family. Operands are 4-bit indices into a 16-entry codebook; the codebook is DATA,
    # k-means over the quantized values, so it is read from the header alongside the operands.
    "fp8_e5m2": ("matmul_data_mx_lut_e5m2_64x64.h", "matmul_tiled_fp8_e5m2_64x64"),
    "fp8_e4m3_quad": ("matmul_data_mx_lut_e4m3_64x64.h", "matmul_tiled_fp8_e4m3_lut_64x64"),
    "fp6_e2m3": ("matmul_data_mx_lut_e2m3_64x64.h", "matmul_tiled_fp6_e2m3_lut_64x64"),
    "fp6_e3m2": ("matmul_data_mx_lut_hw.h", "matmul_tiled_fp6_128x128"),
    "fp6_e3m2@128x512x128": ("matmul_fp6_128x128x512.h", "matmul_tiled_fp6_128x128x512"),
}


def parse_array(text: str, name: str) -> tuple[np.ndarray, tuple[int, int]]:
    # Dimensions may be expressions (`B_in[MATMUL_K][MATMUL_N / 2]`), so accept anything but `]`.
    m = re.search(rf"static const uint(\d+)_t {re.escape(name)}\s*\[([^\]]*)\]"
                  r"\s*\[([^\]]*)\]\s*=\s*\{(.*?)\n\};", text, re.S)
    if not m:
        raise KeyError(f"{name} not found in header")
    width, body = int(m.group(1)), m.group(4)
    vals = [int(t, 0) for t in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", body)]
    dt = {8: np.uint8, 16: np.uint16, 32: np.uint32}[width]
    return np.array(vals, dtype=dt), (m.group(2), m.group(3))


def defines(text: str) -> dict[str, int]:
    return {k: int(v) for k, v in re.findall(r"#define\s+(MATMUL_\w+)\s+(\d+)", text)}


def run_case(fmt_name: str, header: str, test: str) -> tuple[bool, str]:
    f = mxformats.FORMATS[fmt_name.split("@")[0]]
    path = ROCC / "include" / header
    if not path.exists():
        return True, f"SKIP (no {header})"
    text = path.read_text()
    d = defines(text)
    M, K, N = d["MATMUL_M"], d["MATMUL_K"], d["MATMUL_N"]

    a_name = "A_in_hw" if f.bits == 4 else "A_in"
    a_codes = parse_array(text, a_name)[0].reshape(M // f.packed_per_byte, K)
    b_codes = parse_array(text, "B_in")[0].reshape(K, N // f.packed_per_byte)
    a_scales = parse_array(text, "A_scales_row")[0].reshape(K // mxformats.BLOCK, M)
    b_scales = parse_array(text, "B_scales_col")[0].reshape(K // mxformats.BLOCK, N)
    want = parse_array(text, "C_out_bf16")[0].reshape(M, N)

    bundle = {"a_codes": a_codes, "b_codes": b_codes,
              "a_scales": a_scales, "b_scales": b_scales}
    if f.lut:
        g = mxformats.LUT_GRANULARITY
        words = mxformats.lut_words(f.entry_bits)
        for key, name, n_lut in (("a_lut", "A_lut", M >> g), ("b_lut", "B_lut", N >> g),
                                 ("c_lut", "C_lut", M >> g)):
            bundle[key] = parse_array(text, name)[0].reshape(n_lut, words)

    stage = mxiface.MatmulStage(m=M, k=K, n=N, weight="W0", out="Y0", lhs="X", out_dtype="bf16")
    iface = mxiface.chain_interface_mlir([stage], operand_fmt=f.name)
    cb = mxiface.to_command_buffer(iface, [bundle])

    with tempfile.TemporaryDirectory() as td:
        elf = runner.compile_command_buffer(cb, td)
        console = runner.run_elf(elf)
    out, _metrics = runner.parse_output(console)
    if "Y0" not in out:
        return False, f"no OUT Y0 in the console output: {console[:300]!r}"
    got = np.array(out["Y0"], dtype=np.uint16).reshape(M, N)

    n_diff = int((got != want).sum())
    if n_diff == 0:
        return True, f"{M}x{K}x{N}  {got.size}/{got.size} bf16 patterns identical to {test}'s golden"
    idx = np.argwhere(got != want)[:3]
    ex = ", ".join(f"[{i},{j}] got 0x{got[i, j]:04x} want 0x{want[i, j]:04x}" for i, j in idx)
    return False, f"{M}x{K}x{N}  {n_diff}/{got.size} differ; e.g. {ex}"


#: Chained cases: (format, header, test). The gate is the RESIDENT INTERMEDIATE -- stage 0's
#: requantized codes and block scales, read back out of the scratchpad -- because that is precisely
#: what the resident chain produces and what the reference test checks first.
CHAIN_CASES = {
    "fp8_e4m3": ("matmul_fp8_64x64_chain.h", "matmul_tiled_fp8_64x64_chain"),
    "fp4_e2m1": ("matmul_fp4_64x64_chain.h", "matmul_tiled_fp4_64x64_chain"),
    # The codebook chain. Its header ships C1_lut -- stage 0's OUTPUT book, which is also stage 1's
    # ACTIVATION book. Using the reference's own tables here separates two questions that would
    # otherwise be tangled: does the emitter drive a LUT chain correctly (this), and is our
    # estimated codebook any good (measured separately, since it changes accuracy not correctness).
    "fp6_e3m2": ("matmul_fp6_64x64_chain.h", "matmul_tiled_fp6_64x64_chain"),
}


def run_chain_case(fmt_name: str, header: str, test: str) -> tuple[bool, str]:
    """A 2-stage chain on the reference's own operands; compare our C1 to its golden."""
    f = mxformats.FORMATS[fmt_name]
    path = ROCC / "include" / header
    if not path.exists():
        return True, f"SKIP (no {header})"
    text = path.read_text()
    d = defines(text)
    M, K, N = d["MATMUL_M"], d["MATMUL_K"], d["MATMUL_N"]
    ppb, gk, gn = f.packed_per_byte, K // mxformats.BLOCK, N // mxformats.BLOCK

    a_name = "A_in_hw" if f.bits == 4 else "A_in"
    bundles = [
        {"a_codes": parse_array(text, a_name)[0].reshape(M // ppb, K),
         "b_codes": parse_array(text, "B_in")[0].reshape(K, N // ppb),
         "a_scales": parse_array(text, "A_scales_row")[0].reshape(gk, M),
         "b_scales": parse_array(text, "B_scales_col")[0].reshape(gk, N)},
        {"b_codes": parse_array(text, "B2_in")[0].reshape(K, N // ppb),
         "b_scales": parse_array(text, "B2_scales_col")[0].reshape(gk, N)},
    ]
    if f.lut:
        g, w = mxformats.LUT_GRANULARITY, mxformats.lut_words(f.entry_bits)
        c1 = parse_array(text, "C1_lut")[0].reshape(M >> g, w)
        bundles[0] |= {"a_lut": parse_array(text, "A_lut")[0].reshape(M >> g, w),
                       "b_lut": parse_array(text, "B_lut")[0].reshape(N >> g, w),
                       "c_lut": c1}
        # stage 1's A book IS stage 0's C book -- that is the whole coupling.
        bundles[1] |= {"a_lut": c1,
                       "b_lut": parse_array(text, "B2_lut")[0].reshape(N >> g, w),
                       "c_lut": parse_array(text, "C2_lut")[0].reshape(M >> g, w)}
    want_codes = parse_array(text, "C1_out")[0].reshape(M // ppb, N)
    want_scales = parse_array(text, "C1_scales_out")[0].reshape(M, gn)

    et = f.mlir or f.name
    stages = [
        mxiface.MatmulStage(m=M, k=K, n=N, weight="W0", out="T0", lhs="X", out_dtype=et),
        mxiface.MatmulStage(m=M, k=N, n=N, weight="W1", out="Y0", lhs="T0", out_dtype="bf16"),
    ]
    cb = mxiface.to_command_buffer(
        mxiface.chain_interface_mlir(stages, operand_fmt=f.name), bundles)

    with tempfile.TemporaryDirectory() as td:
        elf = runner.compile_command_buffer(cb, td)
        console = runner.run_elf(elf)
    out, _ = runner.parse_output(console)
    if "T0" not in out:
        return False, f"no OUT T0: {console[:300]!r}"
    got_codes = np.array(out["T0"], dtype=np.uint8).reshape(M // ppb, N)
    got_scales = np.array(out["T0_scales"], dtype=np.uint8).reshape(M, gn)

    nc = int((got_codes != want_codes).sum())
    ns = int((got_scales != want_scales).sum())
    if nc == ns == 0:
        return True, (f"{M}x{K}x{N}  resident C1: {got_codes.size}/{got_codes.size} codes and "
                      f"{got_scales.size}/{got_scales.size} scales identical to {test}'s golden")
    return False, f"{M}x{K}x{N}  C1 differs: {nc}/{got_codes.size} codes, {ns}/{got_scales.size} scales"


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(CASES)
    fails = n = 0
    print(f"{'case':22s} {'result':6s}  detail")
    for name in wanted:
        if name not in CASES:
            print(f"{name:22s} SKIP    no shipped test registered for it")
            continue
        try:
            ok, detail = run_case(name, *CASES[name])
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        fails += not ok
        n += 1
        print(f"{name:22s} {'PASS' if ok else 'FAIL':6s}  {detail}")

    print(f"\n{'chained (resident intermediate)':22s}")
    for name, args in CHAIN_CASES.items():
        if argv[1:] and name not in argv[1:]:
            continue
        try:
            ok, detail = run_chain_case(name, *args)
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        fails += not ok
        n += 1
        print(f"{name + ' chain':22s} {'PASS' if ok else 'FAIL':6s}  {detail}")

    print(f"\n{n} case(s), {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
