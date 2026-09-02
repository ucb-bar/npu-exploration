# app — model-facing libraries

The model side of the stack: turning float tensors into what the accelerator consumes, and
describing the computation in the compiler's input language. **Libraries only — no scripts.**
`run_kernel.py` is the entry point; `kernels/` says what to run.

| file | role |
|---|---|
| `mxquant.py` | float tensors ↔ MX operand codes + E8M0 block scales; bf16 decode; the chain rescale |
| `mxiface.py` | emit `merlin_iface` interface MLIR, and lower it to a command buffer via merlin |

Everything operand- and model-specific lives here. The compiler backend receives a command buffer
and knows nothing about tensors, models, or files.

## Quantization scaling

`mxquant.TARGET_CODE_EXP = 0` is **not** the textbook OCP MX choice (which normalizes each block to
the full ±448 e4m3 range). The mesh accumulates a 16-deep column at exponent width 4 for 15 of its
16 rows and saturates near 2⁸, so full-range operands overflow to ±inf and the tile returns NaN.
A target exponent `e` puts peak codes in `[2**e, 2**(e+1))`; `e=0` leaves 4× headroom under the
`16·C² ≤ 256` bound for almost no precision cost. See the constant's docstring.

`quantize_rows` **refuses non-finite input**: `fp8_e4m3_to_code` maps NaN/inf to code 0 (mirroring
`mx_fp_math.h`), so a silent pass would turn an upstream accumulator overflow into a zero and the
run would look clean.

## The MLIR handoff

`mxiface` emits **`merlin_iface`** — merlin's frozen contract grammar
(`merlin/contract/interface_grammar.md`), the documented input format for an out-of-tree target
package — and merlin's own `parse_interface_mlir` lowers it to the command buffer. It is the same
grammar the shipped MX capsules use, so a capsule and this front end are interchangeable inputs to
the backend (verified: `MB0_mxfp8_linear` builds and runs on spike). The grammar is deliberately
xDSL-free: plain text in, plain text out.

## Reference

`../../software/gemmini-rocc-tests/` is the baremetal C reference for the ISA and the hand-written
MX tests, buildable with `build_spike.sh`.
