# app — model-facing libraries

The model side of the stack: turning float tensors into what the accelerator consumes, and
describing the computation in the compiler's input language. **Libraries only — no scripts.**
`run_kernel.py` is the entry point; `kernels/` says what to run.

| file | role |
|---|---|
| `mxquant.py` | float tensors ↔ MX operand codes + E8M0 block scales; bf16 decode; the chain rescale |

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

## The handoff to the compiler

`compiler/lower.py` turns a KernelSpec into the command buffer the backend emits C from; nothing here writes MLIR or talks to merlin any more.

## Reference

`../../software/gemmini-rocc-tests/` is the baremetal C reference for the ISA and the
hand-written ISA tests: `./build_spike.sh bareMetalC` for spike, `./build_mx_rocket.sh
bareMetalC` for the RTL path. The whole-application kernels that used to live beside them are now
in [`../baremetal/mxgemmini/`](../baremetal/mxgemmini/README.md).

See [`../README.md`](../README.md) for install and the run command.
