# L1 — app

The front end: models, quantization, and the experiments you actually want to run.

| file | role |
|---|---|
| `torch_linear/run_linear.py` | PyTorch `nn.Linear` → MLIR → ELF → spike |
| `mxquant.py` | float tensors → MX operand codes + E8M0 block scales |
| `mxiface.py` | matmul → `merlin_iface` interface MLIR, and MLIR → command buffer |

Everything operand- and model-specific lives here. The compiler backend receives a command buffer
and knows nothing about tensors, models, or files.

```bash
cd <chipyard-root> && source ./env.sh
export PYTHONPATH=$PWD/merlin/merlin/python      # merlin lowers the MLIR
.venv/bin/python app/torch_linear/run_linear.py --m 32 --k 128 --n 96
```

## The MLIR handoff

The front end emits **`merlin_iface`** interface MLIR — merlin's frozen contract grammar
(`merlin/contract/interface_grammar.md`), the documented input format for an out-of-tree target
package. merlin's own `parse_interface_mlir` lowers it to the command buffer.

The emitted `.mlir` is written beside the ELF, and it is the same grammar the shipped MX capsules
use — so a capsule from `merlin/contract/capsules/mx_gemmini/` and this front end are
**interchangeable inputs** to the backend. Verified: `MB0_mxfp8_linear` builds and runs on spike.

The grammar is deliberately decoupled from xDSL (plain text in, plain text out), so no xDSL is
needed.

## Note on quantization scaling

`mxquant.TARGET_CODE_EXP = 2` is **not** the textbook OCP MX choice (which normalizes each block to
the full ±448 e4m3 range). This datapath's intermediate accumulator has a 4-bit exponent and
saturates near 256, so full-range operands overflow to ±inf and the tile comes back all-NaN. See the
constant's docstring for the derivation.

## Reference

`../../software/gemmini-rocc-tests/` is the baremetal C reference for the ISA and the hand-written
MX tests — including `bareMetalC/matmul_tiled_fp8_64x64.c`, which the emitter was written from and
which can be built and run on spike directly (`build_spike.sh`).
