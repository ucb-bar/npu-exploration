# L1 — app

The front end: models, quantization, and the experiments you actually want to run.

| file | role |
|---|---|
| `torch_linear/run_linear.py` | PyTorch `nn.Linear` → ELF → spike |
| `mxquant.py` | float tensors → MX operand codes + E8M0 block scales |
| `mxcb.py` | operands → a merlin command buffer |

Everything operand- and model-specific lives here. The compiler backend receives a command buffer
and knows nothing about tensors, models, or files.

```bash
cd <chipyard-root> && source ./env.sh
.venv/bin/python app/torch_linear/run_linear.py --m 32 --k 128 --n 96
```

## Note on quantization scaling

`mxquant.TARGET_CODE_EXP = 2` is **not** the textbook OCP MX choice (which normalizes each block to
the full ±448 e4m3 range). This datapath's intermediate accumulator has a 4-bit exponent and
saturates near 256, so full-range operands overflow to ±inf and the tile comes back all-NaN. See the
constant's docstring for the derivation.

## Reference

`../../software/gemmini-rocc-tests/` is the baremetal C reference for the ISA and the hand-written
MX tests — including `bareMetalC/matmul_tiled_fp8_64x64.c`, which the emitter was written from and
which can be built and run on spike directly (`build_spike.sh`).
