# Step 2: Compile

The compile step lowers the traced model to MxGemmini RoCC instructions and builds one bare-metal ELF
for a Rocket host.

```bash
npu.py --hw baseline --model m.py:Block --skip evaluate                       # compiler/out/Block/spike/
npu.py --hw baseline --model m.py:Block --skip evaluate --target mx_rocket    # the RTL / FPGA build
```

## The output directory

| file | contents |
|---|---|
| `mx_gemmini_rocket.elf`, `main.c` | the program and its C driver |
| `command_buffer.json` | what the backend emitted from (without the byte arrays) |
| `operands.npz` | the input and every wire operand |
| `expected.npy` | the bits the ELF must print: the quantized model of this hardware |
| `manifest.json` | configs, `build_id`, format, lowering, toolchain, repo heads, hashes |

`--target` changes only the build define (`spike`: `-DSPIKE_SIM`; `mx_rocket`: `-DMX_ROCKET`); the C is
byte-identical. Running the `mx_rocket` ELF on RTL simulation or FPGA happens outside this repo.

## How it lowers

```
nn.Module --trace--> KernelSpec --lower--> command buffer --emit--> main.c --gcc--> ELF
            kernels/trace.py   compiler/lower.py        compiler/targets/mx_gemmini_rocket/backend/
```

`compiler/lower.py` turns a KernelSpec and an operand format into one **command buffer**, following
merlin's MLIR interface conventions (`compiler/merlin/` submodule). Two shapes come out:

- a straight **chain** of matmuls fuses into one program whose intermediates stay in the scratchpad;
- anything else (several live values, computed operands, host ops) becomes a **graph**, every edge
  through host fp32 memory, which the hardware requires wherever a host op sits between two matmuls.

Both produce one ELF. Graph kernels support `fp8_e4m3` only (the host-side requantizer in `mx_host.h`
encodes E4M3).

| file | role |
|---|---|
| `compiler/formats.py`, `wire.py`, `codebook.py` | the format table; float <-> wire bytes and E8M0 scales; LUT encoding |
| `compiler/operands.py` | the operand encoder shared with the quantized model |
| `compiler/graph.py`, `lower.py` | KernelSpec -> command buffer (chain or graph) |
| `backend/mxgemm_emit.py`, `mxgraph_emit.py` | command buffer -> C driver |
| `backend/runner.py` | toolchain, build, spike run, console parser |
| `backend/runtime/mx_host.h` | the host-side C every driver includes (softmax, RMSNorm, requantize) |

The backend is also an out-of-tree merlin target (`mx_gemmini_rocket`, resolved from
`MERLIN_TARGET_PATH=<repo>/compiler/targets`).

## Status: the merlin path

Today `lower.py` builds the command buffer directly, held by `tests/selftest_lower.py` to what merlin's
parser produced (`tests/oracle/command_buffers.json`), and `npu.py` does not load merlin. Routing the
compile step through merlin's MLIR pipeline end to end is the next piece of work.
