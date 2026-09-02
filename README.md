# npu-exploration

End-to-end exploration of models on MxGemmini hardware: define a model in PyTorch, compile it to
Rocket-hosted MxGemmini RoCC instructions, run it on spike — and later on a cycle-accurate substrate.

Every layer of the stack lives in this one repo, split by directory.

| Dir | Layer | Contents |
|---|---|---|
| `app/` | L1 — model | PyTorch models, quantization, experiment definitions |
| `compiler/` | L2 — compiler | `targets/mx_gemmini_rocket/` — the out-of-tree merlin target (contract + backend) |
| `sim/` | L3 — substrates | reserved for RTL simulation and FPGA emulation (the spike runner lives in the backend) |
| `merlin/` | framework | compiler framework, git submodule, **unforked** |
| `radiance-kernels/` | reference | read-only; **not a dependency** (see below) |
| `planning/` | — | plans and design decisions |
| `out/`, `.venv/` | — | build artifacts and the Python environment (both gitignored) |

External references, outside this repo:

- `../software/gemmini-rocc-tests/` — the **baremetal C reference**: `include/gemmini.h` (the MX
  intrinsics this target emits) and the hand-written MX tests, buildable with `build_spike.sh`.
- `../software/libgemmini/` — the spike functional model of the MX datapath.

## Getting started

```bash
cd <chipyard-root> && source ./env.sh            # sets $RISCV; not set by default
export MERLIN_TARGET_PATH=$PWD/compiler/targets  # resolves the OOT target, beats anything in-tree
export PYTHONPATH=$PWD/merlin/merlin/python
```

## Two standing constraints

**Do not depend on `radiance-kernels/`.** The Radiance MX flow is the reference for *structure* — it
is the same datapath and nearly the same programming model — but nothing here may include, link, or
resolve a path into that tree. Reimplement; take ideas, not code. Every *fact* the compiler needs is
grounded in `generators/gemmini/` (headers, the spike model, the RTL Scala) instead.

**One config artifact, two consumers.** The long-term direction is that software emits a JSON of
configurations driving *both* compilation *and* hardware generation. Until then the target contract
is the near-term stand-in, and each of its fields is tagged `[RTL]` / `[ABI]` / `[COMPILE]` so the
migration is mechanical.

## Status

**PyTorch → ELF → spike works.** One command takes a layer defined in PyTorch all the way to
hardware:

```bash
cd <chipyard-root> && source ./env.sh                 # sets $RISCV
export PYTHONPATH=$PWD/merlin/merlin/python           # merlin lowers the MLIR
.venv/bin/python app/torch_linear/run_linear.py
```

```
model     nn.Linear(64 -> 64, bias=False), input [64][64]
quantize  mxfp8 e4m3 + E8M0 block scales (group 32, peak code 2^2)
lower     merlin_iface MLIR -> command buffer (4 commands: RES_PACK, MATMUL_RESIDENT, COMMIT, EVICT)
elf       out/build/torch_linear/mx_gemmini_rocket.elf  (27024 bytes)
spike     Y0 (64, 64) bf16   METRIC {'cycles': 282, ...}
          finite 4096/4096   range [-2.062, 1.875]
```

The front end emits **`merlin_iface` interface MLIR** — merlin's frozen contract grammar — and
merlin lowers it to the command buffer. That is the same grammar the shipped MX capsules use, so a
capsule and this front end are interchangeable inputs to the backend.

Arbitrary shapes (M, K, N independently) — `--m 32 --k 128 --n 96` works. The front end's job is to
*produce the ELF*; **spike is the reference** for what the hardware computes, so nothing here tries
to predict the numbers.

merlin also discovers and loads the backend (`get_backend("mx_gemmini_rocket")`).

Next: more layer types, then whole models. Cycle-accurate (Verilator) is deferred — not on the path
to the software bridge.

> **No automated correctness gate.** The emitter was written from
> `../software/gemmini-rocc-tests/bareMetalC/matmul_tiled_fp8_64x64.c` and verified bit-exact
> against it, but that check was a bring-up scaffold and has been removed along with the rest of the
> baremetal-C mapping. Today's runs only confirm outputs are *finite*. To re-check correctness after
> changing the emitter, build and run that reference test directly (`build_spike.sh`) and compare.

See [`planning/npu_exploration_bridge_plan.md`](planning/npu_exploration_bridge_plan.md).
