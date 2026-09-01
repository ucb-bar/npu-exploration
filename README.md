# npu-exploration

End-to-end exploration of models on MxGemmini hardware: define a model in PyTorch, compile it to
Rocket-hosted MxGemmini RoCC instructions, run it on spike — and later on a cycle-accurate substrate.

Every layer of the stack lives in this one repo, split by directory.

| Dir | Layer | Contents |
|---|---|---|
| `app/` | L1 — model | PyTorch models, quantization, experiment definitions |
| `compiler/` | L2 — compiler | `targets/mx_gemmini_rocket/` — the out-of-tree merlin target (contract + backend) |
| `runtime/` | L3 — runtime | C harness, output protocol, include glue |
| `sim/` | L4 — simulation | spike (`libgemmini`) runner; later Verilator / FireSim |
| `merlin/` | framework | compiler framework, git submodule, **unforked** |
| `radiance-kernels/` | reference | read-only; **not a dependency** (see below) |
| `planning/` | — | plans and design decisions |
| `out/` | — | runs and artifacts |

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

Step 3 of 10. Toolchain verified and the fp8 64×64 golden established on spike; the target contract
resolves. No backend yet.

See [`planning/npu_exploration_bridge_plan.md`](planning/npu_exploration_bridge_plan.md).
