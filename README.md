# npu-exploration

Define a model in PyTorch, compile it to Rocket-hosted MxGemmini RoCC instructions, run it on
spike, and grade it. **One ELF per kernel, any MX datatype.**

```bash
source scripts/env.sh /path/to/chipyard
.venv/bin/python run_kernel.py --list
.venv/bin/python run_kernel.py --kernel llama_mlp
.venv/bin/python run_kernel.py --kernel linear --dtype fp4_e2m1 --config wide_acc
```

## Status

Nine kernels, every one a single ELF, every one **bit-identical to the quantization reference under
`rtl_exact`** — the configuration in which that reference reproduces the datapath exactly:

| kernel | shape | steps | vs reference | vs fp32 |
|---|---|---|---|---|
| `linear` | 64³ | 1 mesh | 4096/4096 | 5.91% |
| `mlp2` … `mlp8` | 64³ | 2–8 mesh, fused chain | 4096/4096 | 8.97% … 22.7% |
| `attention` | 64³ | 6 mesh + 1 host | 4096/4096 | 13.62% |
| `llama_mlp` | 32×2048 | 3 mesh + 2 host | 65536/65536 | 11.5928% |
| `llama_attention` | 32×2048 | 6 mesh + 4 host | 65536/65536 | 13.4157% |

The two llama kernels are a real TinyLlama decoder layer — captured activations and weights,
`d_model = 2048` kept full — reproducing the hand-written `bareMetalC/llama_*.c` kernels to within
**5 ppm**.

All six MX formats are bit-identical as single matmuls **and** as fused chains eight stages deep:
`fp8_e4m3`, `fp8_e4m3_quad`, `fp8_e5m2`, `fp6_e3m2`, `fp6_e2m3`, `fp4_e2m1`.

**Not verified:** nothing has run on RTL. The C compiles for `-DMX_ROCKET` and its instruction
stream matches the baremetal tests', but no simulator is built here.

## Layout

| Dir | Contents |
|---|---|
| `app/` | MX quantization, host ops, the format table, codebooks, interface MLIR |
| `kernels/` | the kernel registry — a kernel is data, not code |
| `config/` | hardware recipes: one JSON = one machine (`--config`) |
| `compiler/` | `targets/mx_gemmini_rocket/` — the out-of-tree merlin target (contract + backend) |
| `grade/` | run, compare, record |
| `rtl_exact/` | the reference configuration that matches the hardware bit for bit |
| `sim/` | reserved for RTL simulation and FPGA emulation |
| `merlin/` | the compiler framework, git submodule, **unforked** |
| `planning/` | plans, decisions, and the measurements behind them |
| `scripts/`, `tests/` | environment setup; self-tests |
| `out/`, `results/`, `.venv/` | build artifacts, run records, the environment (gitignored) |

Hardware sources come from the surrounding chipyard tree — the MX intrinsics this target emits
(`include/gemmini.h`), the baremetal reference tests, and spike's functional model of the datapath.
There is **no in-repo pin**, so the model and the `spike` that loads it cannot drift apart. Build
and install the functional model from gemmini, and spike picks it up from there.

Two external checkouts are expected alongside, neither checked in: the **MX quantization
framework** (the reference this repo grades against — a hard dependency, imported by
`app/mxq_golden.py`) and a **sibling reference kernel flow**, which is read-only and *not* a
dependency.

## Getting started

```bash
source scripts/env.sh          # or: source scripts/env.sh /path/to/chipyard
```

Sets `MERLIN_CHIPYARD`, `RISCV` and `PATH`. Two things that cost real debugging time: chipyard's own
`env.sh` does **not** set `$RISCV` (it only activates conda — the toolchain is at
`.conda-env/riscv-tools`), and spike shells out to `dtc`, which is not on `PATH` by default.

First time only, if `merlin/` is empty — `.gitmodules` uses SSH:

```bash
git config submodule.merlin.url https://github.com/ucb-bar/merlin.git
git submodule update --init merlin
```

## Two standing constraints

**Do not depend on the sibling reference flow.** It is the reference for *structure* — the same
datapath, nearly the same programming model — but nothing here may include, link, or resolve a path
into it. Reimplement; take ideas, not code. Every *fact* the compiler needs is grounded in
`generators/gemmini/` instead: the headers, the spike model, the RTL Scala.

**One config artifact, two consumers.** `config/recipes/*.json` drives *both* compilation and
hardware generation: it patches the spike model (`config/build_spike.py`) and elaborates the Chisel
(`config/scala/JsonGemminiConfig.scala`), with `tests/test_recipe_drift.py` holding the two in
agreement.

## Running a kernel

`run_kernel.py` is the single entry point: PyTorch → quantize → merlin → ELF → spike → graded.

```
[recipe  ] baseline  dim=16  operand=fp8->bf16  prod=e4m3  acc[e4..8 m4..7]  build_id=854265d5…
[ladder  ] col 0-7      acc=e4m4  prod=e4m3
[build   ] recipe 'baseline' matches the stock build; using the shipped libgemmini.so
[geometry] spike reports dim=16, matches recipe
[stage   ] 0 (L0) 64x64x64 -> bf16   cycles 445
[grade   ] finite 4096/4096  cycles 445
```

The final `VERDICT` line states whether the hardware matched the reference bit for bit, and how
many elements were identical. Exit `0` on PASS, `1` on FAIL, `2` on error.

A **recipe** is the hardware half of a run: mesh size, the per-column product and accumulator
precisions, the block-scale group. A recipe that is not the stock machine gets its own functional
model, built and cached under `out/builds/<build_id>/`.

| recipe | product | accumulator ladder |
|---|---|---|
| `baseline` | e4m3 | m4×8 → m5×2 → m6×5 → e8m7 (stock) |
| `flat_acc4` | e4m3 | e4m4 flat |
| `wide_acc` | e4m3 | e8m7 flat |
| `narrow_prod` | **e4m2** | same ladder as baseline |

| flag | default | meaning |
|---|---|---|
| `--kernel` | `linear` | which kernel (`--list`) |
| `--config` | `baseline` | which hardware recipe (`--list`) |
| `--dtype` | `fp8_e4m3` | MX operand format (`app/mxformats.py`) |
| `--m --k --h --n` | 64 | batch rows, in_features, hidden, out_features |
| `--tol` | 0.15 | pass threshold on relative Frobenius error vs fp32 |
| `--artifacts` | off | also write an RTL-replay bundle (MLIR + C + `operands.npz`) |
| `--build-only` | off | stop at the ELF |

**Shape rules**, checked before anything is built: `M`, `K`, `N` multiples of the PE tile (`dim`);
`K` a multiple of the block-scale group (32); in a chain each stage's `K` equals the previous
stage's `N`, and a stage feeding another needs `N` a multiple of 32 (the requantizer emits one E8M0
code per 32 output columns).

**Adding a kernel** is a few lines in `kernels/registry.py` — a kernel is *data*. What a kernel
cannot express is an op the *backend* would lower differently: a fused epilogue, or a convolution.

Two emitters, on purpose: one fuses a straight **chain** and keeps intermediates in the scratchpad,
never touching the host; the other handles any **graph** — multiple live values, computed operands,
host ops — with every edge through host fp32 memory, which is what the hardware requires wherever a
host op sits in a seam. Both produce one ELF.

## Grading

The verdict is **bit-identity against the quantization reference**, not a tolerance against fp32:

| tier | compares against | answers |
|---|---|---|
| rtl-exact | the reference under `rtl_exact/rtl_datapath.install()` | is the hardware correct? |
| as-shipped | the reference as normally configured | how far is the model from the silicon? |
| fp32 | `torch.matmul` | what MX costs at all — context only |

Every run is recorded under `results/<timestamp>_<kernel>_<shape>/`: `config.json` (shapes, seed,
recipe + ladder, toolchain paths, git heads, and a hash of the model that actually ran),
`metrics.json`, `log.jsonl`, plus the hardware and fp32 outputs as `.npy`.

## Tests

```bash
.venv/bin/python tests/selftest_grade.py            # metrics, telemetry, reporting
.venv/bin/python tests/selftest_quantizer.py        # our quantizer == the baremetal headers, byte for byte
.venv/bin/python tests/selftest_formats.py          # all 6 formats + 3 chains vs their shipped goldens
.venv/bin/python tests/selftest_mx_host.py          # the C runtime == its Python twin
.venv/bin/python tests/selftest_requant.py          # the chained requantizer, per step, vs a C oracle
.venv/bin/python tests/selftest_extracted.py        # app/mxmesh/ == the models it was extracted from
.venv/bin/python tests/selftest_mx_rocket_build.py  # one C source, builds for spike AND MX_ROCKET
.venv/bin/python tests/test_recipe_drift.py         # recipe JSON == the Chisel it elaborates
.venv/bin/python rtl_exact/verify_rtl_exact.py      # the reference under rtl_exact == the hardware
```

## Troubleshooting

**`*** FAILED *** (tohost = 1337)`** — an unhandled trap, nearly always a stale functional model.
Rebuild and install it from gemmini (`make clean` first — its Makefile misses header changes), or
force a per-recipe rebuild:

```bash
.venv/bin/python -m config.build_spike --config <recipe> --force
```

**`Unable to load extlib … GLIBCXX_3.4.32 not found`** — the model was built with a newer g++ than
spike, and spike's `DT_RPATH` outranks `LD_LIBRARY_PATH`, so no environment variable fixes it.
Rebuild it with an older g++; `MX_HOST_GXX` selects one.

**`Failed to run dtc`** — spike shells out to the device-tree compiler in the chipyard conda env;
`scripts/env.sh` puts it on `PATH`.

**`$RISCV is unset`** — chipyard's `env.sh` only activates conda. Use `scripts/env.sh`.

## Plans

`planning/merlin_glue_port_plan.md` is the live one — decisions, every step's gate, and the
measurements behind them.
