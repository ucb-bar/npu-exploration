# npu-exploration

End-to-end exploration of models on MxGemmini hardware: define a model in PyTorch, compile it to
Rocket-hosted MxGemmini RoCC instructions, run it on spike — and later on a cycle-accurate substrate.

Every layer of the stack lives in this one repo, split by directory.

| Dir | Layer | Contents |
|---|---|---|
| `app/` | L1 — model | libraries: MX quantization, and emitting/lowering `merlin_iface` MLIR |
| `kernels/` | L1 — model | what to run: a kernel is a PyTorch model flattened into matmul stages |
| `config/` | L1 — hardware | recipes: one JSON = one machine (`--config`) |
| `compiler/` | L2 — compiler | `targets/mx_gemmini_rocket/` — the out-of-tree merlin target (contract + backend) |
| `grade/` | L2 — evaluation | run a kernel, compare against an FP32 reference, record the run |
| `sim/` | L3 — substrates | reserved for RTL simulation and FPGA emulation (the spike runner lives in the backend) |
| `merlin/` | framework | compiler framework, git submodule, **unforked** |
| `radiance-kernels/` | reference | read-only; **not a dependency** (see below) |
| `planning/` | — | plans and design decisions |
| `scripts/`, `tests/` | — | environment setup; self-tests |
| `out/`, `results/`, `.venv/` | — | build artifacts, run records, the Python environment (all gitignored) |

The hardware references come from the surrounding chipyard tree — there is **no in-repo pin**, so
the spike model and the `spike` that loads it cannot drift apart:

- `generators/gemmini/software/gemmini-rocc-tests/` — the **baremetal C reference**:
  `include/gemmini.h` (the MX intrinsics this target emits) and the hand-written MX tests.
- `generators/gemmini/software/libgemmini/` — the spike functional model of the MX datapath.

A pin was tried and removed: it pinned `libgemmini` (a nested submodule) five commits behind the
working tree, which fails *silently* — the model still builds and runs, just with the pre-RNE
rounding, and only chained kernels notice.

`radiance-kernels/` is not checked in; clone it alongside only if you want to read it.

## Getting started

```bash
source scripts/env.sh          # or: source scripts/env.sh /path/to/chipyard
```

Sets `MERLIN_CHIPYARD`, `RISCV` and `PATH`, and warns if the spike functional model is stale.

Two things worth knowing, both of which cost real debugging time: chipyard's own `env.sh` **does
not** set `$RISCV` (it only activates conda — the toolchain is at `.conda-env/riscv-tools`), and
spike shells out to `dtc`, which lives in the chipyard conda env and is not on `PATH` by default.
`scripts/env.sh` handles both.

First time only, if `merlin/` is empty — `.gitmodules` uses SSH:

```bash
git config submodule.merlin.url https://github.com/ucb-bar/merlin.git
git submodule update --init merlin
```


## Two standing constraints

**Do not depend on `radiance-kernels/`.** The Radiance MX flow is the reference for *structure* — it
is the same datapath and nearly the same programming model — but nothing here may include, link, or
resolve a path into that tree. Reimplement; take ideas, not code. Every *fact* the compiler needs is
grounded in `generators/gemmini/` (headers, the spike model, the RTL Scala) instead.

**One config artifact, two consumers.** Software emits a JSON of configurations driving *both*
compilation *and* hardware generation. `config/recipes/*.json` is that artifact: it patches the
spike model (`config/build_spike.py`) and elaborates the Chisel
(`config/scala/JsonGemminiConfig.scala`), with `tests/test_recipe_drift.py` holding the two in
agreement. The older target contract remains the stand-in for the fields recipes do not cover
yet, tagged `[RTL]` / `[ABI]` / `[COMPILE]` so that migration stays mechanical.

## Running a kernel

`run_kernel.py` is the single entry point: PyTorch → quantize → merlin → ELF → spike → **graded**.

```bash
source scripts/env.sh
.venv/bin/python run_kernel.py --list
.venv/bin/python run_kernel.py --kernel linear --config baseline
```

```
[setup   ] repo=/home/…/npu-exploration  merlin=yes  wired=3 paths
[kernel  ] linear: [64][64]  L0[64x64x64]   (1 mesh, 0 host)   (1 stage, seam=n/a)
[recipe  ] baseline  dim=16  operand=fp8->bf16  prod=e4m3  acc[e4..8 m4..7]  block=32   build_id=854265d5…
[ladder  ] col 0-7      acc=e4m4  prod=e4m3
[ladder  ] col 8-9      acc=e4m5  prod=e4m3
[ladder  ] col 10-14    acc=e4m6  prod=e4m3
[ladder  ] col 15       acc=e8m7  prod=e4m3
[reference] fp32 torch (64, 64)  range [-2.137, 1.917]
[build   ] recipe 'baseline' matches the stock build; using the shipped libgemmini.so
[toolchain] gcc=…/riscv64-unknown-elf-gcc  spike=…/spike
[geometry] spike reports dim=16, matches recipe
[stage   ] 0 (L0) 64x64x64 -> bf16   cycles 282
[grade   ] rel_fro=5.9122%  mae=0.02743  max_abs=0.1369  finite 4096/4096  cycles 282
[report  ] PASS (tier=fp32) -- saved to results/20260910-…_linear_64x64x64

VERDICT  PASS  (tier=fp32, no golden (fp32 tier only))
COST     rel_fro=5.9122% vs fp32  (tol=15.00%, cycles=282)
```

Exit `0` on PASS, `1` on FAIL, `2` on error. The front end emits **`merlin_iface` interface MLIR** —
merlin's frozen contract grammar — and merlin lowers it to the command buffer. That is the same
grammar the shipped MX capsules use, so a capsule and this front end are interchangeable inputs to
the backend. merlin also discovers and loads the backend (`get_backend("mx_gemmini_rocket")`).

## Configuring the kernel

```bash
.venv/bin/python run_kernel.py --kernel linear --m 32 --k 128 --n 96
.venv/bin/python run_kernel.py --kernel mlp2 --h 128 --seam rescale
.venv/bin/python run_kernel.py --kernel mlp3 --m 128 --k 128 --h 128 --n 128 --artifacts
.venv/bin/python run_kernel.py --kernel linear --config wide_acc
```

A **recipe** is the hardware half of a run: mesh size, the per-column product and
accumulator precisions, and the block-scale group. `--config` picks one; the hashed
sections give it a `build_id`, and a recipe that is not the stock machine gets its own
`libgemmini.so` built and cached under `out/builds/<build_id>/`.

| recipe | product | accumulator ladder |
|---|---|---|
| `baseline` | e4m3 | m4×8 → m5×2 → m6×5 → e8m7 (stock) |
| `flat_acc4` | e4m3 | e4m4 flat |
| `wide_acc` | e4m3 | e8m7 flat |
| `narrow_prod` | **e4m2** | same ladder as baseline |

| kernel | model | matmuls |
|---|---|---|
| `linear` | `nn.Linear(K→N, bias=False)` | 1 |
| `mlp2` | `nn.Sequential(Linear, Linear)` | 2 |
| `mlp3` | three stacked `Linear` | 3 |
| `attention` | single-head attention | 6 mesh + 1 host (softmax) |

| flag | default | meaning |
|---|---|---|
| `--kernel` | `linear` | which kernel (`--list`) |
| `--config` | `baseline` | which hardware recipe (`--list`) |
| `--m --k --h --n` | 64 | batch rows, in_features, hidden, out_features (`attention`: seq, d_model, d_head) |
| `--seed` | 0 | tensor values |
| `--seam weight\|rescale` | `weight` | how a chained intermediate's scale is made safe |
| `--tol` | 0.15 | pass threshold on relative Frobenius error vs FP32 |
| `--artifacts` | off | also write an RTL-replay bundle (MLIR + C + `operands.npz`) |
| `--build-only` | off | stop at the ELF |
| `--simulator` | `spike` | verilator is not wired up |

**Shape rules**, checked before anything is built so mistakes are cheap: `M`, `K`, `N` multiples of
the PE tile (`dim`, currently 16); `K` a multiple of the block-scale group (32); in a chain each
stage's `K` equals the previous stage's `N`, and a stage feeding another needs `N` a multiple of 32
(the requantizer emits one E8M0 code per 32 output columns).

**Adding a kernel** is a few lines in `kernels/registry.py` and no other change — a kernel is *data*,
not code. `from_module()` accepts `nn.Linear` and `nn.Sequential` of them, and raises on bias or
activations rather than quietly computing something else: both need a COMMIT epilogue, which the
backend refuses.

Attention **is** expressible: softmax is a `HostStage`, and `S = Q@K^T` / `O = P@V` use operands
that reference earlier stages rather than resident weights. What a kernel still cannot express is an
op the *backend* would have to lower differently — a fused epilogue, or a convolution.

### Chains and the scale seam

The backend lowers one matmul per command buffer, so an N-layer model is N buffers, N ELFs, N spike
runs, with intermediates travelling back through the host. The last stage commits to `bf16`; earlier
stages commit to fp8 codes + E8M0 scales through the requantizer.

That junction needs care. The requantizer normalizes each block to the element format's **full**
range (codes peak at 448) while the mesh accumulates a 16-deep column at exponent width 4, saturating
near 2⁸ — feed one straight into the other and the tile comes back all-NaN. `--seam weight` absorbs
the shift into the next layer's block scales (peak code 448); `--seam rescale` re-splits
`(code, scale)` on the host, dividing codes by 2⁶ and adding 6 to the exponent (peak code 7,
value-identical). Both grade the same.

## Grading and telemetry

Every stage logs to stderr and to a structured `log.jsonl`, and each run is recorded:

```
results/<timestamp>_<kernel>_<shape>/
├── config.json           shapes, seed, quantization settings, mesh geometry,
│                         the recipe (build_id + per-column ladder), toolchain
│                         paths, repo + merlin + gemmini git heads
├── metrics.json          verdict, error metrics, per-stage cycles
├── log.jsonl             one JSON record per stage, with timestamps
├── hardware_output.npy   what the device computed
└── fp32_reference.npy    what PyTorch computed
```

The verdict is **bit-identity against MXQuant**, not a tolerance against fp32:

| tier | compares against | answers |
|---|---|---|
| `mxquant_rtl_exact` | MXQuant + `rtl_exact/rtl_datapath.install()` | is the hardware correct? |
| `mxquant_shipped` | MXQuant as normally configured | how far is the model from the silicon? |
| `fp32` | `torch.matmul` | what MX costs at all — context only |

## Status

Nine kernels, every one a single ELF, every one **bit-identical to MXQuant under `rtl_exact`**:

| kernel | shape | steps | vs MXQuant | vs fp32 |
|---|---|---|---|---|
| `linear` | 64³ | 1 mesh | 4096/4096 | 5.91% |
| `mlp2` … `mlp8` | 64³ | 2–8 mesh, fused chain | 4096/4096 | 8.97% … 22.7% |
| `attention` | 64³ | 6 mesh + 1 host | 4096/4096 | 13.62% |
| **`llama_mlp`** | 32×2048 | 3 mesh + 2 host | **65536/65536** | **11.5928%** |
| **`llama_attention`** | 32×2048 | 6 mesh + 4 host | **65536/65536** | **13.4157%** |

The two llama kernels are a **real TinyLlama decoder layer** — real captured activations and
weights, `d_model = 2048` kept full — and reproduce the hand-written `bareMetalC/llama_*.c`
reference kernels to within **5 ppm** (11.5933% / 13.4161%).

**All six MX formats are bit-identical to MXQuant under `rtl_exact`** — as single matmuls and as
fused chains up to eight stages deep (4096/4096 in every cell):

| | `fp8_e4m3` | `fp8_e4m3_quad` | `fp8_e5m2` | `fp6_e3m2` | `fp6_e2m3` | `fp4_e2m1` |
|---|---|---|---|---|---|---|
| single | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2-stage chain | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 8-stage chain | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Each is additionally gated against its shipped baremetal test's golden. **Two of those goldens are
currently stale**: spike moved to RNE rounding on 2026-09-10, so `matmul_fp8_64x64_chain.h` and
`matmul_fp6_64x64_chain.h` need regenerating (the old datapath reproduces them exactly; the new one
differs by 250 and 56 adjacent codes, all `odd → even`). `selftest_formats` reports those two as
FAIL until they are. Everything above is measured against the current spike.

**Not verified:** nothing has been run on RTL. The C compiles for `-DMX_ROCKET` and its instruction
stream matches the reference tests', but no simulator is built here.

## Tests

```bash
.venv/bin/python tests/selftest_grade.py          # metrics, telemetry, reporting
.venv/bin/python tests/selftest_quantizer.py      # our quantizer == the baremetal headers, byte for byte
.venv/bin/python tests/selftest_formats.py        # all 6 formats + 3 chains vs their shipped goldens
.venv/bin/python tests/selftest_mx_host.py        # the C runtime == its Python twin
.venv/bin/python tests/selftest_requant.py       # the chained requantizer, per step, vs a C oracle
.venv/bin/python tests/selftest_extracted.py      # app/mxmesh/ == the models it was extracted from
.venv/bin/python tests/selftest_mx_rocket_build.py  # one C source, builds for spike AND MX_ROCKET
.venv/bin/python rtl_exact/verify_rtl_exact.py    # MXQuant under rtl_exact == the hardware
```

## Troubleshooting

**`*** FAILED *** (tohost = 1337)`** — an unhandled trap, nearly always a **stale
`libgemmini.so`**. Its Makefile lists only `gemmini.cc` as a prerequisite, so edits to
`mx_fp_math.h` never trigger a rebuild, and a stale model fails silently this way rather than with a
useful error. `scripts/env.sh` compares mtimes and warns.

Sources come from the chipyard tree, and a per-recipe build is
rebuilt automatically when the pin moves. To force one:

```bash
.venv/bin/python -m config.build_spike --config <recipe> --force
```

**`Unable to load extlib … GLIBCXX_3.4.32 not found`** — `libgemmini.so` was built with a newer g++
than spike. `spike` carries a **DT_RPATH**, which outranks `LD_LIBRARY_PATH`, so no environment
variable can fix it. Do **not** reach for the `g++` on `PATH`: `scripts/env.sh` puts chipyard's
conda first, and its g++ (13.2) is exactly the one that causes this. `config/build_spike.py`
resolves a spack 12.2 g++ explicitly; override with `MX_HOST_GXX` if it moves.

**`Failed to run dtc`** — spike shells out to the device-tree compiler, which lives in the chipyard
conda env. `scripts/env.sh` puts it on `PATH`.

**`$RISCV is unset`** — chipyard's `env.sh` only activates conda. Use `scripts/env.sh`.

**`toolchain NOT AVAILABLE`** — `gcc`, `spike` or `libgemmini.so` is missing; the `[toolchain]` log
line prints the paths being checked.

See [`planning/npu_exploration_bridge_plan.md`](planning/npu_exploration_bridge_plan.md).
