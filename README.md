# npu-exploration

End-to-end exploration of models on MxGemmini hardware: define a model in PyTorch, compile it to
Rocket-hosted MxGemmini RoCC instructions, run it on spike — and later on a cycle-accurate substrate.

Every layer of the stack lives in this one repo, split by directory.

| Dir | Layer | Contents |
|---|---|---|
| `app/` | L1 — model | libraries: MX quantization, and emitting/lowering `merlin_iface` MLIR |
| `kernels/` | L1 — model | what to run: a kernel is a PyTorch model flattened into matmul stages |
| `compiler/` | L2 — compiler | `targets/mx_gemmini_rocket/` — the out-of-tree merlin target (contract + backend) |
| `grade/` | L2 — evaluation | run a kernel, compare against an FP32 reference, record the run |
| `sim/` | L3 — substrates | reserved for RTL simulation and FPGA emulation (the spike runner lives in the backend) |
| `merlin/` | framework | compiler framework, git submodule, **unforked** |
| `radiance-kernels/` | reference | read-only; **not a dependency** (see below) |
| `planning/` | — | plans and design decisions |
| `scripts/`, `tests/` | — | environment setup; self-tests |
| `out/`, `results/`, `.venv/` | — | build artifacts, run records, the Python environment (all gitignored) |

External references, outside this repo:

- `../software/gemmini-rocc-tests/` — the **baremetal C reference**: `include/gemmini.h` (the MX
  intrinsics this target emits) and the hand-written MX tests, buildable with `build_spike.sh`.
- `../software/libgemmini/` — the spike functional model of the MX datapath.

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

**One config artifact, two consumers.** The long-term direction is that software emits a JSON of
configurations driving *both* compilation *and* hardware generation. Until then the target contract
is the near-term stand-in, and each of its fields is tagged `[RTL]` / `[ABI]` / `[COMPILE]` so the
migration is mechanical. **Not started.**

## Running a kernel

`run_kernel.py` is the single entry point: PyTorch → quantize → merlin → ELF → spike → **graded**.

```bash
source scripts/env.sh
.venv/bin/python run_kernel.py --list
.venv/bin/python run_kernel.py --kernel linear
```

```
[setup    ] repo=/home/…/npu-exploration  merlin=yes  wired=3 paths
[kernel   ] linear: [64][64] through 64 -> 64   (1 stage, seam=n/a)
[reference] fp32 torch (64, 64)  range [-2.137, 1.917]
[toolchain] gcc=…/riscv64-unknown-elf-gcc  spike=…/spike
[stage    ] 0 (L0) 64x64x64 -> bf16   cycles 282
[grade    ] rel_fro=5.9100%  mae=0.02741  max_abs=0.1369  finite 4096/4096  cycles 282
[report   ] PASS (tier=fp32) -- saved to results/20260902-…_linear_64x64x64

VERDICT  PASS  (tier=fp32, rel_fro=5.9100%, tol=15.00%, cycles=282)
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
```

| kernel | model | matmuls |
|---|---|---|
| `linear` | `nn.Linear(K→N, bias=False)` | 1 |
| `mlp2` | `nn.Sequential(Linear, Linear)` | 2 |
| `mlp3` | three stacked `Linear` | 3 |
| `attention` | single-head attention | 6 mesh + 1 host (softmax) |

| flag | default | meaning |
|---|---|---|
| `--kernel` | `linear` | which kernel (`--list`) |
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
│                         toolchain paths, repo + merlin git heads
├── metrics.json          verdict, error metrics, per-stage cycles
├── log.jsonl             one JSON record per stage, with timestamps
├── hardware_output.npy   what the device computed
└── fp32_reference.npy    what PyTorch computed
```

Grading has two tiers, and **only the first exists today**:

| tier | compares against | answers |
|---|---|---|
| **fp32** *(current)* | `torch.matmul` in fp32 | what the MX pipeline costs in accuracy |
| **golden** *(not built)* | an exact model of the MX datapath | whether the hardware is **correct** |

The FP32 tier cannot distinguish "MX is lossy" from "the RTL is wrong". `metrics.compare` and
`report.write_report` already take the golden as an optional argument and `metrics["tier"]` reports
which was used, so a weaker grade is never mistaken for a stronger one.

```bash
.venv/bin/python tests/selftest_grade.py
```

24 checks on the grading machinery itself — metrics math, verdict logic (a tightened tolerance really
does flip PASS→FAIL; a single NaN fails regardless of tolerance), telemetry buffering, results
layout. No hardware needed. It exists so you can tell "the hardware is wrong" from "our checker is
wrong"; it does **not** test quantization or the datapath.

## Status

**PyTorch → ELF → spike works, and is graded.** Measured on the current elaboration:

| kernel | shape | cycles | rel_fro vs fp32 |
|---|---|---|---|
| `linear` | 64×64×64 | 282 | 5.91% |
| `linear` | 32×128×96 | 467 | 5.74% |
| `mlp2` | 64×64×64 | 564 | 9.33% |
| `mlp3` | 64×64×64 | 846 | 12.25% |

Arbitrary shapes (M, K, N independently) work. Consistency across shapes is itself evidence the
operand layout is right — a transposed operand or an off-by-one scale index would not land on the
same figure repeatedly.

Next: the MX golden model (hardware correctness), then the JSON hardware config.

## Troubleshooting

**`*** FAILED *** (tohost = 1337)`** — an unhandled trap, nearly always a **stale
`libgemmini.so`**. Its Makefile lists only `gemmini.cc` as a prerequisite, so edits to
`mx_fp_math.h` never trigger a rebuild, and a stale model fails silently this way rather than with a
useful error. `scripts/env.sh` compares mtimes and warns.

```bash
(cd $MERLIN_CHIPYARD/generators/gemmini/software/libgemmini && make)
```

**`Unable to load extlib … GLIBCXX_3.4.32 not found`** — `libgemmini.so` was built with a newer g++
than spike. `spike` carries a **DT_RPATH**, which outranks `LD_LIBRARY_PATH`, so no environment
variable can fix it; rebuild libgemmini with the plain `g++` on `PATH`.

**`Failed to run dtc`** — spike shells out to the device-tree compiler, which lives in the chipyard
conda env. `scripts/env.sh` puts it on `PATH`.

**`$RISCV is unset`** — chipyard's `env.sh` only activates conda. Use `scripts/env.sh`.

**`toolchain NOT AVAILABLE`** — `gcc`, `spike` or `libgemmini.so` is missing; the `[toolchain]` log
line prints the paths being checked.

See [`planning/npu_exploration_bridge_plan.md`](planning/npu_exploration_bridge_plan.md).
