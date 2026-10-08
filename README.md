# npu-exploration

Define a model in PyTorch, compile it to Rocket-hosted MxGemmini RoCC instructions, run it, and grade
it bit for bit against the mxquant model of the same machine. **One ELF per kernel, any MX datatype.**

## Quickstart

Prerequisites: Linux x86_64, `conda` (miniconda is fine), git SSH access to
`ucb-bar/npu-exploration`, `ucb-bar/merlin`, `ucb-bar/gemmini` and `chloe-wong/microscaling-quant`
(optionally `Rakanic/MxGemmini-workspace` for silicon-cost numbers, `chooper1/MXQuant` for the
capture scripts and the legacy reference, and a CUDA GPU for the perplexity path), and ~10 GB of disk.

```bash
git clone --recurse-submodules git@github.com:ucb-bar/npu-exploration.git
cd npu-exploration
bash scripts/setup.sh                       # provisions EVERYTHING (idempotent; see below)
source scripts/env.sh                       # sets MERLIN_CHIPYARD, RISCV, PATH
.venv/bin/python run_kernel.py --kernel linear --hw baseline
```

`setup.sh` provisions, inside the clone: the python env (`.venv` + `requirements.txt`), the
`microscaling-quant` submodule (mxq, the quantization library every model here is built on), the
RISC-V toolchain (`riscv64-unknown-elf-gcc` and `dtc` from the `ucb-bar` conda channel, `spike` built
from `riscv-isa-sim` source — no chipyard build needed), the gemmini hardware sources, and a stock
`libgemmini.so` built with the toolchain's own g++. `bash scripts/setup.sh --check` prints
PASS/FAIL for every requirement.
Details, phases and troubleshooting: [`scripts/README.md`](scripts/README.md).

Without SSH keys for GitHub, switch the submodules to HTTPS first:

```bash
git config submodule.merlin.url https://github.com/ucb-bar/merlin.git
git config submodule.microscaling-quant.url https://github.com/chloe-wong/microscaling-quant.git
git submodule update --init merlin microscaling-quant
```

The MXQuant clone is optional (`bash scripts/setup.sh --with-mxquant`): the capture scripts, the
legacy reference (`--legacy-mxquant`) and regenerating `tests/oracle/block_fixture.npz` need it;
compiling and grading a kernel do not.

Already have a chipyard tree? Skip the toolchain phases and point at it instead:
`source scripts/env.sh /path/to/chipyard`. Hardware sources always come from that tree, with no
in-repo pin, so the model and the `spike` that loads it cannot drift apart.

If something fails to start, it is almost always the environment — see
[`scripts/README.md`](scripts/README.md).

## Use

```bash
.venv/bin/python run_kernel.py --list
.venv/bin/python run_kernel.py --kernel llama_mlp
.venv/bin/python run_kernel.py --kernel linear --hw wide_acc --run fp4_e2m1
.venv/bin/python run_kernel.py --kernel linear --hw wide_acc --models mxquant           # the model alone, no spike
.venv/bin/python -m models.mxquant --workload tinyllama --hw wide_acc --gpus 0,1,2,3    # perplexity
.venv/bin/python compile_kernel.py --kernel mlp3 --target mx_rocket                       # compile only: ELF + expected bits
.venv/bin/python compile_kernel.py --module tests/fixtures/modules.py:Attn                # a plain PyTorch module
```

`run_kernel.py` is the exploration entry point: PyTorch → quantize → lower → ELF → run → graded,
with every model of the recipe's machine ([`models/`](models/README.md)) run from the same
command. `compile_kernel.py` is the compile entry point: the same lowering and the same C, for the
spike or the RTL build, plus the bits the ELF must print (`expected.npy`) and a manifest, with no
spike run and no grade. Both take a registry kernel; `compile_kernel.py --module FILE.py:Name`
also takes a plain PyTorch module through the tracer in `kernels/trace.py`.

```
[recipe  ] baseline  dim=16  operand=fp8->bf16  prod=e4m3  acc[e4..8 m4..7]  build_id=854265d5…
[stage   ] 0 (L0) 64x64x64 -> bf16   cycles 445
[grade   ] finite 4096/4096  cycles 445
```

The final `VERDICT` line states whether the hardware matched the mxquant model bit for bit, followed
by one line per model (`PPA`, `PERF`, `PPL`). Without spike there is no verdict, only the model's
numbers (`MXQUANT … NO VERDICT`). Exit `0` on PASS or NO VERDICT, `1` on FAIL, `2` on error. Every run
is recorded under `results/<timestamp>_<kernel>_<shape>/`.

### Running RTL simulations: `simq`

[`tools/simq`](tools/README.md) queues Chipyard RTL simulation runs (VCS, Xcelium or Verilator) so you do not hand-drive
`make run-binary`. It
elaborates each config **once**, serially — sbt holds a lock — and runs everything else 8 at a time.

```bash
tools/simq build      CONFIG=MxGemminiRocketConfig
tools/simq run-debug  CONFIG=MxGemminiRocketConfig BINARY=mxl4
tools/simq run        CONFIG=MxGemminiRocketConfig BINARY=baremetal/mxgemmini/ladder.jobs JOBS=8
```

`BINARY=` takes one binary, a comma-separated list, or a `.jobs` file of them. `CONFIG=` and
`BINARY=` cross-multiply. Other variables: `TIMEOUT= MAX_CYCLES= JOBS= OUT= SIM= EXTRA_SIM_FLAGS=`.

`run-debug` also **dumps a waveform** next to each job's log (the simulator's Chipyard default format). Those are GB-scale, one per job, never
cleaned up — so give a big waved batch its own `OUT=`, or use `--no-waves` for the debug build
without the dump.

A run verb **queues and returns in about a second**, starting a background daemon on first use, so
you can keep appending. Add `--wait` to block in this terminal instead — that is the only form with
a meaningful exit code, since queueing returns before the work happens.

```bash
tools/simq run-debug CONFIG=MxGemminiRocketConfig BINARY=baremetal/mxgemmini/ladder.jobs
tools/simq run       CONFIG=MxDim32AllGemminiRocketConfig BINARY=matmul_tiled_fp8_64x64
tools/simq status        # queued / running / done, and which ones did not pass
tools/simq watch         # follow progress live (ctrl-C stops watching, not the daemon)
tools/simq clear         # retire finished records (--logs also drops their waveforms)
tools/simq stop          # claim no new jobs, exit once in-flight runs finish
tools/simq stop --now    # also kill the simulations that are mid-run
```

A queued job for a not-yet-built config elaborates it *while the already-built configs keep
running*. A simulator older than its Scala is re-elaborated by default, so you cannot accidentally
measure stale RTL — the mistake that cost two days in
[`planning/rtl_mx_faults_handoff.md`](planning/rtl_mx_faults_handoff.md).

| flag | default | meaning |
|---|---|---|
| `--kernel` | `linear` | which kernel (`--list`) |
| `--hw` | `baseline` | which hardware recipe (`--list`); `--config` is the same flag |
| `--run` | `default` | which run recipe: operand format, rounding, scale floor, pass threshold (`--list`) |
| `--models` | `default` | which models run: reference, mxquant, spike, ppa, perf, or a comma list of them |
| `--m --k --h --n` | 64 | batch rows, in_features, hidden, out_features |
| `--artifacts` | off | also write an RTL-replay bundle (C + `operands.npz`) |
| `--build-only` | off | stop at the ELF |
| `--per-stage-elf` | off | one ELF per matmul, intermediates carried by the host; the default fuses a chain or emits a graph as one ELF |
| `--legacy-mxquant` | off | grade with the previous reference (`grade/mxquant_ref.py`, MXQuant bundle extracted from the clone on first use); for the equivalence test, removed in the next PR |

### Tests

```bash
.venv/bin/python tests/selftest_grade.py       # and the rest — see tests/README.md
```

## How it fits together

```
    run_kernel.py  (every model of one machine)      compile_kernel.py  (ELF + expected bits, no run)
                              │                                   │
       ┌──────────────────────┼──────────────────────────┐        │
       ▼                      ▼                          ▼        ▼
 kernels/registry       config/recipe.py              --models   kernels/trace.py  (--module: a PyTorch module)
 KernelSpec (x, stages) Recipe = one machine       which models run
       │                      │
       │            ┌─────────┴──────────┐
       │            ▼                    ▼
       │     config/scheme.py     models/spike/build_spike.py
       │     recipe → mxq         recipe → libgemmini.so (per build_id)
       ▼            ▼                    ▼
 ┌───────────── grade/pipeline.run ────────────────────────────────────────┐
 │ reference  fp32                                                         │
 │ spike      LOWER: compiler/lower.py (compiler/graph, host_ops) → command buffer│
 │            run with GEMMINI_MODE=both: the bits, and the cycle model's time     │
 │            (fused chain | graph | per-stage)   → compiler/targets backend│
 │            mxgemm_emit / mxgraph_emit → main.c → ELF → spike             │
 │ mxquant    models/mxquant on mxq, fed the same wire operands             │
 │ ppa, perf  models/ppa, models/perf                                       │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            ▼
              grade/metrics + report → results/<run>/  → VERDICT · PPA · PERF · PPL
```

The recipe is the only source of the machine: `config/scheme.py` turns it into mxq's quantizer and
arithmetic for the mxquant model (bits per kernel, perplexity per workload), and `build_spike.py` turns the same JSON into the
functional model spike loads. The kernel is data (`kernels/registry.py`); the pipeline lowers it,
runs it, and grades the bits that came back against the mxquant model of the same recipe.

Inside `pipeline.run` the spike run and the three models that do not need its output run **at the
same time**: the mxquant model needs only the lowering's edges, ppa only the recipe, perf only the
stage shapes, so they start the moment the lowering has decided the edges and the grade joins them
after spike returns. Perplexity is its own command (`python -m models.mxquant --workload …`), one
worker process per GPU named in `--gpus`. To run many kernel × recipe combinations at once,
launch several `run_kernel.py` processes; each run writes its own `results/<timestamp>_…/` directory.

## Entry points

| what you want | command |
|---|---|
| grade a kernel on a machine, every model at once | `run_kernel.py --kernel attention --hw wide_acc` |
| just the bits a machine must produce, no spike, seconds | `run_kernel.py --kernel mlp3 --hw narrow_prod --models mxquant` |
| compile a kernel to an ELF for spike or the RTL build | `compile_kernel.py --kernel mlp3 --target mx_rocket` |
| compile a plain PyTorch module, no registry entry | `compile_kernel.py --module my.py:Block --input x.npy` |
| perplexity of a workload on one machine, same arithmetic | `python -m models.mxquant --workload tinyllama --hw baseline --run default --gpus 0,1,2,3` |
| silicon cost of one machine | `python -m models.ppa.ppa --hw baseline` |
| predicted timeline of one matmul on it | `python -m models.perf.perf --hw baseline --m 64 --k 64 --n 64` |
| build or list the per-recipe functional models | `python -m models.spike.build_spike --hw R --force` |
| the claims, one test each | `tests/selftest_*.py` (18 of them) and `tests/test_recipe_drift.py` |
| environment | `bash scripts/setup.sh --check`, `source scripts/env.sh` |

Kernels by name: `linear`, `mlp2` … `mlp8`, `attention`, and `llama_attention` / `llama_mlp` once a
capture exists. Machines: `baseline`, `flat_acc4`, `wide_acc`, `narrow_prod`, or any hardware recipe
JSON. Run recipes: `default`, `exact`, `bf16_tiles`, `fp4_e2m1`, or any run recipe JSON. Formats
(the run recipe's `operand_fmt`): every entry of `compiler/formats.py` on chains; `fp8_e4m3` only on
graph kernels. The old `--dtype`, `--tol` and `--allow-lossy-chain` flags are refused with the run
field that replaced each.
`compile_kernel.py` builds and writes `expected.npy` but runs nothing; running the `mx_rocket` ELF
on VCS or FPGA is the hardware team's step.

Less common:

| command | what it does |
|---|---|
| `python -m models.mxquant --workload tinyllama --hw R --dry-run` | which layers get the recipe's Scheme |
| `kernels/captures/llama_layer.py`, `tests/fixtures/llama_tiles.py` | capture real TinyLlama tensors for the llama kernels (needs the MXQuant clone) |
| `baremetal/mxgemmini/gen/gen_*.py` | generators for the hand-written TinyLlama kernels |
| `tests/verify_rtl_exact.py`, `tests/oracle/make_fixture.py` | the frozen llama-MLP fixture and its verifier (MXQuant under `rtl_exact/` equals the hardware) |
| `tools/extract_model.py` | one-off extraction from the gemmini tree |

Run everything with `.venv/bin/python` from the repo root after `source scripts/env.sh`.

## Which model reads which recipe field

`--hw` picks a hardware recipe (`config/hardware/`) and `--run` a run recipe (`config/run/`); fields
of the run recipe are written `run.*` below. Every field and its meaning is in
[`config/README.md`](config/README.md).

| field | mxquant | spike | ppa | perf | emitters |
|---|---|---|---|---|---|
| mesh size | window | DIM | cols | rows, cols | tile plan |
| product precision | yes | yes | yes | | |
| accumulator ladder | yes | yes | yes | | |
| `types.prodFloor` | flush | | | | |
| `mx.scaleSize`, `scaleSizeOut` | block | GROUP, GROUP_OUT | | | |
| `mx.enable_lut` | | | | | |
| `mx.lut` | | | | | lut_tables |
| `scratchpad` | | | | | bank_num, bank_rows |
| `implementation` | | | clock, util | clock | |
| `run.operand_fmt` | format | | stim | act, wei | wire format |
| `run.rounding`, `scale_floor` | quantizer | | | | |
| `run.reduce` | reducer | | | | |
| `run.fp32_tol` | | | | | |
| `run.allow_lossy_chain` | | | | | lowering |
| `run.lut` (LUT formats) | codebooks | | | lut-a/w/c | codebooks, lut_group |
| `run.vector` (optional) | softmax, RMSNorm precision | | | | |

`run.fp32_tol` is read by the grading step, not by a model: it is the pass threshold on the error
against fp32. `array.tileRows/tileColumns` and the `accumulator` widths are in `build_id` but no
Python code reads them. The kernel path accepts only a 16x16 mesh, 32-element blocks, a 4x4096
scratchpad, `rne`, the 2^-23 floor and `reduce: hardware`; `config.recipe.check` refuses anything else
before any work, and the perplexity path runs it. A LUT format needs, on both paths, a build whose
`mx.lut` serves it and a run recipe with a `lut` block; every LUT setting is in those two files and the
code holds no default for any of them. Both paths run a LUT format through the chip's tables
(`mxq.lut`, one rule: the compiler emits them, the perplexity path quantizes A and B through them).

## Where things are

```
run_kernel.py                 the exploration entry point (graded)
compile_kernel.py             the compile entry point (ELF + expected bits)
kernels/     registry.py spec.py trace.py host_ops.py captures/    the kernel IR; --list shows what is registered; trace.py = PyTorch module -> KernelSpec; host_ops = the host op vocabulary; captures/ = real TinyLlama tensors
config/      recipe.py hardware/*.json run/*.json scheme.py     the machine, how it is driven; both → mxq
models/      reference/ mxquant/ (bits + perplexity) spike/ ppa/ perf/   one folder per model of the machine
grade/       pipeline.py metrics.py report.py telemetry.py     run, compare, record
compiler/    formats.py wire.py codebook.py operands.py graph.py (tensors -> the wire); lower.py (KernelSpec -> command buffer); targets/mx_gemmini_rocket/ backend/{mxgemm_emit,mxgraph_emit,runner} runtime/
baremetal/   hand-written TinyLlama kernels and their generators
rtl_exact/   the MXQuant config that is the hardware (rtl_datapath.py, mxgemmini_rtl.json); mxmesh/ = the hardware team's extracted mesh model
tests/       every test: selftest_*.py, verify_rtl_exact.py, oracle/ fixtures, fixtures/ (modules, llama tiles), rtl/ simq regression lists
tools/       simq (RTL job queue), extraction        scripts/ setup.sh env.sh
merlin/  microscaling-quant/       submodules             MXQuant/   optional clone
out/  results/  .venv/  toolchain/ generated, gitignored
```

Each directory has its own README covering what it holds and what to do there.

| Dir | Contents |
|---|---|
| [`kernels/`](kernels/README.md) | the kernel registry — a kernel is data, not code. **Add kernels here.** |
| [`baremetal/`](baremetal/README.md) | hand-written application kernels per target (TinyLlama on MxGemmini) |
| [`config/`](config/README.md) | the two recipes: `hardware/` = one machine (`--hw`), `run/` = how software drives it (`--run`); `scheme.py` maps the pair onto mxq |
| [`models/`](models/README.md) | one folder per model of that machine: reference, mxquant (bits and perplexity), spike, ppa, perf |
| [`compiler/`](compiler/README.md) | the lowering (`lower.py`) and the backend that emits, builds and runs the ELF |
| [`grade/`](grade/README.md) | run, compare, record — and what the verdict means |
| [`rtl_exact/`](rtl_exact/README.md) | the reference configuration that matches the hardware bit for bit |
| [`sim/`](sim/README.md) | RTL simulation and FPGA emulation substrates |
| [`tests/`](tests/README.md) | the self-tests, and which claim each one defends |
| [`scripts/`](scripts/README.md) | environment setup, and the errors it exists to prevent |
| [`tools/`](tools/README.md) | `simq`, the Chipyard simulation queue; one-off extraction utilities |
| [`planning/`](planning/README.md) | plans, decisions, and the measurements behind them |
| `merlin/` | the compiler framework, git submodule, **unforked** |
| `microscaling-quant/` | mxq, the quantization library (block quantizers, the systolic arithmetic, `nn.patch`), git submodule pinned by SHA |
| `out/`, `results/`, `.venv/` | build artifacts, run records, the environment (gitignored) |

## Two standing constraints

**Do not depend on the sibling reference flow.** It is the reference for *structure* — the same
datapath, nearly the same programming model — but nothing here may include, link, or resolve a path
into it. Reimplement; take ideas, not code. Every *fact* the compiler needs is grounded in
`generators/gemmini/` instead: the headers, the spike model, the RTL Scala.

**The machine is one file.** `config/hardware/*.json` is the only description of the machine that
every model reads: mxquant, spike, ppa and perf. How software drives it is a separate file,
`config/run/*.json`. See [`config/README.md`](config/README.md).
