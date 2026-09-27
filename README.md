# npu-exploration

Define a model in PyTorch, compile it to Rocket-hosted MxGemmini RoCC instructions, run it, and grade
it against the quantization reference. **One ELF per kernel, any MX datatype.**

## Quickstart

Prerequisites: Linux x86_64, `conda` (miniconda is fine), git SSH access to
`ucb-bar/npu-exploration`, `ucb-bar/merlin`, `ucb-bar/gemmini` and `chooper1/MXQuant`
(optionally `Rakanic/MxGemmini-workspace` for silicon-cost numbers), and ~10 GB of disk.

```bash
git clone --recurse-submodules git@github.com:ucb-bar/npu-exploration.git
cd npu-exploration
bash scripts/setup.sh                       # provisions EVERYTHING (idempotent; see below)
source scripts/env.sh                       # sets MERLIN_CHIPYARD, RISCV, PATH
.venv/bin/python run_kernel.py --kernel linear --config baseline
```

`setup.sh` provisions, inside the clone: the python env (`.venv` + `requirements.txt`), the
MXQuant checkout (`app/mxq_golden.py` imports it and nothing here works without it), the RISC-V
toolchain (`riscv64-unknown-elf-gcc` and `dtc` from the `ucb-bar` conda channel, `spike` built
from `riscv-isa-sim` source — no chipyard build needed), the gemmini hardware sources, and a stock
`libgemmini.so` built with the toolchain's own g++. `bash scripts/setup.sh --check` prints
PASS/FAIL for every requirement.
Details, phases and troubleshooting: [`scripts/README.md`](scripts/README.md).

Without SSH keys for GitHub, switch the merlin submodule to HTTPS first:

```bash
git config submodule.merlin.url https://github.com/ucb-bar/merlin.git
git submodule update --init merlin
```

Already have a chipyard tree? Skip the toolchain phases and point at it instead:
`source scripts/env.sh /path/to/chipyard`. Hardware sources always come from that tree, with no
in-repo pin, so the model and the `spike` that loads it cannot drift apart.

If something fails to start, it is almost always the environment — see
[`scripts/README.md`](scripts/README.md).

## Use

```bash
.venv/bin/python run_kernel.py --list
.venv/bin/python run_kernel.py --kernel llama_mlp
.venv/bin/python run_kernel.py --kernel linear --dtype fp4_e2m1 --config wide_acc
```

`run_kernel.py` is the single entry point: PyTorch → quantize → merlin → ELF → run → graded.

```
[recipe  ] baseline  dim=16  operand=fp8->bf16  prod=e4m3  acc[e4..8 m4..7]  build_id=854265d5…
[stage   ] 0 (L0) 64x64x64 -> bf16   cycles 445
[grade   ] finite 4096/4096  cycles 445
```

The final `VERDICT` line states whether the hardware matched the reference bit for bit. Exit `0` on
PASS, `1` on FAIL, `2` on error. Every run is recorded under `results/<timestamp>_<kernel>_<shape>/`.

| flag | default | meaning |
|---|---|---|
| `--kernel` | `linear` | which kernel (`--list`) |
| `--config` | `baseline` | which hardware recipe (`--list`) |
| `--dtype` | `fp8_e4m3` | MX operand format |
| `--m --k --h --n` | 64 | batch rows, in_features, hidden, out_features |
| `--tol` | 0.15 | pass threshold on relative Frobenius error vs fp32 |
| `--artifacts` | off | also write an RTL-replay bundle (MLIR + C + `operands.npz`) |
| `--build-only` | off | stop at the ELF |

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

### Tests

```bash
.venv/bin/python tests/selftest_grade.py       # and the rest — see tests/README.md
```

## Where things are

Each directory has its own README covering what it holds and what to do there.

| Dir | Contents |
|---|---|
| [`app/`](app/README.md) | MX quantization, host ops, the format table, codebooks, interface MLIR |
| [`kernels/`](kernels/README.md) | the kernel registry — a kernel is data, not code. **Add kernels here.** |
| [`baremetal/`](baremetal/README.md) | hand-written application kernels per target (TinyLlama on MxGemmini) |
| [`config/`](config/README.md) | hardware recipes: one JSON = one machine (`--config`) |
| [`compiler/`](compiler/README.md) | the out-of-tree merlin target: contract + backend |
| [`grade/`](grade/README.md) | run, compare, record — and what the verdict means |
| [`rtl_exact/`](rtl_exact/README.md) | the reference configuration that matches the hardware bit for bit |
| [`sim/`](sim/README.md) | RTL simulation and FPGA emulation substrates |
| [`tests/`](tests/README.md) | the self-tests, and which claim each one defends |
| [`scripts/`](scripts/README.md) | environment setup, and the errors it exists to prevent |
| [`tools/`](tools/README.md) | `simq`, the Chipyard simulation queue; one-off extraction utilities |
| [`planning/`](planning/README.md) | plans, decisions, and the measurements behind them |
| `merlin/` | the compiler framework, git submodule, **unforked** |
| `out/`, `results/`, `.venv/` | build artifacts, run records, the environment (gitignored) |

## Two standing constraints

**Do not depend on the sibling reference flow.** It is the reference for *structure* — the same
datapath, nearly the same programming model — but nothing here may include, link, or resolve a path
into it. Reimplement; take ideas, not code. Every *fact* the compiler needs is grounded in
`generators/gemmini/` instead: the headers, the spike model, the RTL Scala.

**One config artifact, two consumers.** `config/recipes/*.json` drives *both* compilation and
hardware generation — see [`config/README.md`](config/README.md).
