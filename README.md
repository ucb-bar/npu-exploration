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
toolchain (`spike`, `riscv64-unknown-elf-gcc`, `dtc` — binaries from the `ucb-bar` conda channel,
no chipyard build needed), the gemmini hardware sources, and a stock `libgemmini.so` built with
the toolchain's own g++. `bash scripts/setup.sh --check` prints PASS/FAIL for every requirement.
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
| [`config/`](config/README.md) | hardware recipes: one JSON = one machine (`--config`) |
| [`compiler/`](compiler/README.md) | the out-of-tree merlin target: contract + backend |
| [`grade/`](grade/README.md) | run, compare, record — and what the verdict means |
| [`rtl_exact/`](rtl_exact/README.md) | the reference configuration that matches the hardware bit for bit |
| [`sim/`](sim/README.md) | RTL simulation and FPGA emulation substrates |
| [`tests/`](tests/README.md) | the self-tests, and which claim each one defends |
| [`scripts/`](scripts/README.md) | environment setup, and the errors it exists to prevent |
| [`tools/`](tools/README.md) | one-off extraction utilities |
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
