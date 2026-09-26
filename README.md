# npu-exploration

Define a model in PyTorch, compile it to Rocket-hosted MxGemmini RoCC instructions, run it, and grade
it bit for bit against the mxquant model of the same machine. **One ELF per kernel, any MX datatype.**

## Quickstart

Prerequisites: Linux x86_64, `conda` (miniconda is fine), git SSH access to
`ucb-bar/npu-exploration`, `ucb-bar/merlin`, `ucb-bar/gemmini` and `chloe-wong/microscaling-quant`
(optionally `Rakanic/MxGemmini-workspace` for silicon-cost numbers, `chooper1/MXQuant` for the
capture scripts and the legacy reference, and a CUDA GPU for the accuracy model), and ~10 GB of disk.

```bash
git clone --recurse-submodules git@github.com:ucb-bar/npu-exploration.git
cd npu-exploration
bash scripts/setup.sh                       # provisions EVERYTHING (idempotent; see below)
source scripts/env.sh                       # sets MERLIN_CHIPYARD, RISCV, PATH
.venv/bin/python run_kernel.py --kernel linear --config baseline
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
.venv/bin/python run_kernel.py --kernel linear --dtype fp4_e2m1 --config wide_acc
.venv/bin/python run_kernel.py --kernel linear --config wide_acc --models mxquant       # the model alone, no spike
.venv/bin/python run_kernel.py --kernel linear --config wide_acc --models all --gpus 0,1,2,3   # + perplexity
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

| flag | default | meaning |
|---|---|---|
| `--kernel` | `linear` | which kernel (`--list`) |
| `--config` | `baseline` | which hardware recipe (`--list`) |
| `--models` | `default` | which models run: `default` = reference, mxquant, spike, ppa, perf; `all` adds accuracy; or a comma list |
| `--dtype` | `fp8_e4m3` | MX operand format |
| `--m --k --h --n` | 64 | batch rows, in_features, hidden, out_features |
| `--tol` | 0.15 | pass threshold on relative Frobenius error vs fp32 |
| `--artifacts` | off | also write an RTL-replay bundle (C + `operands.npz`) |
| `--build-only` | off | stop at the ELF |
| `--per-stage-elf` | off | one ELF per matmul, intermediates carried by the host; the default fuses a chain or emits a graph as one ELF |
| `--legacy-mxquant` | off | grade with the previous reference (`grade/mxquant_ref.py`, MXQuant bundle extracted from the clone on first use); for the equivalence test, removed in the next PR |
| `--gpus --nsamples --model-id` | | accuracy model: GPUs to split the samples over, sample count, HF model |

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
 │ spike      LOWER: compiler/lower.py (app/mxgraph, mxhost) → command buffer│
 │            (fused chain | graph | per-stage)   → compiler/targets backend│
 │            mxgemm_emit / mxgraph_emit → main.c → ELF → spike             │
 │ mxquant    models/mxquant on mxq, fed the same wire operands             │
 │ ppa, perf  models/ppa, models/perf                                       │
 │ accuracy   models/accuracy: TinyLlama perplexity on the recipe's Scheme  │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            ▼
              grade/metrics + report → results/<run>/  → VERDICT · PPA · PERF · PPL
```

The recipe is the only source of the machine: `config/scheme.py` turns it into mxq's quantizer and
arithmetic for the mxquant and accuracy models, and `build_spike.py` turns the same JSON into the
functional model spike loads. The kernel is data (`kernels/registry.py`); the pipeline lowers it,
runs it, and grades the bits that came back against the mxquant model of the same recipe.

Inside `pipeline.run` the spike run and the three models that do not need its output run **at the
same time**: the mxquant model needs only the lowering's edges, ppa only the recipe, perf only the
stage shapes, so they start the moment the lowering has decided the edges and the grade joins them
after spike returns. The accuracy model runs after the grade and splits its samples over the GPUs
named in `--gpus`, one worker process per GPU. To run many kernel × recipe combinations at once,
launch several `run_kernel.py` processes; each run writes its own `results/<timestamp>_…/` directory.

## Entry points

| what you want | command |
|---|---|
| grade a kernel on a machine, every model at once | `run_kernel.py --kernel attention --config wide_acc` |
| the same, with TinyLlama perplexity for that machine | `run_kernel.py --kernel linear --config wide_acc --models all --gpus 0,1,2,3` |
| just the bits a machine must produce, no spike, seconds | `run_kernel.py --kernel mlp3 --config narrow_prod --models mxquant` |
| compile a kernel to an ELF for spike or the RTL build | `compile_kernel.py --kernel mlp3 --target mx_rocket` |
| compile a plain PyTorch module, no registry entry | `compile_kernel.py --module my.py:Block --input x.npy` |
| perplexity of one machine on its own | `python -m models.accuracy --config baseline --gpus 0,1,2,3` |
| silicon cost of one machine | `python -m models.ppa.ppa --config baseline` |
| predicted timeline of one matmul on it | `python -m models.perf.perf --config baseline --m 64 --k 64 --n 64` |
| build or list the per-recipe functional models | `python -m models.spike.build_spike --config R --force` |
| the claims, one test each | `tests/selftest_*.py` (17 of them) and `tests/test_recipe_drift.py` |
| environment | `bash scripts/setup.sh --check`, `source scripts/env.sh` |

Kernels by name: `linear`, `mlp2` … `mlp8`, `attention`, and `llama_attention` / `llama_mlp` once a
capture exists. Machines: `baseline`, `flat_acc4`, `wide_acc`, `narrow_prod`, or any recipe JSON.
Formats: every entry of `app/mxformats.py` on chains; `fp8_e4m3` only on graph kernels.
`compile_kernel.py` builds and writes `expected.npy` but runs nothing; running the `mx_rocket` ELF
on VCS or FPGA is the hardware team's step.

Less common:

| command | what it does |
|---|---|
| `python -m models.accuracy --config R --dry-run` | which layers the accuracy model would patch |
| `app/capture_llama_layer.py`, `app/capture_llama_tiles.py` | capture real TinyLlama tensors for the llama kernels (needs the MXQuant clone) |
| `baremetal/mxgemmini/gen/gen_*.py` | generators for the hand-written TinyLlama kernels |
| `rtl_exact/verify_rtl_exact.py`, `rtl_exact/make_fixture.py` | the frozen fixture and its verifier |
| `tools/extract_model.py` | one-off extraction from the gemmini tree |

Run everything with `.venv/bin/python` from the repo root after `source scripts/env.sh`.

## Where things are

```
run_kernel.py                 the exploration entry point (graded)
compile_kernel.py             the compile entry point (ELF + expected bits)
kernels/     registry.py spec.py trace.py            the kernel IR; --list shows what is registered; trace.py = PyTorch module -> KernelSpec
config/      recipe.py recipes/*.json scheme.py     one JSON = one machine; recipe → mxq
models/      reference/ mxquant/ spike/ ppa/ perf/ accuracy/   one folder per model of the machine
grade/       pipeline.py metrics.py report.py telemetry.py     run, compare, record
app/         mxiface mxgraph mxhost (lowering front half); mxformats mxwire mxlut (the wire);
             mxq_golden.py (operand quantizer, renamed in the next PR); mxmesh/; capture_*
compiler/    lower.py (KernelSpec -> command buffer); targets/mx_gemmini_rocket/ backend/{mxgemm_emit,mxgraph_emit,runner} runtime/
baremetal/   hand-written TinyLlama kernels and their generators
rtl_exact/   the frozen fixture and verifier
tests/       the self-tests        tools/ extraction        scripts/ setup.sh env.sh
merlin/  microscaling-quant/       submodules             MXQuant/   optional clone
out/  results/  .venv/  toolchain/ generated, gitignored
```

Each directory has its own README covering what it holds and what to do there.

| Dir | Contents |
|---|---|
| [`app/`](app/README.md) | MX quantization, host ops, the format table, codebooks, interface MLIR |
| [`kernels/`](kernels/README.md) | the kernel registry — a kernel is data, not code. **Add kernels here.** |
| [`baremetal/`](baremetal/README.md) | hand-written application kernels per target (TinyLlama on MxGemmini) |
| [`config/`](config/README.md) | hardware recipes: one JSON = one machine (`--config`); `scheme.py` maps a recipe onto mxq |
| [`models/`](models/README.md) | one folder per model of that machine: reference, mxquant, spike, ppa, perf, accuracy |
| [`compiler/`](compiler/README.md) | the lowering (`lower.py`) and the backend that emits, builds and runs the ELF |
| [`grade/`](grade/README.md) | run, compare, record — and what the verdict means |
| [`rtl_exact/`](rtl_exact/README.md) | the reference configuration that matches the hardware bit for bit |
| [`sim/`](sim/README.md) | RTL simulation and FPGA emulation substrates |
| [`tests/`](tests/README.md) | the self-tests, and which claim each one defends |
| [`scripts/`](scripts/README.md) | environment setup, and the errors it exists to prevent |
| [`tools/`](tools/README.md) | one-off extraction utilities |
| [`planning/`](planning/README.md) | plans, decisions, and the measurements behind them |
| `merlin/` | the compiler framework, git submodule, **unforked** |
| `microscaling-quant/` | mxq, the quantization library (block quantizers, the systolic arithmetic, `nn.patch`), git submodule pinned by SHA |
| `out/`, `results/`, `.venv/` | build artifacts, run records, the environment (gitignored) |

## Two standing constraints

**Do not depend on the sibling reference flow.** It is the reference for *structure* — the same
datapath, nearly the same programming model — but nothing here may include, link, or resolve a path
into it. Reimplement; take ideas, not code. Every *fact* the compiler needs is grounded in
`generators/gemmini/` instead: the headers, the spike model, the RTL Scala.

**One config artifact, two consumers.** `config/recipes/*.json` drives *both* compilation and
hardware generation — see [`config/README.md`](config/README.md).
