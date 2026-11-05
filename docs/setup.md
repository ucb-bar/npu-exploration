# Setup

Prerequisites: Linux x86_64, `conda` (miniconda is fine), about 10 GB of disk, and git access to
`ucb-bar/npu-exploration`, `ucb-bar/merlin`, `ucb-bar/gemmini` and `chloe-wong/microscaling-quant`.
Optional: `Rakanic/MxGemmini-workspace` for the cost step.

```bash
git clone --recurse-submodules git@github.com:ucb-bar/npu-exploration.git
cd npu-exploration
bash scripts/setup.sh            # every phase, then the doctor
bash scripts/setup.sh --check    # doctor only: PASS/FAIL per requirement
source scripts/env.sh            # sets MERLIN_CHIPYARD, RISCV, PATH
```

Without SSH keys, switch the submodules to HTTPS first:

```bash
git config submodule.merlin.url https://github.com/ucb-bar/merlin.git
git config submodule.microscaling-quant.url https://github.com/chloe-wong/microscaling-quant.git
git submodule update --init compiler/merlin compiler/mx-gemmini-mlir microscaling-quant
```

Already have a chipyard tree? Skip the toolchain phases and point at it:
`source scripts/env.sh /path/to/chipyard`. The hardware sources then come from that tree, so the
model and the spike that loads it cannot drift apart.

## What setup.sh provisions

Each phase checks its result and skips when it is already there, so re-running after a failure
resumes. `--phase <p>` runs one phase; `--root` puts the toolchain somewhere other than `<repo>/toolchain`.

| phase | provides | needed by |
|---|---|---|
| `python` | `.venv` with `requirements.txt` | everything |
| `mxq` | the `microscaling-quant` submodule at its pinned SHA | quantize |
| `toolchain` | `riscv64-unknown-elf-gcc`, `dtc`, host g++ (conda, `ucb-bar` channel) | compile |
| `spike` | `riscv-isa-sim` built from source into `$RISCV` | evaluate |
| `gemmini` | `generators/gemmini` with `libgemmini` and `gemmini-rocc-tests` | compile, evaluate |
| `libgemmini` | the stock `libgemmini.so` spike loads, built with the toolchain's own g++ | evaluate |
| `ppa` | `../MxGemmini-workspace` (optional; `--no-ppa` skips it) | cost |
| `mxquant` | optional (`--with-mxquant`): the MXQuant clone | TinyLlama capture scripts, `tests/verify_rtl_exact.py` |

## Troubleshooting

**`$RISCV is unset`.** Chipyard's own `env.sh` only activates conda. Use `scripts/env.sh`.

**`Failed to run dtc`.** spike shells out to the device-tree compiler; `scripts/env.sh` puts it on `PATH`.

**`*** FAILED *** (tohost = 1337)`.** An unhandled trap, nearly always a stale functional model.
Force a rebuild for your hardware config:

```bash
.venv/bin/python -m models.spike.build_spike --hw <config> --force
```

**`Unable to load extlib ... GLIBCXX_3.4.32 not found`.** The model was built with a newer g++ than
spike. spike's `DT_RPATH` outranks `LD_LIBRARY_PATH`, so rebuild with an older g++ (`MX_HOST_GXX` selects one).
