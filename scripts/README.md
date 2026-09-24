# scripts — environment setup

Two scripts, and the four failures they exist to prevent. Everything here runs before any work does.

| file | role |
|---|---|
| `setup.sh` | provisions everything a fresh clone needs (toolchain, sources, python env, the mxq submodule; MXQuant optional) |
| `env.sh` | sets `MERLIN_CHIPYARD`, `RISCV` and `PATH` from a chipyard(-shaped) tree |

## setup.sh

```bash
bash scripts/setup.sh                 # all phases, then the doctor
bash scripts/setup.sh --check        # doctor only: PASS/FAIL per requirement
bash scripts/setup.sh --phase <p>    # one phase (see table)
```

Idempotent: each phase checks its postcondition and skips when satisfied, so re-running after a
failure resumes. The default root is `<repo>/toolchain`, a chipyard-*shaped* tree
(`.conda-env/` + `generators/gemmini/`) that `env.sh` picks up with no arguments; `--root` swaps
in any location, including a real chipyard checkout.

| phase | provides | required by |
|---|---|---|
| `python` | `.venv` + `requirements.txt` installed | every `.venv/bin/python` command |
| `mxq` | `<repo>/microscaling-quant` submodule checked out at the pinned SHA | every model (`models/`), `app/mxq_golden.py` through `models/mxquant/block.py` |
| `mxquant` | **optional** (`--with-mxquant`): `<repo>/MXQuant` clone (or symlink via `--mxquant <dir>`), `origin/chloe-branch-all` fetched | the capture scripts, `grade/mxquant_ref.py` (`--legacy-mxquant`), `tests/selftest_block.py --update` |
| `toolchain` | conda env with `riscv64-unknown-elf-gcc`, `dtc` and a host g++ (binaries from the `ucb-bar` channel — no chipyard build) | the runner gate; spike shells out to `dtc` |
| `spike` | `riscv-isa-sim` built from source into `$RISCV` (spike is not packaged anywhere — chipyard builds it from source too; override: `--spike-ref`) | the execution substrate |
| `gemmini` | `generators/gemmini` @ `gemmini-mx-cleanup` (override: `--gemmini-ref`), with `libgemmini` + `gemmini-rocc-tests` submodules | `models/spike/build_spike.py`, the ELF harness |
| `libgemmini` | stock `libgemmini.so`, built with the toolchain env's **own** g++ so its libstdc++ can never be newer than spike's | spike `--extlib` |
| `ppa` | `../MxGemmini-workspace` clone (optional, `--no-ppa` skips; failures warn and continue) | `models/ppa/ppa.py` silicon-cost numbers |

## env.sh

```bash
source scripts/env.sh                      # default: <repo>/toolchain (what setup.sh builds)
source scripts/env.sh /path/to/chipyard    # or point it at a real chipyard tree
```

Two things here cost real debugging time, which is why this is a script and not a line in a README:
chipyard's own `env.sh` does **not** set `$RISCV` (it only activates conda — the toolchain is at
`.conda-env/riscv-tools`), and `spike` shells out to `dtc`, which is not on `PATH` by default.

## Troubleshooting

**`$RISCV is unset`** — chipyard's `env.sh` only activates conda. Use `scripts/env.sh`.

**`Failed to run dtc`** — spike shells out to the device-tree compiler in the chipyard conda env;
`scripts/env.sh` puts it on `PATH`.

**`*** FAILED *** (tohost = 1337)`** — an unhandled trap, nearly always a stale functional model.
Rebuild and install it from gemmini (`make clean` first — its Makefile misses header changes), or
force a per-recipe rebuild:

```bash
.venv/bin/python -m models.spike.build_spike --config <recipe> --force
```

The same code is also what a baremetal test prints when it was built without `-DSPIKE_SIM` and took
the MMIO command-mimic path instead of real RoCC. In `gemmini-rocc-tests`, that define comes from
`RUNNER` containing `spike`, not from the target name.

**`Unable to load extlib … GLIBCXX_3.4.32 not found`** — the model was built with a newer g++ than
spike, and spike's `DT_RPATH` outranks `LD_LIBRARY_PATH`, so no environment variable fixes it.
Rebuild it with an older g++; `MX_HOST_GXX` selects one.

See [`../README.md`](../README.md) for install and the run command.
