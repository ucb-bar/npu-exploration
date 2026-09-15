# scripts — environment setup

One script, and the four failures it exists to prevent. Everything here runs before any work does.

| file | role |
|---|---|
| `env.sh` | sets `MERLIN_CHIPYARD`, `RISCV` and `PATH` from a chipyard checkout |

## Using it

```bash
source scripts/env.sh                      # uses the default chipyard location
source scripts/env.sh /path/to/chipyard    # or point it at yours
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
.venv/bin/python -m config.build_spike --config <recipe> --force
```

The same code is also what a baremetal test prints when it was built without `-DSPIKE_SIM` and took
the MMIO command-mimic path instead of real RoCC. In `gemmini-rocc-tests`, that define comes from
`RUNNER` containing `spike`, not from the target name.

**`Unable to load extlib … GLIBCXX_3.4.32 not found`** — the model was built with a newer g++ than
spike, and spike's `DT_RPATH` outranks `LD_LIBRARY_PATH`, so no environment variable fixes it.
Rebuild it with an older g++; `MX_HOST_GXX` selects one.

See [`../README.md`](../README.md) for install and the run command.
