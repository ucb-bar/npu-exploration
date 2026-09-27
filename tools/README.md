# tools — one-off utilities

Scripts run by hand rather than by `run_kernel.py`. Nothing in the normal path imports them; if a
thing runs on every kernel it belongs in [`app/`](../app/README.md) instead. `simq` is the exception
to "occasionally" — it is the everyday way to drive RTL simulations.

| file | role |
|---|---|
| `extract_model.py` | pull a datapath model out of an upstream checkout into `app/mxmesh/`, so the repo does not depend on that checkout at run time |
| `simq.py` | queue Chipyard RTL simulation jobs (VCS, Xcelium or Verilator): elaborate each config once (serial, sbt holds a lock), run everything else up to `-j` at a time, pipelined. Run it as `tools/simq` |

## Using it

```bash
.venv/bin/python tools/extract_model.py --help
python3 tools/simq.py --help
```

## `simq` — the Chipyard simulation queue

`tools/simq` is a wrapper around `simq.py`; put `tools/` on your `PATH` and it is just `simq`.

It depends only on Chipyard. The checkout is found by walking up from `tools/` (npu-exploration lives
inside it), or set `CHIPYARD_ROOT`. The simulator is `SIM=vcs|xcelium|verilator` (or `$SIMQ_SIM`),
defaulting to the first of `vcs` / `xrun` / `verilator` on `PATH`; everything lives under that
`sims/<SIM>/`. Binary names, seed and waveform flags are the ones that simulator's Chipyard Makefile
uses, so a run is the command `make run-binary` would have issued. `EXTRA_SIM_FLAGS=` passes extra
plusargs, as in Chipyard; `RANDOM_SEED`, `USE_VPD` and `USE_FST` are honored from the environment.

Make-style, so it reads like the `make run-binary CONFIG=... BINARY=...` it replaces:

```bash
tools/simq build      CONFIG=MxGemminiRocketConfig                      # synchronous
tools/simq run        CONFIG=MxGemminiRocketConfig BINARY=mxl4          # queues, returns
tools/simq run-debug  CONFIG=MxGemminiRocketConfig BINARY=ladder.jobs JOBS=8
tools/simq run        CONFIG=... BINARY=mxl9 MAX_CYCLES=40000000 TIMEOUT=14400
tools/simq run        CONFIG=... BINARY=mxl4 --wait                     # block; has an exit code
```

**A run verb queues by default** — it returns in about a second and a background daemon does the
work, so you can keep appending. `--wait` blocks in this terminal instead; it is the only form with
a meaningful exit code, because queueing returns before the work happens. `build` is always
synchronous: one serial action, nothing to append to.

**`run-debug` dumps a waveform** beside each job's log, in the simulator's Chipyard default format
(VCS: `<job>.fsdb`, or `.vpd` with `USE_VPD=1`; Xcelium: `.vcd`; Verilator: `.vcd`, or `.fst` with
`USE_FST=1`) — the debug build exists to be looked at, so it dumps by default. `--no-waves` keeps
the debug build without the dump. Waveforms are
GB-scale and there is one per job with no cleanup, so give a large waved batch its own `OUT=`.

While a config elaborates, simq reports elapsed time and the current build phase every 60 s
(`HEARTBEAT=` or `--build-heartbeat`, `0` to silence). A full-format asym config takes many minutes
of Chisel plus a long simulator compile, and one line followed by silence is indistinguishable from a
hang — which is how it was first reported.

`CONFIG=` and `BINARY=` both take comma-separated lists and cross-multiply, so
`CONFIG=A,B BINARY=x,y` is four jobs over two elaborations. A `BINARY=` value ending `.jobs`,
`.txt` or `.list` is a FILE of binaries, one per line, `#` comments allowed. Variables:
`CONFIG BINARY JOBS TIMEOUT MAX_CYCLES OUT BINDIR MAKE_JOBS SIM EXTRA_SIM_FLAGS HEARTBEAT` — anything else in `KEY=VALUE` form is
an error rather than silently ignored, because a typo'd `TIMEOUT` is otherwise a four-hour surprise.

Every dash option still works and overrides a variable; the explicit job form remains for when the
cross product is not what you want:

```bash
python3 tools/simq.py MxGemminiRocketConfig:mxl4 MxDim32GemminiRocketConfig:mxl4
python3 tools/simq.py -c MxGemminiRocketConfig mxl0 mxl1 @more.jobs
```

Two pools, because only one part of the flow is serial: **elaboration** goes through sbt, which
holds a lock, so it runs one config at a time in a single thread; **runs** invoke the simulator directly
— no make, so no lock and no chance of a stray re-elaboration — up to `-j` at once. A config's jobs
are enqueued the moment its simulator exists, so config A's runs overlap config B's elaboration instead
of the machine idling through a build phase.

Bare binary names resolve against `out/baremetal/mx_rocket/` and the ISA suite's
`build_mx_rocket/bareMetalC/` (with or without the `-baremetal` suffix); `-b` adds more. A job that
produces no verdict at all — hang, trap, exhausted `+max-cycles` — is reported `NO-VERDICT` and is
**not** counted as a pass. Exit status is 0 only if every job passed.

### The queue

Watching and managing what you queued. (`add` is still accepted as an explicit prefix —
`add run-debug ...` — but it is now the default, so it is redundant.)

```bash
tools/simq run-debug CONFIG=MxGemminiRocketConfig BINARY=ladder.jobs   # returns in ~1s
tools/simq status            # queued / running / done, and which ones did not pass
tools/simq watch             # follow the daemon log live (ctrl-C stops watching, not the daemon)
tools/simq cancel <id>       # drop one queued job
tools/simq drain             # drop everything still queued; running jobs continue
tools/simq clear             # retire finished records, so `status` stops being a wall of history
tools/simq clear --logs      # also the per-job logs and their waveforms (this is the one that frees space)
tools/simq clear --all       # + the daemon log, build logs, and claims orphaned by a dead daemon
tools/simq clear --dry-run   # what would go, and how many bytes
tools/simq stop              # graceful: claim no new jobs, exit once in-flight runs finish
tools/simq stop --now        # hard: SIGTERM the daemon AND its running simulations
```

Every event carries a **wall-clock** timestamp, not elapsed-since-start: `status` shows when each
job was queued, started and ended, and the daemon log is prefixed `[HH:MM:SS]` so it lines up with
a build log or a waveform read days later. `clear` never touches queued or running jobs, and it **truncates** the daemon log rather than
unlinking it — the daemon holds that file open for its whole life, so removing it leaves the daemon
writing to a deleted inode while `watch` follows a new empty file forever. Logs belonging to a
currently-running job are skipped for the same reason. `watch` reports daemon and queue state before
it blocks, and if it finds an empty log beside a live daemon it follows the orphaned inode instead
of pretending nothing is happening.

`stop` leaves queued jobs queued — `simq drain` discards them, or start the daemon again to pick up
where it left off. After `stop --now`, the jobs that were mid-run stay in `running/` and `status`
flags them as orphans with the `mv` to requeue them; nothing is silently lost.

The first `add` starts the daemon automatically (`--no-start` to prevent it, `simq daemon` to start
it by hand). The daemon holds the same discipline as the synchronous mode — one elaboration at a
time, `--jobs` runs — so appending a job for a not-yet-built config elaborates it while the
already-built configs keep running. It exits after `--idle-exit` seconds idle (default 3600, `0` =
never).

Per-job options are stored in the job file, so `add run` and `add run-debug` coexist in one queue and
each job gets the build it asked for. State lives in `sims/<SIM>/.simq/{pending,running,done}` (one
queue per simulator) as one
JSON file per job; claiming a job is a single atomic `rename`, so two daemons cannot double-run one
and a crashed daemon leaves its claim visible in `running/` rather than losing it.

**Elaboration is serialized across PROCESSES, not just within one queue.** `simq build CONFIG=X`
in one terminal and `simq run CONFIG=X ...` in another do not both reach `make`: the second waits on
an `flock` (naming the pid that holds it), then re-checks and finds the simulator the first just
produced, so it reuses it instead of elaborating again. The simulator binary is the only state — there is
no database — which is what makes `build` and `run` compose across invocations the way make does.

**It refuses a stale simulator.** If the elaborated simulator is older than the newest `.scala` under
`generators/gemmini/src/main/scala`, the config's jobs are marked `NO-BUILD` rather than run — that
exact trap (a five-day-old simulator reused through a whole bisection, its results read as properties of
the hardware) is why the check exists. `--rebuild-if-stale` re-elaborates just the configs that are
behind; `--rebuild` re-elaborates all of them; `--allow-stale` runs the old one deliberately.

`baremetal/mxgemmini/run_ladder_rtl.sh` is the ladder-specific shell version of the same idea; use
`simq.py` when the configs or binaries vary. `baremetal/mxgemmini/ladder.jobs` is the ladder as a
job file.

What is extracted is then pinned by `tests/selftest_extracted.py`, which fails if `app/mxmesh/` has
drifted from the source it came from. Re-extract, don't hand-edit.

See [`../README.md`](../README.md) for install and the run command.
