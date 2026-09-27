#!/usr/bin/env python3
"""simq -- a pipelined queue for Chipyard RTL simulation jobs (VCS, Xcelium or Verilator).

Submit N (config, binary) jobs; simq elaborates each config ONCE, serially, and runs everything
else up to `--jobs` at a time. Results print the moment each job finishes, with an ordered table
at the end.

WHY THE TWO POOLS. Chisel elaboration goes through sbt, which holds a lock: two concurrent
elaborations collide. Nothing else in the flow is serial. `make run-binary` makes that awkward
because the simulator is an implicit PREREQUISITE of the run target -- fire eight of those at a stale
simulator and all eight try to elaborate. So simq splits the two:

  * BUILD  `make [debug] CONFIG=<cfg>` -- one at a time, in a single builder thread.
  * RUN    the simulator binary invoked DIRECTLY -- `--jobs` at a time, touching no lock and no make.

The split is also what makes it pipeline: as soon as config A's simulator exists its runs start, while
config B is still elaborating. A build-everything-then-run-everything script leaves the machine
idle through the whole build phase.

    simq build      CONFIG=MxGemminiRocketConfig                       # synchronous, one action
    simq run        CONFIG=MxGemminiRocketConfig BINARY=mxl4           # QUEUES, prompt comes back
    simq run-debug  CONFIG=MxGemminiRocketConfig BINARY=ladder.jobs JOBS=8
    simq run        CONFIG=... BINARY=mxl9 MAX_CYCLES=40000000 TIMEOUT=14400
    simq run        CONFIG=... BINARY=mxl4 --wait                      # block, for a script
    simq status | simq watch | simq clear | simq stop

A RUN VERB QUEUES BY DEFAULT: the daemon does the work and you keep typing. `--wait` blocks in this
terminal instead, and is the only form with a meaningful exit code -- `add`/queueing returns before
the work happens, so it cannot report a verdict.

`make`-style variables, so it reads like the `make run-binary CONFIG=... BINARY=...` it replaces:

    CONFIG=      one config, or several comma-separated
    BINARY=      one binary, several comma-separated, or a `.jobs`/`.txt` file of them
    JOBS=        concurrent runs (default 8)
    TIMEOUT=     per-run wall seconds (default 3600)
    MAX_CYCLES=  +max-cycles (default 10000000)
    OUT=         log directory
    SIM=         vcs | xcelium | verilator -- which `sims/<SIM>` directory to use
    EXTRA_SIM_FLAGS=  extra plusargs for every run, as in Chipyard's `make run-binary`

CONFIG x BINARY is a cross product, so `CONFIG=A,B BINARY=x,y` is four jobs over two elaborations.
Every dash option below still works and overrides a variable.

WHERE THINGS ARE. Everything is relative to the Chipyard checkout, which is found by walking up
from this script (npu-exploration lives inside it) or taken from `$CHIPYARD_ROOT`. The simulator
defaults to the first of vcs / xrun / verilator on PATH; `SIM=` or `$SIMQ_SIM` picks one. Binary
names, run plusargs, seed and waveform flags follow that simulator's own `sims/<SIM>` Makefile, so
a simq run is the same command `make run-binary` would have issued.

`run-debug` uses the debug simulator AND dumps a waveform next to the job's log -- there is no
reason to pay for the debug build and not look at it. `--no-waves` keeps the debug build without
the dump. `run-waves` is a synonym for `run-debug`, kept because it says what it does. The format
is the simulator's Chipyard default (VCS: FSDB, or VPD with USE_VPD=1; Xcelium: VCD; Verilator:
VCD, or FST with USE_FST=1). A waveform is GB-scale and there is one per job with no cleanup, so
point a big waved batch at its own `OUT=`.

**By default a simulator older than its Scala is re-elaborated** (`--no-rebuild-if-stale` to stop
that, `--allow-stale` to run the old one anyway, `--rebuild` to re-elaborate regardless).

The low-level form still takes explicit jobs, for when the cross product is not what you want:

    simq.py MxGemminiRocketConfig:mxl4 MxDim32GemminiRocketConfig:mxl4
    simq.py -c MxGemminiRocketConfig mxl0 mxl1 @more.jobs

A job is `CONFIG:BINARY`, or a bare `BINARY` when `-c/--config` gives the default, or `@file` to
read one job per line (`#` comments allowed). BINARY is an absolute path, or a name looked up in
`--bindir` (repeatable; the default list covers the npu-exploration kernels and the ISA suite).

Exit status is 0 only if every job reached a PASS verdict. A job that produced no verdict at all --
hang, trap, exhausted `+max-cycles` -- is reported NO-VERDICT and is NOT a pass.
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

NPU = Path(__file__).resolve().parent.parent            #: this npu-exploration checkout


def _is_chipyard(d: Path) -> bool:
    return (d / "common.mk").is_file() and (d / "variables.mk").is_file() and (d / "sims").is_dir()


def find_chipyard() -> Path:
    """`$CHIPYARD_ROOT`, else the nearest Chipyard checkout above this script or the cwd."""
    env = os.environ.get("CHIPYARD_ROOT")
    if env:
        if not _is_chipyard(Path(env)):
            raise SystemExit(f"CHIPYARD_ROOT={env} is not a Chipyard checkout "
                             "(no common.mk / variables.mk / sims/)")
        return Path(env).resolve()
    for start in (NPU, Path.cwd().resolve()):
        for d in (start, *start.parents):
            if _is_chipyard(d):
                return d
    raise SystemExit("Chipyard checkout not found above this script or the cwd; "
                     "set CHIPYARD_ROOT=<chipyard dir>")


REPO = find_chipyard()
DRAMSIM = REPO / "generators" / "testchipip" / "src" / "main" / "resources" / "dramsim2_ini"
#: Where a bare binary name is looked up, in order.
DEFAULT_BINDIRS = [
    NPU / "out/baremetal/mx_rocket",
    REPO / "generators/gemmini/software/gemmini-rocc-tests/build_mx_rocket/bareMetalC",
]


@dataclass(frozen=True)
class Simulator:
    """What differs between Chipyard's `sims/<name>` flows, transcribed from their Makefiles.

    prefix  `sim_prefix`: the binary is `<prefix>-<MODEL_PACKAGE>-<CONFIG>[-debug]`
    tool    the executable whose presence on PATH makes this the auto-detected default
    """
    name: str
    prefix: str
    tool: str

    def seed_flag(self) -> str:
        """`SEED_FLAG`: RANDOM_SEED pins the seed, otherwise the simulator's own automatic seed."""
        seed = os.environ.get("RANDOM_SEED")
        if self.name == "verilator":
            return f"+verilator+seed+{seed}" if seed else ""
        return f"+ntb_random_seed={seed}" if seed else "+ntb_random_seed_automatic"

    def waveform_flag(self, stem: Path) -> str:
        """`get_waveform_flag`, including the USE_VPD / USE_FST switches."""
        if self.name == "vcs":
            return (f"+vcdplusfile={stem}.vpd" if os.environ.get("USE_VPD")
                    else f"+fsdbfile={stem}.fsdb")
        if self.name == "verilator":
            return f"+vcdfile={stem}.{'vcd' if os.environ.get('USE_FST', '0') == '0' else 'fst'}"
        return f"+vcdfile={stem}.vcd"


SIMULATORS = {s.name: s for s in (Simulator("vcs", "simv", "vcs"),
                                  Simulator("xcelium", "simx", "xrun"),
                                  Simulator("verilator", "simulator", "verilator"))}


def pick_simulator(name: str | None) -> Simulator:
    """SIM= / $SIMQ_SIM, else the first simulator on PATH whose sims/ directory exists."""
    import shutil
    name = name or os.environ.get("SIMQ_SIM")
    if name:
        if name not in SIMULATORS:
            raise SystemExit(f"SIM={name!r}: choose from {', '.join(SIMULATORS)}")
        return SIMULATORS[name]
    for s in SIMULATORS.values():
        if (REPO / "sims" / s.name).is_dir() and shutil.which(s.tool):
            return s
    raise SystemExit(f"no simulator found on PATH ({', '.join(s.tool for s in SIMULATORS.values())}); "
                     "set SIM=<vcs|xcelium|verilator>")


#: Set by `configure()` once the simulator is known; everything below lives under its sims/ dir.
SIM: Simulator
SIM_DIR: Path
BUILD_LOCK: Path
SPOOL: Path
PENDING: Path
RUNNING: Path
DONE: Path
DAEMON_LOG: Path
DAEMON_PID: Path


def configure(sim: Simulator) -> None:
    global SIM, SIM_DIR, BUILD_LOCK, SPOOL, PENDING, RUNNING, DONE, DAEMON_LOG, DAEMON_PID
    SIM, SIM_DIR = sim, REPO / "sims" / sim.name
    BUILD_LOCK = SIM_DIR / ".simq-build.lock"
    SPOOL = SIM_DIR / ".simq"
    PENDING, RUNNING, DONE = SPOOL / "pending", SPOOL / "running", SPOOL / "done"
    DAEMON_LOG = SPOOL / "daemon.log"
    DAEMON_PID = SPOOL / "daemon.pid"
#: The ISA suite's ELFs carry this suffix; a bare name is tried both ways.
BIN_SUFFIXES = ["", "-baremetal"]

PASS_RE = re.compile(r"(^|\s)PASSED")
FAIL_RE = re.compile(r"FAILED")
#: Lines worth echoing next to a verdict. Kept deliberately narrow -- a summary table is useless
#: if one runaway line pushes the verdict off screen.
DETAIL_RE = re.compile(
    r"mesh\s+[^:\n]*:\s*\d+/\d+[^\n,]*(?:,\s*\d+/\d+\s*scales differ)?"
    r"|Scale\[\d+\]\[\d+\], Got: \S+, Exp: \S+"
    r"|\d+ code, \d+ scale mismatches"
    r"|TRAP cause=\d+ epc=\S+"
)


# --- job model ---------------------------------------------------------------------------------

@dataclass
class Job:
    config: str
    binary: Path
    name: str                      #: display name, = the binary's stem
    log: Path = field(init=False)
    verdict: str = "PENDING"
    detail: str = ""
    seconds: float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.config, str(self.binary))


@dataclass
class ConfigState:
    name: str
    simv: Path | None = None
    ready: bool = False
    error: str = ""
    seconds: float = 0.0


def hhmmss(t: float | None) -> str:
    """Wall clock, or blanks of the same width so columns stay aligned."""
    return time.strftime("%H:%M:%S", time.localtime(t)) if t else "        "


def ago(t: float | None) -> str:
    """How long ago, in the largest unit that stays readable."""
    if not t:
        return "?"
    d = max(time.time() - t, 0)
    if d < 90:
        return f"{d:.0f}s"
    if d < 5400:
        return f"{d / 60:.0f}m"
    if d < 172800:
        return f"{d / 3600:.1f}h"
    return f"{d / 86400:.1f}d"


class Printer:
    """Serialized line output. One `print` per event so parallel workers interleave by LINE.

    The prefix is WALL CLOCK, not elapsed-since-start: the daemon log is long-lived and gets read
    days later next to a build log or a waveform, and `[  4821s]` cannot be lined up with anything.
    Per-event durations are on the lines that have them.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.t0 = time.monotonic()

    def __call__(self, msg: str) -> None:
        with self._lock:
            print(f"[{hhmmss(time.time())}] {msg}", flush=True)


# --- resolution --------------------------------------------------------------------------------

def resolve_binary(spec: str, bindirs: list[Path]) -> Path:
    p = Path(spec)
    if p.is_absolute() or spec.startswith("."):
        if p.is_file():
            return p.resolve()
        raise SystemExit(f"binary not found: {spec}")
    for d in bindirs:
        for suf in BIN_SUFFIXES:
            cand = d / (spec + suf)
            if cand.is_file():
                return cand.resolve()
    tried = ", ".join(str(d) for d in bindirs)
    raise SystemExit(f"binary {spec!r} not found under: {tried}")


def parse_jobs(specs: list[str], default_config: str | None, bindirs: list[Path]) -> list[Job]:
    """Expand the spec list into deduplicated jobs, preserving first-appearance order."""
    out: list[Job] = []
    seen: set[tuple[str, str]] = set()

    def add(spec: str) -> None:
        spec = spec.strip()
        if not spec or spec.startswith("#"):
            return
        if spec.startswith("@"):
            f = Path(spec[1:])
            if not f.is_file():
                raise SystemExit(f"job file not found: {f}")
            for line in f.read_text().splitlines():
                add(line.split("#", 1)[0])
            return
        # A Windows-style drive letter is not a concern here; the only ':' is the config separator.
        if ":" in spec:
            cfg, _, binspec = spec.partition(":")
        else:
            if not default_config:
                raise SystemExit(f"{spec!r} has no config and -c/--config was not given")
            cfg, binspec = default_config, spec
        cfg, binspec = cfg.strip(), binspec.strip()
        if not cfg or not binspec:
            raise SystemExit(f"malformed job spec: {spec!r}")
        b = resolve_binary(binspec, bindirs)
        j = Job(config=cfg, binary=b, name=b.name)
        if j.key in seen:
            return                      # same config+binary twice would collide on its log
        seen.add(j.key)
        out.append(j)

    for s in specs:
        add(s)
    return out


def newest_scala(dirs: list[Path]) -> tuple[Path | None, float]:
    """The most recently modified .scala under `dirs`, for the staleness check."""
    newest, mtime = None, 0.0
    for d in dirs:
        if not d.is_dir():
            continue
        for f in d.rglob("*.scala"):
            m = f.stat().st_mtime
            if m > mtime:
                newest, mtime = f, m
    return newest, mtime


def check_staleness(cfg_name: str, simv: Path, args) -> str | None:
    """Refuse to run RTL older than the Scala it came from.

    This exists because it actually happened: a simulator five days older than the source was reused for
    a whole bisection, and the results were read as properties of the hardware. Reuse-by-existence
    is the right default for speed and the wrong one for correctness, so the tie-break is an
    explicit flag rather than a guess.
    """
    if args.allow_stale or args.rebuild:
        return None
    newest, m = newest_scala(args.scala_dir)
    if newest is None:
        return None
    if simv.stat().st_mtime >= m:
        return None
    return (f"{simv.name} was built {time.strftime('%Y-%m-%d %H:%M', time.localtime(simv.stat().st_mtime))}, "
            f"but {newest.relative_to(REPO) if str(newest).startswith(str(REPO)) else newest} "
            f"changed {time.strftime('%Y-%m-%d %H:%M', time.localtime(m))}")


def find_simv(config: str, debug: bool) -> Path | None:
    """The elaborated simulator for `config`, if it exists. MODEL_PACKAGE is not assumed."""
    suffix = "-debug" if debug else ""
    cands = [p for p in SIM_DIR.glob(f"{SIM.prefix}-*-{config}{suffix}")
             if p.is_file() and os.access(p, os.X_OK)
             and (debug or not p.name.endswith("-debug"))]
    return sorted(cands)[0] if cands else None


# --- the two workers ---------------------------------------------------------------------------

#: Elaboration is serialized ACROSS PROCESSES too, not just across this queue's builder thread.
#: Without it, `simq build CONFIG=X` in one terminal and `simq run CONFIG=X ...` in another both
#: reach `make` and collide on the sbt lock -- which fails the build rather than merely slowing it.
#: The lock file is BUILD_LOCK, set by `configure()`.


class build_lock:
    """Exclusive flock around an elaboration, announcing itself if it has to wait."""

    def __init__(self, name: str, say: Printer):
        self.name, self.say, self.fh = name, say, None

    def __enter__(self):
        import fcntl
        BUILD_LOCK.touch(exist_ok=True)
        self.fh = BUILD_LOCK.open("r+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = self.fh.read().strip() or "another simq"
            self.say(f"BUILD  {self.name:<34} waiting for {holder} to finish elaborating")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"pid {os.getpid()} building {self.name}\n")
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        import fcntl
        try:
            self.fh.seek(0)
            self.fh.truncate()
            self.fh.flush()
            fcntl.flock(self.fh, fcntl.LOCK_UN)
        finally:
            self.fh.close()
        return False


def _decide(cfg: ConfigState, args, say: Printer, quiet: bool = False) -> tuple[str, object]:
    """('reuse', simv) | ('refuse', msg) | ('build', reason). Pure inspection of the filesystem --
    the simulator binary IS the state, which is what makes `build` then `run` work across invocations."""
    existing = find_simv(cfg.name, args.debug)
    if not existing:
        return ("build", "no simulator")
    if args.rebuild:
        return ("build", "--rebuild")
    stale = check_staleness(cfg.name, existing, args)
    if stale and args.rebuild_if_stale:
        return ("build", stale)
    if stale:
        return ("refuse", stale)
    return ("reuse", existing)


def elaborate(cfg: ConfigState, args, out: Path, say: Printer) -> None:
    """`make [debug] CONFIG=<cfg>` in SIM_DIR, under a cross-process lock: sbt holds one of its own."""
    action, info = _decide(cfg, args, say)
    if action == "refuse":
        cfg.error = f"STALE simulator -- {info}"
        say(f"BUILD  {cfg.name:<34} REFUSED: {info}")
        say(f"       {'':<34} -> --rebuild-if-stale to re-elaborate, --allow-stale to run it anyway")
        return
    if action == "reuse":
        simv = info                                      # type: ignore[assignment]
        cfg.simv, cfg.ready = simv, True                 # type: ignore[assignment]
        age = time.strftime("%Y-%m-%d %H:%M", time.localtime(simv.stat().st_mtime))  # type: ignore[union-attr]
        say(f"BUILD  {cfg.name:<34} reuse {simv.name}  (built {age})")               # type: ignore[union-attr]
        return
    if info != "no simulator":
        say(f"BUILD  {cfg.name:<34} re-elaborating -- {info}")

    with build_lock(cfg.name, say):
        # Re-decide now that we hold the lock: another simq may have built exactly what we want
        # while we waited. Skipped under --rebuild, where the caller asked for the work explicitly.
        if not args.rebuild:
            action2, info2 = _decide(cfg, args, say)
            if action2 == "reuse":
                simv = info2                              # type: ignore[assignment]
                cfg.simv, cfg.ready = simv, True          # type: ignore[assignment]
                say(f"BUILD  {cfg.name:<34} built by another simq while waiting -> "
                    f"{simv.name}")                       # type: ignore[union-attr]
                return
        _make(cfg, args, out, say)


#: sbt/Chisel/simulator-compile noise that says nothing about progress. A heartbeat showing "]" or a stack of
#: warnings is worse than no heartbeat, because it looks like output without being information.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_DULL = re.compile(r"^\W*$|^\[?\s*warn|warning:|^\s*\^|^\[?\s*$")


def _build_phase(log: Path) -> str:
    """The most recent line of the build log that actually names a phase."""
    try:
        with log.open("rb") as fh:                     # tail without reading the whole file
            fh.seek(0, 2)
            fh.seek(max(fh.tell() - 65536, 0))
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return ""
    for raw in reversed(lines):
        line = _ANSI.sub("", raw).strip()
        if line and not _DULL.match(line):
            return line[:96]
    return ""


def _make(cfg: ConfigState, args, out: Path, say: Printer) -> None:
    target = ["debug"] if args.debug else []
    cmd = ["make", f"-j{args.make_jobs}", *target, f"CONFIG={cfg.name}"]
    log = out / f"build-{cfg.name}.log"
    say(f"BUILD  {cfg.name:<34} elaborating ({' '.join(cmd)})")
    t0 = time.monotonic()
    # A full-format elaboration plus the simulator compile runs for many minutes. Printing one line and
    # then nothing is indistinguishable from a hang -- which is exactly how it was reported. So
    # poll, and say both how long it has been and which phase the log is in.
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=SIM_DIR, stdout=fh, stderr=subprocess.STDOUT)
        beat = max(args.build_heartbeat, 0)
        next_beat = time.monotonic() + beat
        while proc.poll() is None:
            time.sleep(1.0)
            if beat and time.monotonic() >= next_beat:
                next_beat += beat
                mins = (time.monotonic() - t0) / 60.0
                phase = _build_phase(log)
                say(f"BUILD  {cfg.name:<34} still elaborating, {mins:.1f} min"
                    + (f" -- {phase}" if phase else ""))
        rc = proc.returncode
    cfg.seconds = time.monotonic() - t0

    simv = find_simv(cfg.name, args.debug)
    if rc != 0 or simv is None:
        # A nonzero make with the simulator present is still a failure worth surfacing, but an absent
        # simulator is the one that must stop this config's runs -- they would otherwise all NO-VERDICT.
        cfg.error = f"make rc={rc}" + ("" if simv else ", no simulator produced")
        say(f"BUILD  {cfg.name:<34} FAILED in {cfg.seconds:.0f}s -- {cfg.error}  ({log})")
        return
    cfg.simv, cfg.ready = simv, True
    say(f"BUILD  {cfg.name:<34} ok in {cfg.seconds:.0f}s -> {simv.name}")


def run_job(job: Job, cfg: ConfigState, args, say: Printer) -> None:
    """Invoke the simulator DIRECTLY -- no make, so no lock and no chance of a stray re-elaboration."""
    assert cfg.simv is not None
    # The same argument order as Chipyard's run-binary recipe: PERMISSIVE_ON, SIM_FLAGS,
    # EXTRA_SIM_FLAGS, SEED_FLAG, loadmem, [VERBOSE_FLAGS], [waveform], PERMISSIVE_OFF, binary.
    cmd = [
        f"./{cfg.simv.name}", "+permissive",
        "+dramsim", f"+dramsim_ini_dir={DRAMSIM}",
        f"+max-cycles={args.max_cycles}",
        *shlex.split(args.extra_sim_flags or ""),
        SIM.seed_flag(),
        f"+loadmem={job.binary}",
    ]
    if args.verbose:
        cmd.append("+verbose")
    if args.waves:
        cmd.append(SIM.waveform_flag(job.log.with_suffix("")))
    cmd += ["+permissive-off", str(job.binary)]
    cmd = [c for c in cmd if c]                    # an unset seed flag contributes nothing

    t0 = time.monotonic()
    with job.log.open("w") as fh:
        fh.write(f"# {' '.join(shlex.quote(c) for c in cmd)}\n")
        fh.flush()
        try:
            subprocess.run(cmd, cwd=SIM_DIR, stdout=fh, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            fh.write(f"\n# simq: TIMEOUT after {args.timeout}s\n")
    job.seconds = time.monotonic() - t0
    classify(job)
    say(f"RUN    {job.config}/{job.name:<26} {job.verdict:<11} {job.seconds:5.0f}s  {job.detail}")


def classify(job: Job) -> None:
    """FAIL is checked BEFORE pass: a log carrying both an intermediate failure and a later
    summary line must not be read as a pass. No verdict at all is its own outcome."""
    try:
        text = job.log.read_text(errors="replace")
    except OSError:
        job.verdict, job.detail = "NO-VERDICT", "no log"
        return
    body = text.split("\n", 1)[1] if text.startswith("#") else text   # drop the command echo
    hits = DETAIL_RE.findall(body)
    job.detail = " | ".join(h.strip() for h in hits[:2])[:88]
    if "simq: TIMEOUT" in text:
        job.verdict = "TIMEOUT"
    elif FAIL_RE.search(body):
        job.verdict = "FAIL"
    elif PASS_RE.search(body):
        job.verdict = "PASS"
    else:
        job.verdict = "NO-VERDICT"


# --- the scheduler -----------------------------------------------------------------------------

def schedule(jobs: list[Job], args, out: Path, say: Printer) -> dict[str, ConfigState]:
    """One builder thread feeds a run queue that `--jobs` runners drain.

    A config's jobs are only enqueued once its simulator exists, which is what keeps the runners from
    blocking on a build -- and configs already built are enqueued up front, so runs start at once.
    """
    order: list[str] = []
    for j in jobs:
        if j.config not in order:
            order.append(j.config)
    configs = {name: ConfigState(name) for name in order}
    by_config: dict[str, list[Job]] = {name: [] for name in order}
    for j in jobs:
        by_config[j.config].append(j)

    runq: queue.Queue[Job | None] = queue.Queue()

    def builder() -> None:
        for name in order:
            cfg = configs[name]
            elaborate(cfg, args, out, say)
            if cfg.ready and not args.build_only:
                for j in by_config[name]:
                    runq.put(j)
            elif not cfg.ready:
                for j in by_config[name]:
                    j.verdict, j.detail = "NO-BUILD", cfg.error
        for _ in range(args.jobs):
            runq.put(None)

    def runner() -> None:
        while True:
            j = runq.get()
            try:
                if j is None:
                    return
                run_job(j, configs[j.config], args, say)
            finally:
                runq.task_done()

    bt = threading.Thread(target=builder, name="build", daemon=True)
    bt.start()
    runners = [threading.Thread(target=runner, name=f"run{i}", daemon=True)
               for i in range(args.jobs)]
    for r in runners:
        r.start()
    bt.join()
    for r in runners:
        r.join()
    return configs


# --- the spool: a background queue you can append to ------------------------------------------
#
# The synchronous modes above block the terminal for as long as the batch takes. The spool lets you
# keep typing: `simq add ...` writes a job file and returns, and a daemon drains the queue with the
# same discipline the in-process scheduler uses -- ONE elaboration at a time, `--jobs` runs.
#
# No sockets and no database. A job is a JSON file; claiming it is a single `os.rename` from
# pending/ to running/, which is atomic on a POSIX filesystem, so two daemons cannot double-run a
# job and a crashed daemon leaves its claim visible in running/ rather than losing it.

#: The spool lives at SIM_DIR/.simq (SPOOL, set by `configure()`): one queue per simulator.
#: Run options carried per job, so `add run` and `add run-debug` can share one queue.
JOB_OPTS = ("debug", "waves", "verbose", "max_cycles", "timeout", "rebuild",
            "rebuild_if_stale", "allow_stale", "make_jobs", "extra_sim_flags")


def spool_init() -> None:
    for d in (PENDING, RUNNING, DONE):
        d.mkdir(parents=True, exist_ok=True)


def spool_add(jobs: list[Job], args) -> list[str]:
    """Write one job file per job. The name carries a monotonic stamp, so the queue is FIFO."""
    import json
    spool_init()
    ids = []
    for i, j in enumerate(jobs):
        jid = f"{time.time_ns()}-{i:03d}"
        payload = {
            "id": jid, "config": j.config, "binary": str(j.binary), "name": j.name,
            "submitted": time.time(),
            "opts": {k: getattr(args, k) for k in JOB_OPTS},
            "scala_dir": [str(p) for p in args.scala_dir],
        }
        tmp = PENDING / f".{jid}.tmp"
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.rename(PENDING / f"{jid}.json")        # atomic publish: never a half-written job
        ids.append(jid)
    return ids


def spool_claim(path: Path) -> dict | None:
    """Move a pending job into running/ and return it, or None if someone else got there first."""
    import json
    dest = RUNNING / path.name
    try:
        path.rename(dest)
    except OSError:
        return None
    try:
        d = json.loads(dest.read_text())
        d["_path"] = str(dest)
        return d
    except Exception:
        dest.rename(DONE / path.name)              # unparseable: retire it rather than spin
        return None


def spool_finish(rec: dict, job: Job) -> None:
    import json
    rec.pop("_path", None)
    rec.update(verdict=job.verdict, detail=job.detail, seconds=round(job.seconds, 1),
               log=str(job.log), finished=time.time())
    (DONE / f"{rec['id']}.json").write_text(json.dumps(rec, indent=1))
    (RUNNING / f"{rec['id']}.json").unlink(missing_ok=True)


def daemon_alive() -> int | None:
    try:
        pid = int(DAEMON_PID.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    try:
        os.kill(pid, 0)                # signal 0 = liveness probe, sends nothing
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid                     # someone else's process, but alive
    return pid


def daemon_loop(args) -> int:
    """Drain the spool until idle for `--idle-exit` seconds (or forever with 0)."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    spool_init()
    DAEMON_PID.write_text(f"{os.getpid()}\n")
    say = Printer()
    say(f"DAEMON up, pid {os.getpid()}, {SIM.name}, {args.jobs} run lane(s), spool {SPOOL}")

    #: Config readiness is keyed on (config, debug): the two builds are different artifacts.
    states: dict[tuple[str, bool], ConfigState] = {}
    pool = ThreadPoolExecutor(max_workers=args.jobs, thread_name_prefix="run")
    inflight: set = set()
    last_activity = time.monotonic()
    stop = SPOOL / "stop"

    try:
        while True:
            if stop.exists():
                say("DAEMON stop requested")
                stop.unlink(missing_ok=True)
                break
            # Claim only what there is a free lane for. Claiming the whole queue and letting the
            # thread pool hold the backlog would make `pending/` a lie: status would report
            # `queued 0`, and drain/cancel/stop could not reach jobs that had not started. The
            # spool directory IS the queue, so it has to hold everything not yet running.
            inflight = {f for f in inflight if not f.done()}
            free = args.jobs - len(inflight)
            pend = sorted(PENDING.glob("*.json"))[:max(free, 0)] if free > 0 else []
            if not pend:
                idle_for = time.monotonic() - last_activity
                if (not inflight and args.idle_exit and idle_for > args.idle_exit
                        and not any(PENDING.glob("*.json"))):
                    say(f"DAEMON idle {idle_for:.0f}s, exiting")
                    break
                time.sleep(1.0)
                continue

            for p in pend:
                # Re-checked HERE, not just at the top of the outer loop: this loop walks the whole
                # queue, and an elaboration inside it takes minutes. Without this, `simq stop` on a
                # queue of 18 would dispatch all 18 first, which is not what stop means.
                if stop.exists():
                    break
                rec = spool_claim(p)
                if rec is None:
                    continue
                last_activity = time.monotonic()
                # Per-job options, layered over the daemon's own namespace.
                jargs = argparse.Namespace(**vars(args))
                for k, v in (rec.get("opts") or {}).items():
                    setattr(jargs, k, v)
                jargs.scala_dir = [Path(s) for s in rec.get("scala_dir", [])] or args.scala_dir

                key = (rec["config"], bool(jargs.debug))
                cfg = states.get(key)
                if cfg is None or not cfg.ready:
                    cfg = states.setdefault(key, ConfigState(rec["config"]))
                    if not cfg.ready:
                        # Blocking here is correct: elaboration is serial anyway, and runs already
                        # handed to the pool keep going while we wait.
                        elaborate(cfg, jargs, SPOOL, say)
                job = Job(config=rec["config"], binary=Path(rec["binary"]), name=rec["name"])
                job.log = SPOOL / "logs" / f"{rec['id']}-{rec['config']}__{rec['name']}.log"
                job.log.parent.mkdir(parents=True, exist_ok=True)
                if not cfg.ready:
                    job.verdict, job.detail = "NO-BUILD", cfg.error
                    say(f"RUN    {job.config}/{job.name:<26} NO-BUILD    {cfg.error}")
                    spool_finish(rec, job)
                    continue
                inflight.add(pool.submit(_daemon_run, job, cfg, jargs, say, rec))
    finally:
        pool.shutdown(wait=True)
        DAEMON_PID.unlink(missing_ok=True)
        say("DAEMON down")
    return 0


def _daemon_run(job: Job, cfg: ConfigState, jargs, say: Printer, rec: dict) -> None:
    # Stamp the running record: `submitted` alone cannot distinguish "queued for an hour" from
    # "running for an hour", and those call for different reactions.
    import json
    rec["started"] = time.time()
    try:
        rp = Path(rec["_path"])
        rp.write_text(json.dumps({k: v for k, v in rec.items() if k != "_path"}, indent=1))
    except OSError:
        pass
    try:
        run_job(job, cfg, jargs, say)
    except Exception as e:                                  # a crashed run must not kill the lane
        job.verdict, job.detail = "ERROR", str(e)[:88]
        say(f"RUN    {job.config}/{job.name:<26} ERROR       {job.detail}")
    spool_finish(rec, job)


def daemon_start(args) -> int:
    """Re-exec ourselves detached, so `add` can return to the prompt."""
    pid = daemon_alive()
    if pid:
        print(f"daemon already running (pid {pid})")
        return 0
    spool_init()
    cmd = [sys.executable, str(Path(__file__).resolve()), "daemon", "--foreground",
           f"--jobs={args.jobs}", f"--idle-exit={args.idle_exit}", f"--sim={SIM.name}"]
    with DAEMON_LOG.open("a") as fh:
        subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         cwd=SIM_DIR, start_new_session=True)
    for _ in range(50):                                      # wait for the pidfile
        if daemon_alive():
            break
        time.sleep(0.1)
    pid = daemon_alive()
    print(f"daemon started (pid {pid}); log {DAEMON_LOG}" if pid
          else f"daemon did not come up -- see {DAEMON_LOG}")
    return 0 if pid else 1


def spool_status(verbose: bool = False) -> int:
    import json
    spool_init()
    pid = daemon_alive()
    print(f"daemon   {'running, pid ' + str(pid) if pid else 'NOT running'}    {SIM.name}    "
          f"spool {SPOOL}")

    def load(d: Path) -> list[dict]:
        out = []
        for f in sorted(d.glob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except Exception:
                pass
        return out

    pend, run, done = load(PENDING), load(RUNNING), load(DONE)
    print(f"queued {len(pend)}   running {len(run)}   done {len(done)}")
    if run and not pid:
        # A claim in running/ with no daemon alive is an orphan: the daemon died mid-run. Saying so
        # beats leaving it to look like something is still in progress.
        print(f"  !! {len(run)} job(s) claimed but no daemon alive -- these did NOT finish.")
        print(f"     mv {RUNNING}/*.json {PENDING}/   to requeue them.")
    for r in run:
        st = r.get("started")
        print(f"  RUN    {r['config']}/{r['name']:<30} started {hhmmss(st)} "
              f"({ago(st)} ago, queued {hhmmss(r.get('submitted'))})")
    for r in pend:
        print(f"  QUEUED {r['config']}/{r['name']:<30} queued  {hhmmss(r.get('submitted'))} "
              f"({ago(r.get('submitted'))} ago)  {r['id']}")
    if done:
        bad = [r for r in done if r.get("verdict") != "PASS"]
        print(f"\n  {len(done) - len(bad)} passed, {len(bad)} did not:")
        for r in sorted(done if verbose else bad, key=lambda r: r.get("finished") or 0):
            fin = r.get("finished")
            print(f"  {r.get('verdict', '?'):<11} {r['config']}/{r['name']:<30} "
                  f"ended {hhmmss(fin)} ({ago(fin)} ago) in {r.get('seconds', 0):5.0f}s  "
                  f"{(r.get('detail') or '')[:60]}")
    return 1 if any(r.get("verdict") != "PASS" for r in done) else 0


def _du(paths) -> int:
    tot = 0
    for f in paths:
        try:
            tot += f.stat().st_size
        except OSError:
            pass
    return tot


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def spool_clear(args) -> int:
    """Retire finished records, and optionally the logs and waveforms beside them.

    Finished jobs accumulate forever otherwise, and `status` becomes a wall of history. The waveforms
    are the part that actually costs anything -- one per waved job, GB-scale, never collected -- so
    the byte count is reported rather than left for `du` to discover later.
    """
    spool_init()
    pid = daemon_alive()
    targets: list[Path] = sorted(DONE.glob("*.json"))
    what = ["done records"]

    logs: list[Path] = []
    if args.logs or args.all:
        logdir = SPOOL / "logs"
        if logdir.is_dir():
            # NEVER remove the log of a job that is still running: the process holds it open, so
            # unlinking silly-renames it (`.nfsXXXX` on NFS) and its remaining output goes to an
            # invisible inode. Same reason daemon.log is truncated rather than unlinked below.
            live = {r.stem for r in RUNNING.glob("*.json")}
            logs = [f for f in logdir.iterdir()
                    if f.is_file() and not any(f.name.startswith(i) for i in live)]
            skipped = sum(1 for f in logdir.iterdir() if f.is_file()) - len(logs)
            if skipped:
                print(f"note: keeping {skipped} log(s) belonging to running job(s).")
        what.append("job logs + waveforms")

    # The daemon holds daemon.log open for its whole life (it is the process's stdout). Unlinking
    # it leaves the daemon writing to a deleted inode and `watch` following a new empty file that
    # never receives anything -- measured, on 2026-09-21. Truncate in place instead: with O_APPEND
    # the daemon's next write simply starts from zero, and the path stays valid.
    truncate: list[Path] = []
    builds: list[Path] = []
    if args.all:
        if DAEMON_LOG.is_file():
            truncate = [DAEMON_LOG]
            what.append("daemon log (truncated)")
        builds = sorted(SPOOL.glob("build-*.log"))
        if builds:
            what.append("build logs")

    orphans = sorted(RUNNING.glob("*.json"))
    if orphans and not pid and args.all:
        targets += orphans
        what.append(f"{len(orphans)} orphaned claim(s)")
    elif orphans and pid:
        print(f"note: {len(orphans)} job(s) are RUNNING and are not touched.")

    freed = _du(targets) + _du(logs) + _du(truncate) + _du(builds)
    n = len(targets) + len(logs) + len(truncate) + len(builds)
    if not n:
        print("nothing to clear")
        return 0
    if args.dry_run:
        print(f"would remove {n} file(s), {_human(freed)}: {', '.join(what)}")
        return 0
    for f in targets + logs + builds:
        try:
            f.unlink()
        except OSError as e:
            print(f"  could not remove {f.name}: {e}")
    for f in truncate:
        try:
            os.truncate(f, 0)
        except OSError as e:
            print(f"  could not truncate {f.name}: {e}")
    print(f"cleared {n} file(s), {_human(freed)} freed ({', '.join(what)})")
    if pid:
        print(f"daemon (pid {pid}) is still running; new events will reopen the log.")
    return 0


# --- cli ---------------------------------------------------------------------------------------

#: The make-like verbs. `run-binary`/`run-binary-debug` are accepted because that is what the
#: chipyard Makefile calls them and muscle memory is a real thing.
# A run verb QUEUES by default -- the terminal comes straight back and the daemon does the work.
# `--wait` blocks instead, which is what you want from a script: only the blocking form can return
# a meaningful exit code, since `add` returns before the work happens.
# `build` is always synchronous: it is one serial action, and there is nothing to append to.
VERBS = {
    "build":            dict(build_only=True),
    "run":              dict(spool_cmd="add"),
    "run-binary":       dict(spool_cmd="add"),
    "run-debug":        dict(spool_cmd="add", debug=True),
    "run-binary-debug": dict(spool_cmd="add", debug=True),
    "run-waves":        dict(spool_cmd="add", debug=True, waves=True),
}
#: Spool verbs. `add <verb>` enqueues instead of blocking, so the terminal comes straight back.
#: `add` alone means `add run`; `add run-debug ...` etc. all work.
SPOOL_VERBS = {"add", "queue", "daemon", "status", "watch", "stop", "cancel", "drain", "clear"}

#: make-style variable -> argparse dest. Anything else in KEY=VALUE form is an error rather than
#: being silently ignored, which is how a typo'd TIMEOUT becomes a four-hour surprise.
VAR_DEST = {
    "CONFIG": "config", "BINARY": "_binaries", "JOBS": "jobs", "TIMEOUT": "timeout",
    "MAX_CYCLES": "max_cycles", "OUT": "out", "BINDIR": "_bindir", "MAKE_JOBS": "make_jobs",
    "HEARTBEAT": "build_heartbeat", "SIM": "sim", "EXTRA_SIM_FLAGS": "extra_sim_flags",
}
VAR_RE = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$", re.DOTALL)
INT_VARS = {"jobs", "timeout", "max_cycles", "make_jobs", "build_heartbeat"}
#: A BINARY= value with one of these suffixes is a LIST of jobs, not a binary. Keyed on the
#: extension rather than on "does this file exist", so the meaning is readable from the command.
JOBFILE_SUFFIXES = (".jobs", ".txt", ".list")


def split_front(argv: list[str]) -> tuple[dict, list[str], list[str]]:
    """Pull a leading verb and any KEY=VALUE tokens off argv.

    Returns (verb defaults, variable assignments as dest->str, the remaining argv for argparse).
    A `CONFIG:BINARY` job spec has no '=' so it passes straight through.
    """
    verb: dict = {}
    rest = list(argv)
    # A spool verb may be followed by a run verb: `add run-debug CONFIG=...`.
    if rest and rest[0].replace("_", "-") in SPOOL_VERBS:
        verb["spool_cmd"] = rest.pop(0).replace("_", "-")
        verb["spool_explicit"] = True
    if rest and rest[0].replace("_", "-") in VERBS:
        name = rest.pop(0).replace("_", "-")
        verb.update(VERBS[name])
        verb["verb_name"] = name          # echoed back in hints; do not re-derive it from flags

    varz: dict[str, str] = {}
    keep: list[str] = []
    for tok in rest:
        m = VAR_RE.match(tok)
        if not m:
            keep.append(tok)
            continue
        key, val = m.group(1), m.group(2)
        if key not in VAR_DEST:
            raise SystemExit(f"unknown variable {key}= (known: {', '.join(sorted(VAR_DEST))})")
        dest = VAR_DEST[key]
        # Repeated BINARY=/CONFIG=/BINDIR= accumulate; scalars take the last value.
        if dest in ("_binaries", "config", "_bindir") and dest in varz:
            varz[dest] += "," + val
        else:
            varz[dest] = val
    return verb, varz, keep


def expand_binary_var(value: str) -> list[str]:
    """A BINARY= value into a list of binary/job specs, expanding job files."""
    out: list[str] = []
    for item in (v.strip() for v in value.split(",")):
        if not item:
            continue
        out.append("@" + item if item.endswith(JOBFILE_SUFFIXES) else item)
    return out


def main() -> int:
    verb, varz, argv = split_front(sys.argv[1:])

    ap = argparse.ArgumentParser(
        prog="simq", description=__doc__, usage="simq <build|run|run-debug|run-waves> "
        "CONFIG=<cfg> [BINARY=<bin|file.jobs>] [TIMEOUT=s] [JOBS=n] [options]",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="*", metavar="JOB",
                    help="CONFIG:BINARY, or BINARY with -c, or @file")
    ap.add_argument("-c", "--config", help="default config for bare binary names")
    ap.add_argument("-j", "--jobs", type=int, default=8, help="concurrent RUNS (default 8)")
    ap.add_argument("--make-jobs", type=int, default=4, help="make -j for elaboration (default 4)")
    ap.add_argument("--build-heartbeat", type=int, default=60, metavar="SEC",
                    help="while elaborating, report elapsed time and the build phase every SEC "
                         "seconds (0 = silent; default 60)")
    ap.add_argument("-o", "--out", help="log directory (default sims/<SIM>/output/simq-<stamp>)")
    ap.add_argument("--sim", choices=sorted(SIMULATORS), default=None,
                    help="Chipyard simulator flow, i.e. which sims/<SIM> to use "
                         "(default: $SIMQ_SIM, else the first of vcs/xrun/verilator on PATH)")
    ap.add_argument("--extra-sim-flags", default="", metavar="FLAGS",
                    help="extra plusargs for every run, like Chipyard's EXTRA_SIM_FLAGS")
    ap.add_argument("-b", "--bindir", action="append", default=[], type=Path,
                    help="extra directory to resolve bare binary names in (repeatable)")
    ap.add_argument("--debug", action="store_true", help="use/build the -debug simulator")
    # Tri-state on purpose: None means "follow --debug". There is no reason to pay for the debug
    # build and then not dump, so debug implies waves; --no-waves is the escape hatch for when you
    # want the debug build's assertions without a GB of waveform per job.
    ap.add_argument("--waves", action=argparse.BooleanOptionalAction, default=None,
                    help="dump a waveform per job (the simulator's Chipyard default format). ON "
                         "whenever --debug is on; --no-waves to suppress. Implies --debug when "
                         "given on its own.")
    ap.add_argument("--verbose", action="store_true", help="+verbose instruction trace (big logs)")
    ap.add_argument("--max-cycles", type=int, default=10_000_000)
    ap.add_argument("--timeout", type=int, default=3600, help="per-run wall seconds (default 3600)")
    ap.add_argument("--rebuild", action="store_true", help="elaborate even if a simulator exists")
    ap.add_argument("--rebuild-if-stale", action=argparse.BooleanOptionalAction, default=True,
                    help="re-elaborate the configs whose simulator is older than its Scala. ON BY "
                         "DEFAULT; --no-rebuild-if-stale refuses them instead. Unlike --rebuild, "
                         "configs that are already current are left alone.")
    ap.add_argument("--allow-stale", action="store_true",
                    help="run a simulator older than its Scala sources (refused by default)")
    ap.add_argument("--scala-dir", action="append", default=[], type=Path,
                    help="Scala tree(s) to date-check the simulator against "
                         "(default: generators/gemmini/src/main/scala)")
    ap.add_argument("--build-only", action="store_true", help="elaborate the configs, run nothing")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--spool-cmd", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--verb-name", default="run", help=argparse.SUPPRESS)
    ap.add_argument("--spool-explicit", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("-w", "--wait", action="store_true",
                    help="run in THIS terminal and block until every job finishes, instead of "
                         "queueing. The only form with a meaningful exit code.")
    ap.add_argument("--foreground", action="store_true",
                    help="daemon: run in this terminal instead of detaching")
    ap.add_argument("--idle-exit", type=int, default=3600,
                    help="daemon: exit after this many idle seconds (0 = never, default 3600)")
    ap.add_argument("--no-start", action="store_true",
                    help="add: do not auto-start the daemon if it is not running")
    ap.add_argument("--all", action="store_true",
                    help="status: list passing jobs too. clear: also remove logs, waveforms, the "
                         "daemon log, build logs and orphaned claims")
    ap.add_argument("--logs", action="store_true",
                    help="clear: also remove the per-job logs and their waveforms")
    ap.add_argument("--now", action="store_true",
                    help="stop: kill the daemon AND its in-flight simulations, instead of "
                         "waiting for them to finish")

    # Variables become argparse DEFAULTS, so an explicit dash flag still wins over KEY=VALUE.
    bin_specs: list[str] = []
    extra_bindirs: list[Path] = []
    defaults: dict = {}
    for dest, raw in varz.items():
        if dest == "_binaries":
            bin_specs = expand_binary_var(raw)
        elif dest == "_bindir":
            extra_bindirs = [Path(p) for p in raw.split(",") if p.strip()]
        elif dest in INT_VARS:
            try:
                defaults[dest] = int(raw)
            except ValueError:
                raise SystemExit(f"{dest.upper()}={raw!r} is not an integer")
        else:
            defaults[dest] = raw
    ap.set_defaults(**{**verb, **defaults})
    args = ap.parse_args(argv)

    # `--waves` on its own selects the debug build, since only it can dump; and `--debug` turns
    # dumping on, because the debug build exists to be looked at. `--no-waves` opts out either way.
    if args.waves:
        args.debug = True
    elif args.waves is None:
        args.waves = bool(args.debug)
    if args.wait:
        if args.spool_explicit and args.spool_cmd in ("add", "queue"):
            ap.error("`add` queues and `--wait` blocks -- pick one. "
                     "`simq run ... --wait` is the blocking form.")
        if args.spool_cmd in ("add", "queue"):
            args.spool_cmd = None
    if not args.scala_dir:
        args.scala_dir = [REPO / "generators" / "gemmini" / "src" / "main" / "scala"]
    if args.jobs < 1:
        ap.error("JOBS/--jobs must be >= 1")
    configure(pick_simulator(args.sim))
    if not SIM_DIR.is_dir():
        raise SystemExit(f"sim dir not found: {SIM_DIR}")

    # ---- spool commands that need no job list -------------------------------------------------
    sc = args.spool_cmd
    if sc == "status":
        return spool_status(args.all)
    if sc == "watch":
        spool_init()
        pid = daemon_alive()
        n_pend = len(list(PENDING.glob("*.json")))
        n_run = len(list(RUNNING.glob("*.json")))
        # Say what is going on BEFORE blocking on tail. A bare "following ..." over an empty file
        # is indistinguishable from a hang, which is exactly how the deleted-log bug presented.
        print(f"daemon {'running, pid ' + str(pid) if pid else 'NOT running'}    "
              f"queued {n_pend}   running {n_run}")
        if not DAEMON_LOG.exists() or DAEMON_LOG.stat().st_size == 0:
            # A daemon whose log was unlinked keeps writing to the deleted inode; on NFS that is
            # visible as a .nfsXXXX sibling, and it is the only place its output still goes.
            orphan = sorted(SPOOL.glob(".nfs*"), key=lambda f: f.stat().st_size, reverse=True)
            if pid and orphan:
                print(f"\nWARNING: {DAEMON_LOG.name} is empty but the daemon is alive -- its log was\n"
                      f"removed out from under it and its output is going to {orphan[0].name}.\n"
                      f"Following that instead. Restart the daemon (`simq stop` then `simq add ...`)\n"
                      f"to get a clean log; `simq clear` no longer unlinks it.\n")
                try:
                    return subprocess.call(["tail", "-n", "40", "-f", str(orphan[0])])
                except KeyboardInterrupt:
                    return 0
            print(f"\n{DAEMON_LOG} is empty" + ("." if pid else
                  " and no daemon is running, so nothing will arrive.\n"
                  "Start one with `simq add ...` or `simq daemon`."))
            if not pid:
                return 1
        print(f"\nfollowing {DAEMON_LOG} -- ctrl-C to stop watching (the daemon keeps running)")
        try:
            return subprocess.call(["tail", "-n", "40", "-f", str(DAEMON_LOG)])
        except KeyboardInterrupt:
            return 0
    if sc == "stop":
        spool_init()
        pid = daemon_alive()
        if not pid:
            print("daemon not running")
            # Claims left behind by a dead daemon are visible rather than lost; say so.
            orphans = list(RUNNING.glob("*.json"))
            if orphans:
                print(f"  note: {len(orphans)} job(s) still in running/ from a previous daemon. "
                      f"`simq status` lists them; move them back to pending/ to retry.")
            return 0
        if args.now:
            # The daemon was started with start_new_session=True, so it leads its own process
            # group -- signalling the group is what also reaches the simulator children. Killing only
            # the daemon would leave hour-long simulations running with nothing collecting them.
            import signal
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                print(f"SIGTERM sent to daemon {pid} and its simulations")
            except (ProcessLookupError, PermissionError) as e:
                print(f"could not signal the process group ({e}); trying the daemon alone")
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for _ in range(50):
                if not daemon_alive():
                    break
                time.sleep(0.1)
            DAEMON_PID.unlink(missing_ok=True)
            left = list(RUNNING.glob("*.json"))
            print(f"daemon stopped." + (f" {len(left)} job(s) were in flight and are still in "
                                        f"running/ -- they did not finish." if left else ""))
            return 0
        (SPOOL / "stop").touch()
        print(f"asked daemon {pid} to stop: it will claim no new jobs and exit once its in-flight\n"
              f"runs finish. Queued jobs stay queued (`simq drain` to discard them, "
              f"`simq stop --now` to kill in-flight runs too).")
        return 0
    if sc == "clear":
        return spool_clear(args)
    if sc == "drain":
        spool_init()
        n = 0
        for f in PENDING.glob("*.json"):
            f.unlink(missing_ok=True)
            n += 1
        print(f"dropped {n} queued job(s); running jobs are untouched")
        return 0
    if sc == "cancel":
        spool_init()
        wanted = set(args.specs)
        if not wanted:
            ap.error("cancel needs job id(s), or use `drain` to clear the whole queue")
        n = 0
        for f in PENDING.glob("*.json"):
            if f.stem in wanted or any(w in f.stem for w in wanted):
                f.unlink(missing_ok=True)
                n += 1
        print(f"cancelled {n} queued job(s)")
        return 0
    if sc == "daemon":
        if args.foreground:
            return daemon_loop(args)
        return daemon_start(args)

    bindirs = [*extra_bindirs, *args.bindir, *DEFAULT_BINDIRS]
    #: CONFIG= may name several; the cross product with BINARY= is the make-like reading.
    cfg_list = [c.strip() for c in (args.config or "").split(",") if c.strip()]

    specs = list(args.specs)
    if bin_specs:
        if not cfg_list:
            ap.error("BINARY= needs CONFIG= (or use the CONFIG:BINARY form)")
        # Qualify each binary with each config explicitly, rather than leaning on the -c default,
        # so CONFIG=A,B really does mean "both configs".
        for c in cfg_list:
            for b in bin_specs:
                specs.append(b if b.startswith("@") and len(cfg_list) == 1 else
                             (b if b.startswith("@") else f"{c}:{b}"))
        if len(cfg_list) > 1 and any(b.startswith("@") for b in bin_specs):
            ap.error("a BINARY= job file with CONFIG= naming several configs is ambiguous; "
                     "run one config at a time, or put CONFIG:BINARY lines in the file")

    default_cfg = cfg_list[0] if len(cfg_list) == 1 else None
    if args.build_only:
        if not cfg_list:
            ap.error("build needs CONFIG=<cfg>")
        jobs = parse_jobs(specs, default_cfg, bindirs) if specs else []
        configs_only = cfg_list if not jobs else []
    else:
        if not specs:
            ap.error("nothing to run: give BINARY=<bin|file.jobs>, or a CONFIG:BINARY job")
        jobs = parse_jobs(specs, default_cfg, bindirs)
        configs_only = []

    # ---- enqueue and return, rather than blocking the terminal --------------------------------
    if sc in ("add", "queue"):
        if args.build_only:
            ap.error("`add build` is not a thing: a queued run elaborates its own config. "
                     "Use `simq build CONFIG=...` to elaborate now, or just queue the runs.")
        if not jobs:
            ap.error("nothing to queue: give BINARY=<bin|file.jobs>, or a CONFIG:BINARY job")
        ids = spool_add(jobs, args)
        print(f"queued {len(ids)} job(s) in the background "
              f"(`simq {args.verb_name} ... --wait` to block instead):")
        for j, jid in zip(jobs, ids):
            print(f"  {jid}  {j.config}/{j.name}")
        if not daemon_alive() and not args.no_start:
            daemon_start(args)
        elif daemon_alive():
            print(f"daemon running (pid {daemon_alive()}) -- `simq status` or `simq watch`")
        else:
            print("daemon NOT running -- start it with `simq daemon`")
        return 0

    out = Path(args.out) if args.out else SIM_DIR / "output" / time.strftime("simq-%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    for j in jobs:
        j.log = out / f"{j.config}__{j.name}.log"

    n_cfg = len({j.config for j in jobs}) or len(configs_only)
    print(f"simq   {len(jobs)} job(s) over {n_cfg} config(s), {args.jobs} concurrent runs, "
          f"1 serial elaboration, {SIM.name}{' (debug build)' if args.debug else ''}")
    print(f"logs   {out}")
    # `run` and `add run` differ by one word, and the blocking one is the more natural thing to
    # type -- so say which mode this is rather than letting a 10-minute elaboration reveal it.
    if len(jobs) > 1 and not args.dry_run:
        print(f"mode   FOREGROUND (--wait): this terminal is blocked until all {len(jobs)} "
              f"finish.\n       Drop --wait to queue them and get the prompt back.")
    if args.dry_run or args.build_only and not jobs:
        for name in (configs_only or sorted({j.config for j in jobs})):
            simv = find_simv(name, args.debug)
            print(f"  build {name:<34} {'reuse ' + simv.name if simv and not args.rebuild else 'ELABORATE'}")
        for j in jobs:
            print(f"  run   {j.config}/{j.name:<26} {j.binary}")
        if args.dry_run:
            return 0
    print()

    if args.build_only and not jobs:
        say = Printer()
        states = [ConfigState(c) for c in configs_only]
        for cfg in states:                       # SERIAL: sbt holds a lock
            elaborate(cfg, args, out, say)
        bad = [c.name for c in states if not c.ready]
        print()
        for c in states:
            print(f"{c.name:<52} {'OK' if c.ready else 'FAILED':<11} {c.seconds:5.0f}s  {c.error}")
        if bad:
            print(f"\n{len(bad)} of {len(states)} config(s) did not build -- logs in {out}")
            return 1
        print(f"\nBUILT {len(states)} config(s)")
        return 0

    say = Printer()
    t0 = time.monotonic()
    configs = schedule(jobs, args, out, say)
    wall = time.monotonic() - t0

    # ---- summary, in submission order (the live lines above are in completion order) ----
    print()
    print(f"{'CONFIG/BINARY':<52} {'VERDICT':<11} {'TIME':>6}  DETAIL")
    print(f"{'-' * 52} {'-' * 11} {'-' * 6}  {'-' * 6}")
    bad = 0
    for j in jobs:
        if j.verdict != "PASS":
            bad += 1
        print(f"{j.config + '/' + j.name:<52} {j.verdict:<11} {j.seconds:5.0f}s  {j.detail}")
    # The baseline worth quoting is "elaborate everything, then run everything one at a time",
    # which is what a serial script costs. Dividing by (wall - build) would be wrong: with
    # pipelining the runs OVERLAP the builds, so that window is not the run window.
    build_time = sum(c.seconds for c in configs.values())
    run_time = sum(j.seconds for j in jobs)
    serial = build_time + run_time
    print()
    print(f"wall {wall:.0f}s   elaboration {build_time:.0f}s (serial)   runs {run_time:.0f}s summed "
          f"over {args.jobs} lanes   vs {serial:.0f}s if fully serial = {serial / max(wall, 1):.1f}x")
    if bad:
        print(f"{bad} of {len(jobs)} job(s) did not PASS -- logs in {out}")
        return 1
    print(f"ALL {len(jobs)} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
