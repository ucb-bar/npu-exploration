#!/usr/bin/env bash
# setup.sh -- provision everything a fresh clone needs to run the full flow.
#
#   bash scripts/setup.sh                 # all phases, then the doctor
#   bash scripts/setup.sh --check         # doctor only: PASS/FAIL per requirement
#   bash scripts/setup.sh --phase <name>  # one phase: python mxquant toolchain
#                                         #            spike gemmini libgemmini ppa
#
# Idempotent: every phase checks its postcondition first and skips if satisfied,
# so re-running after a failure resumes where it left off.
#
# What each phase provides (and which code requires it):
#   python     .venv + requirements.txt        (every documented .venv/bin/python command)
#   mxquant    <repo>/MXQuant checkout          (app/mxq_golden.py imports it at load time;
#                                                grade/mxquant_ref.py needs origin/chloe-branch-all)
#   toolchain  <root>/.conda-env with riscv64-unknown-elf-gcc, dtc, and a host g++
#                                               (runner.py gate; spike shells out to dtc)
#   spike      riscv-isa-sim built from source into <root>/.conda-env/riscv-tools
#              (the ucb-bar conda riscv-tools package is the GNU toolchain ONLY --
#               spike is not packaged anywhere; chipyard builds it from source too)
#   gemmini    <root>/generators/gemmini        (config/build_spike.py patches its libgemmini
#                                                sources; runner.py needs gemmini-rocc-tests)
#   libgemmini stock libgemmini.so, built with the toolchain env's OWN g++ so its
#              libstdc++ can never be newer than the one spike's DT_RPATH resolves
#   ppa        ../MxGemmini-workspace clone     (config/ppa.py silicon-cost model; OPTIONAL --
#                                                runs proceed with "[ppa] UNAVAILABLE" without it)
#
# <root> defaults to <repo>/toolchain and is chipyard-SHAPED (.conda-env/ +
# generators/gemmini/), so a real chipyard tree can be substituted with --root
# (or by pre-setting MERLIN_CHIPYARD before scripts/env.sh) and nothing else changes.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- The one place each external's origin is defined. -------------------------------
# Quantizer: planned to move to the cleaned-up microscaling-quant once it is built out
# (and app/mxq_golden.py is refactored to call only it) -- that swap is this line.
MXQUANT_URL="git@github.com:chooper1/MXQuant.git"
MXQUANT_BRANCH="chloe-branch-all"    # grade/mxquant_ref.py extracts files from origin/<this>
# Public repos over HTTPS so no SSH keys are needed for them.
GEMMINI_URL="https://github.com/ucb-bar/gemmini.git"
GEMMINI_REF="gemmini-mx-cleanup"
SPIKE_URL="https://github.com/riscv-software-src/riscv-isa-sim.git"
SPIKE_REF="master"    # version string 1.1.1-dev, same lineage the orcd flow validated
PPA_URL="git@github.com:Rakanic/MxGemmini-workspace.git"
CONDA_CHANNELS=(--override-channels -c ucb-bar -c conda-forge)
TOOLCHAIN_PKGS=(riscv-tools dtc "gxx_linux-64=12")
# --------------------------------------------------------------------------------------

ROOT="$REPO/toolchain"
MXQUANT_SRC="$MXQUANT_URL"
NO_PPA=0
ONLY_PHASE=""
CHECK_ONLY=0

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --root)        ROOT="$2"; shift 2 ;;
        --mxquant)     MXQUANT_SRC="$2"; shift 2 ;;
        --gemmini-ref) GEMMINI_REF="$2"; shift 2 ;;
        --spike-ref)   SPIKE_REF="$2"; shift 2 ;;
        --no-ppa)      NO_PPA=1; shift ;;
        --phase)       ONLY_PHASE="$2"; shift 2 ;;
        --check)       CHECK_ONLY=1; shift ;;
        -h|--help)     usage ;;
        *) echo "setup.sh: unknown argument: $1" >&2; usage 1 ;;
    esac
done

RISCV_DIR="$ROOT/.conda-env/riscv-tools"
GEMMINI_DIR="$ROOT/generators/gemmini"
LIBGEMMINI_DIR="$GEMMINI_DIR/software/libgemmini"
CONDA_GXX="$ROOT/.conda-env/bin/x86_64-conda-linux-gnu-g++"
PPA_DIR="$(dirname "$REPO")/MxGemmini-workspace"

say()  { printf '\033[1m[setup:%s]\033[0m %s\n' "$1" "$2"; }
skip() { say "$1" "already satisfied -- skip"; }
die()  { printf '\033[1;31m[setup:%s]\033[0m %s\n' "$1" "$2" >&2; exit 1; }

need_conda() {
    command -v conda >/dev/null 2>&1 || die "$1" \
        "conda not found. Install miniconda (https://docs.conda.io) -- the RISC-V toolchain \
comes from the ucb-bar conda channel. Fallback without conda: point --root (or \
MERLIN_CHIPYARD) at a full chipyard checkout built with its build-setup.sh."
}

# ---- postconditions (shared by the phases and the doctor) ----------------------------

have_python()     { "$REPO/.venv/bin/python" -c 'import torch, numpy' >/dev/null 2>&1; }
have_mxquant()    { [ -e "$REPO/MXQuant/prodacc_bundle" ]; }
have_mxq_branch() { git -C "$REPO/MXQuant" rev-parse --verify -q "origin/$MXQUANT_BRANCH" >/dev/null 2>&1; }
have_toolchain()  { [ -x "$RISCV_DIR/bin/riscv64-unknown-elf-gcc" ] \
                    && [ -x "$ROOT/.conda-env/bin/dtc" ] && [ -x "$CONDA_GXX" ]; }
have_spike()      { [ -x "$RISCV_DIR/bin/spike" ] && [ -f "$RISCV_DIR/include/riscv/mmu.h" ]; }
have_gemmini()    { [ -f "$LIBGEMMINI_DIR/gemmini.cc" ] && [ -f "$LIBGEMMINI_DIR/mx_fp_math.h" ] \
                    && [ -d "$GEMMINI_DIR/software/gemmini-rocc-tests/bareMetalC" ]; }
have_merlin()     { [ -e "$REPO/merlin/merlin/python" ]; }
have_ppa()        { [ -f "${MX_PPA_ROOT:-$PPA_DIR/ppa}/compose_gemmini.py" ]; }
have_libgemmini() {
    local so="$LIBGEMMINI_DIR/libgemmini.so"
    [ -f "$so" ] || return 1
    local s
    for s in "$LIBGEMMINI_DIR/gemmini.cc" "$LIBGEMMINI_DIR/mx_fp_math.h"; do
        [ -f "$s" ] && [ "$s" -nt "$so" ] && return 1
    done
    return 0
}

# ---- phases --------------------------------------------------------------------------

phase_python() {
    if have_python; then skip python; return; fi
    if [ ! -x "$REPO/.venv/bin/python" ]; then
        if command -v conda >/dev/null 2>&1; then
            say python "creating $REPO/.venv (conda, python 3.11)"
            conda create -y -p "$REPO/.venv" --override-channels -c conda-forge python=3.11 pip
        else
            say python "creating $REPO/.venv (python3 -m venv)"
            python3 -m venv "$REPO/.venv"
        fi
    fi
    say python "pip install -r requirements.txt"
    "$REPO/.venv/bin/pip" install -r "$REPO/requirements.txt"
    have_python || die python ".venv exists but 'import torch, numpy' still fails"
}

phase_mxquant() {
    if have_mxquant && have_mxq_branch; then skip mxquant; return; fi
    if [ ! -e "$REPO/MXQuant" ]; then
        if [ -d "$MXQUANT_SRC" ]; then
            say mxquant "symlinking existing checkout: $MXQUANT_SRC"
            ln -s "$(cd "$MXQUANT_SRC" && pwd)" "$REPO/MXQuant"
        else
            say mxquant "cloning $MXQUANT_SRC (private -- needs your SSH access)"
            git clone "$MXQUANT_SRC" "$REPO/MXQuant"
        fi
    fi
    have_mxquant || die mxquant "$REPO/MXQuant exists but has no prodacc_bundle/ -- wrong checkout?"
    if ! have_mxq_branch; then
        say mxquant "fetching origin/$MXQUANT_BRANCH (grade/mxquant_ref.py reads from it)"
        git -C "$REPO/MXQuant" fetch origin "$MXQUANT_BRANCH"
    fi
}

phase_toolchain() {
    if have_toolchain; then skip toolchain; return; fi
    need_conda toolchain
    say toolchain "conda env at $ROOT/.conda-env: ${TOOLCHAIN_PKGS[*]} (ucb-bar + conda-forge)"
    mkdir -p "$ROOT"
    if [ -d "$ROOT/.conda-env" ]; then
        conda install -y -p "$ROOT/.conda-env" "${CONDA_CHANNELS[@]}" "${TOOLCHAIN_PKGS[@]}"
    else
        conda create  -y -p "$ROOT/.conda-env" "${CONDA_CHANNELS[@]}" "${TOOLCHAIN_PKGS[@]}"
    fi
    have_toolchain || die toolchain \
        "install finished but riscv64-unknown-elf-gcc/dtc/g++ are not where env.sh expects \
(under $ROOT/.conda-env). Check the conda output above."
}

phase_spike() {
    if have_spike; then skip spike; return; fi
    have_toolchain || die spike "toolchain missing -- run the toolchain phase first"
    local src="$ROOT/riscv-isa-sim"
    if [ ! -d "$src/.git" ]; then
        say spike "cloning $SPIKE_URL @ $SPIKE_REF"
        git clone -b "$SPIKE_REF" "$SPIKE_URL" "$src"
    fi
    say spike "building spike from source into $RISCV_DIR ($(git -C "$src" rev-parse --short HEAD))"
    # Build with the env's own compilers and rpath its lib, so spike is self-contained
    # and its DT_RPATH libstdc++ is the same one libgemmini builds link against.
    mkdir -p "$src/build"
    (cd "$src/build" && \
        PATH="$ROOT/.conda-env/bin:$PATH" \
        CC="$ROOT/.conda-env/bin/x86_64-conda-linux-gnu-gcc" CXX="$CONDA_GXX" \
        LDFLAGS="-Wl,-rpath,$ROOT/.conda-env/lib -L$ROOT/.conda-env/lib" \
        ../configure --prefix="$RISCV_DIR" && \
        make -j"$(nproc)" && make install)
    have_spike || die spike "build finished but $RISCV_DIR/bin/spike or its headers are missing"
}

phase_gemmini() {
    if have_gemmini; then skip gemmini; return; fi
    if [ ! -d "$GEMMINI_DIR/.git" ]; then
        say gemmini "cloning $GEMMINI_URL @ $GEMMINI_REF"
        mkdir -p "$ROOT/generators"
        git clone -b "$GEMMINI_REF" "$GEMMINI_URL" "$GEMMINI_DIR"
    fi
    say gemmini "init submodules: software/libgemmini software/gemmini-rocc-tests (recursive)"
    git -C "$GEMMINI_DIR" submodule update --init --recursive \
        software/libgemmini software/gemmini-rocc-tests
    have_gemmini || die gemmini "clone finished but expected sources are missing under $GEMMINI_DIR"
    say gemmini "hardware sources pinned at $(git -C "$GEMMINI_DIR" rev-parse --short HEAD) ($GEMMINI_REF)"
}

phase_libgemmini() {
    if have_libgemmini; then skip libgemmini; return; fi
    have_gemmini || die libgemmini "gemmini sources missing -- run the gemmini phase first"
    have_spike   || die libgemmini "spike missing (its headers are needed) -- run the spike phase first"
    # Build with the toolchain env's own g++: spike and this .so then share one
    # libstdc++, so the DT_RPATH/GLIBCXX dlopen failure cannot happen by construction.
    local gxx="${MX_HOST_GXX:-$CONDA_GXX}"
    [ -x "$gxx" ] || die libgemmini "no compiler at $gxx (set MX_HOST_GXX to override)"
    say libgemmini "building stock libgemmini.so with $gxx"
    # Same compile line config/build_spike.py uses for per-recipe builds.
    (cd "$LIBGEMMINI_DIR" && "$gxx" -L "$RISCV_DIR/lib" -Wl,-rpath,"$RISCV_DIR/lib" -shared \
        -o libgemmini.so -std=c++17 -I "$RISCV_DIR/include" -I . -fPIC -O3 gemmini.cc)
    have_libgemmini || die libgemmini "build produced no fresh libgemmini.so"
}

phase_ppa() {
    if [ "$NO_PPA" = 1 ]; then say ppa "skipped (--no-ppa)"; return; fi
    if have_ppa; then skip ppa; return; fi
    say ppa "cloning $PPA_URL beside the repo (silicon-cost model; optional)"
    if ! git clone "$PPA_URL" "$PPA_DIR"; then
        say ppa "WARNING: clone failed (no access?). PPA numbers will be unavailable;"
        say ppa "         everything else works. Retry later or set MX_PPA_ROOT."
    fi
}

phase_merlin() {
    if have_merlin; then return; fi
    say merlin "git submodule update --init merlin"
    git -C "$REPO" submodule update --init merlin || \
        die merlin "submodule init failed. Without SSH keys for ucb-bar/merlin, run: \
git config submodule.merlin.url https://github.com/ucb-bar/merlin.git  -- then re-run"
}

# ---- doctor --------------------------------------------------------------------------

doctor() {
    local bad=0
    row() { # row <required 1|0> <ok 1|0> <label> <detail>
        local mark="PASS"
        if [ "$2" != 1 ]; then
            if [ "$1" = 1 ]; then mark="FAIL"; bad=1; else mark="warn"; fi
        fi
        printf '  %-4s  %-34s %s\n' "$mark" "$3" "$4"
    }
    echo "setup.sh --check  (root: $ROOT)"
    row 1 "$([ -x "$RISCV_DIR/bin/spike" ] && echo 1)"                    "spike"                 "$RISCV_DIR/bin/spike"
    row 1 "$([ -x "$RISCV_DIR/bin/riscv64-unknown-elf-gcc" ] && echo 1)"  "riscv64-unknown-elf-gcc" "$RISCV_DIR/bin/"
    row 1 "$([ -x "$ROOT/.conda-env/bin/dtc" ] || command -v dtc >/dev/null 2>&1 && echo 1)" \
                                                                          "dtc"                   "spike shells out to it at runtime"
    row 1 "$(have_gemmini && echo 1)"                                     "gemmini sources + rocc-tests" "$GEMMINI_DIR"
    row 1 "$(have_libgemmini && echo 1)"                                  "libgemmini.so (fresh)" "$LIBGEMMINI_DIR"
    row 1 "$(have_merlin && echo 1)"                                      "merlin submodule"      "$REPO/merlin"
    row 1 "$(have_mxquant && echo 1)"                                     "MXQuant checkout"      "$REPO/MXQuant"
    row 1 "$(have_mxq_branch && echo 1)"                                  "MXQuant origin/$MXQUANT_BRANCH" "grade/mxquant_ref.py needs it"
    row 1 "$(have_python && echo 1)"                                      ".venv (torch, numpy)"  "$REPO/.venv"
    row 0 "$(have_ppa && echo 1)"                                         "PPA workspace (optional)" "${MX_PPA_ROOT:-$PPA_DIR/ppa}"
    echo
    if [ "$bad" = 0 ]; then
        echo "All required checks pass. Next:"
        echo "    source scripts/env.sh${ROOT:+ $ROOT}"
        echo "    .venv/bin/python run_kernel.py --kernel linear --config baseline"
    else
        echo "FAIL above. Re-run 'bash scripts/setup.sh' (idempotent) or the named phase:"
        echo "    bash scripts/setup.sh --phase <python|mxquant|toolchain|spike|gemmini|libgemmini|ppa>"
        return 1
    fi
}

# ---- main ----------------------------------------------------------------------------

if [ "$CHECK_ONLY" = 1 ]; then doctor; exit $?; fi

if [ -n "$ONLY_PHASE" ]; then
    case "$ONLY_PHASE" in
        python|mxquant|toolchain|spike|gemmini|libgemmini|ppa|merlin) "phase_$ONLY_PHASE" ;;
        *) die setup "unknown phase: $ONLY_PHASE" ;;
    esac
    exit 0
fi

phase_merlin
phase_python
phase_mxquant
phase_toolchain
phase_spike
phase_gemmini
phase_libgemmini
phase_ppa
echo
doctor
