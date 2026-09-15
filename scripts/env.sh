# Source this before running anything that touches spike.
#   source scripts/env.sh [chipyard-root]
#
# Sets MERLIN_CHIPYARD, RISCV and PATH. Chipyard's own env.sh only activates
# conda -- it does NOT set $RISCV -- so setting it here is required, not optional.
# Override the tree by passing a path or pre-setting MERLIN_CHIPYARD.
#
# Default: <repo>/toolchain, the chipyard-shaped tree scripts/setup.sh builds
# (.conda-env/ + generators/gemmini/). A real chipyard checkout works the same way.

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_cy="${1:-${MERLIN_CHIPYARD:-$_here/../toolchain}}"

if [ ! -d "$_cy" ]; then
    echo "env.sh: chipyard tree not found: $_cy" >&2
    echo "        run 'bash scripts/setup.sh' first, or pass a chipyard root:" >&2
    echo "        source scripts/env.sh /path/to/chipyard" >&2
    return 1 2>/dev/null || exit 1
fi
_cy="$(cd "$_cy" && pwd)"

export MERLIN_CHIPYARD="$_cy"
export RISCV="$_cy/.conda-env/riscv-tools"
# dtc lives in the chipyard conda env; spike shells out to it and dies without it.
export PATH="$_cy/.conda-env/bin:$RISCV/bin:$PATH"

for _b in "$RISCV/bin/spike" "$RISCV/bin/riscv64-unknown-elf-gcc" "$_cy/.conda-env/bin/dtc"; do
    [ -x "$_b" ] || echo "env.sh: WARNING missing $_b" >&2
done

# libgemmini.so is the spike functional model of the MX datapath. Its Makefile
# lists ONLY gemmini.cc as a prerequisite, so edits to mx_fp_math.h do not
# trigger a rebuild -- and a stale model fails as an unhandled trap
# (tohost = 1337), not as a clear error. Warn loudly when it is out of date.
_lg="$_cy/generators/gemmini/software/libgemmini"
if [ -f "$_lg/libgemmini.so" ]; then
    for _src in "$_lg/gemmini.cc" "$_lg/mx_fp_math.h"; do
        if [ -f "$_src" ] && [ "$_src" -nt "$_lg/libgemmini.so" ]; then
            echo "env.sh: WARNING libgemmini.so is STALE (older than $(basename "$_src"))." >&2
            echo "        rebuild:  (cd $_lg && make)" >&2
        fi
    done
else
    echo "env.sh: WARNING libgemmini.so not built -- (cd $_lg && make)" >&2
fi
unset _cy _b _lg _src _here

echo "MERLIN_CHIPYARD=$MERLIN_CHIPYARD"
echo "RISCV=$RISCV"

# Hardware sources AND toolchain both come from the chipyard tree -- there is no in-repo pin, so
# the spike model and the spike that loads it cannot drift apart. Report which tree that is.
# (Read $MERLIN_CHIPYARD, not $_cy -- it was unset above, which also used to leak exit status 1
# out of the [ -n ] short-circuit and break `source scripts/env.sh && ...` chains.)
_cy_head=$(git -C "$MERLIN_CHIPYARD/generators/gemmini" rev-parse --short HEAD 2>/dev/null)
if [ -n "$_cy_head" ]; then
    echo "gemmini sources: $MERLIN_CHIPYARD/generators/gemmini @ $_cy_head"
fi
unset _cy_head
true
