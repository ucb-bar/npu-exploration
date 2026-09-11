# Source this before running anything that touches spike.
#   source scripts/env.sh [chipyard-root]
#
# Sets MERLIN_CHIPYARD, RISCV and PATH. Chipyard's own env.sh only activates
# conda -- it does NOT set $RISCV -- so setting it here is required, not optional.
# Override the tree by passing a path or pre-setting MERLIN_CHIPYARD.

_cy="${1:-${MERLIN_CHIPYARD:-$HOME/orcd/scratch/npu-exploration/chipyard-graphics}}"

if [ ! -d "$_cy" ]; then
    echo "env.sh: chipyard tree not found: $_cy" >&2
    return 1 2>/dev/null || exit 1
fi

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
unset _cy _b _lg _src

echo "MERLIN_CHIPYARD=$MERLIN_CHIPYARD"
echo "RISCV=$RISCV"

# Hardware sources AND toolchain both come from the chipyard tree -- there is no in-repo pin, so
# the spike model and the spike that loads it cannot drift apart. Report which tree that is.
_cy_head=$(git -C "$_cy/generators/gemmini" rev-parse --short HEAD 2>/dev/null)
[ -n "$_cy_head" ] && echo "gemmini sources: $_cy/generators/gemmini @ $_cy_head"
