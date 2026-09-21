#!/usr/bin/env bash
# Run the MX bisection ladder (and optionally the fp8 ISA requant tests) on a VCS simv, in
# parallel, and print one PASS/FAIL table.
#
# `make run-ladder` runs the ladder on SPIKE. This is the RTL counterpart: same ELFs, same goldens,
# against an elaborated simv. A rung that fails here and passes on spike is a real RTL divergence.
#
#   ./run_ladder_rtl.sh                       # all 19 rungs, MxGemminiRocketConfig-debug
#   ./run_ladder_rtl.sh --isa                 # + the fp8 requant/chain ISA tests
#   ./run_ladder_rtl.sh mxl4 mxl10 mxl13      # just these
#   SIMV=simv-chipyard.harness-MxGemminiRocketConfig-PREWIDEN ./run_ladder_rtl.sh mxl4
#   JOBS=4 ./run_ladder_rtl.sh                # concurrency (default 6)
#
# Logs land in $OUTDIR (default: a ladder-rtl/ dir beside the simv's usual output).
set -u

REPO=${REPO:-/bwrcq/scratch/nicorakela/radiance-cy-dev}
SIM_DIR=${SIM_DIR:-$REPO/sims/vcs}
SIMV=${SIMV:-simv-chipyard.harness-MxGemminiRocketConfig-debug}
DRAMSIM=${DRAMSIM:-$REPO/generators/testchipip/src/main/resources/dramsim2_ini}
ELFDIR=${ELFDIR:-$REPO/generators/gemmini/npu-exploration/out/baremetal/mx_rocket}
ISADIR=${ISADIR:-$REPO/generators/gemmini/software/gemmini-rocc-tests/build_mx_rocket/bareMetalC}
OUTDIR=${OUTDIR:-$SIM_DIR/output/ladder-rtl}
JOBS=${JOBS:-6}
MAXCYC=${MAXCYC:-10000000}

LADDER=(mxl0 mxl1 mxl2 mxl3 mxl4 mxl5 mxl6 mxl7 mxl8 mxl9
        mxl10 mxl11 mxl12 mxl13 mxl14 mxl15 mxl16 mxl17 mxl18)
# The two tests that gate Fault A, plus the dim32 pair -- the MxRequantizer and ScaleFactorMem
# fixes touch code shared with the DIM=32 path, so it is not enough to check DIM=16.
ISA=(matmul_tiled_fp8_64x64 matmul_tiled_fp8_64x64_requant matmul_tiled_fp8_64x64_chain
     matmul_tiled_fp4_64x64_requant_dim32 matmul_tiled_fp4_128x128_nonrequant_dim32)

WANT_ISA=0
SEL=()
for a in "$@"; do
  case "$a" in
    --isa) WANT_ISA=1 ;;
    --isa-only) WANT_ISA=1; SEL=(__none__) ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) SEL+=("$a") ;;
  esac
done
[ ${#SEL[@]} -eq 0 ] && SEL=("${LADDER[@]}")
[ "${SEL[0]:-}" = "__none__" ] && SEL=()

[ -x "$SIM_DIR/$SIMV" ] || { echo "no simv at $SIM_DIR/$SIMV" >&2; exit 1; }
mkdir -p "$OUTDIR"
echo "simv    $SIMV  ($(date -r "$SIM_DIR/$SIMV" '+%Y-%m-%d %H:%M'))"
echo "elfs    $ELFDIR"
echo "logs    $OUTDIR"
echo "jobs    $JOBS"
echo

# The ladder prints its own "<name> PASSED/FAILED"; the ISA tests print "... test PASSED/FAILED".
# Anything with neither is NO-VERDICT -- a real outcome (hang, trap, max-cycles exhausted) that must
# never be read as a pass.
verdict_of() {
  local l=$1
  if   [ ! -s "$l" ];                                                      then echo NO-VERDICT
  elif grep -q MISSING "$l" 2>/dev/null;                                   then echo MISSING
  elif grep -qE 'FAILED' "$l" 2>/dev/null;                                 then echo FAIL
  elif grep -qE '(^| )PASSED' "$l" 2>/dev/null;                            then echo PASS
  else echo NO-VERDICT; fi
}

detail_of() {
  grep -hoE 'mesh [^:]*: [0-9]+/[0-9]+[^,]*(, [0-9]+/[0-9]+ scales differ)?|Scale\[[0-9]+\]\[[0-9]+\], Got: [0-9a-fx]+, Exp: [0-9a-fx]+' \
    "$1" 2>/dev/null | head -2 | paste -sd'|' - | cut -c1-88
}

# One binary. $1 = name, $2 = elf path. Never fails the batch: a crashed or timed-out run is a
# result, not a reason to stop the other 18. Reports THE MOMENT IT FINISHES -- with 19 rungs and a
# multi-minute sim each, a script that only speaks at the end is indistinguishable from a hung one.
run_one() {
  local name=$1 elf=$2 log="$OUTDIR/$1.log" t0 dt
  t0=$SECONDS
  if [ ! -f "$elf" ]; then
    echo "MISSING $elf" > "$log"
  else
    ( cd "$SIM_DIR" && timeout "${TIMEOUT:-3600}" "./$SIMV" \
        +permissive +dramsim "+dramsim_ini_dir=$DRAMSIM" "+max-cycles=$MAXCYC" \
        +vcs+initreg+1 +ntb_random_seed_automatic "+loadmem=$elf" \
        +permissive-off "$elf" </dev/null 2>/dev/null ) > "$log"
  fi
  dt=$((SECONDS - t0))
  # One printf = one write, so parallel workers interleave by line rather than mid-line.
  printf '  %-8s %-11s %4ds  %s\n' "$name" "$(verdict_of "$log")" "$dt" "$(detail_of "$log")"
}

launch() {
  printf '  %-8s started\n' "$1"
  run_one "$1" "$2" &
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
}

TOTAL=${#SEL[@]}
[ "$WANT_ISA" = 1 ] && TOTAL=$((TOTAL + ${#ISA[@]}))
echo "running $TOTAL test(s) -- results appear as each finishes, table at the end"
echo
for t in "${SEL[@]}";  do launch "$t" "$ELFDIR/$t"; done
if [ "$WANT_ISA" = 1 ]; then
  for t in "${ISA[@]}"; do launch "$t" "$ISADIR/$t-baremetal"; done
fi
wait

# ---- summary, in ladder order (the live lines above are in completion order) ------------------
echo
printf '%-42s %-11s %s\n' TEST VERDICT DETAIL
printf '%-42s %-11s %s\n' "$(printf '%0.s-' {1..42})" "$(printf '%0.s-' {1..11})" ------
fails=0; unknown=0
ALL=("${SEL[@]}")
[ "$WANT_ISA" = 1 ] && ALL+=("${ISA[@]}")
for n in "${ALL[@]}"; do
  l="$OUTDIR/$n.log"
  v=$(verdict_of "$l")
  case $v in FAIL) fails=$((fails+1));; MISSING|NO-VERDICT) unknown=$((unknown+1));; esac
  printf '%-42s %-11s %s\n' "$n" "$v" "$(detail_of "$l")"
done
echo
echo "$fails failed, $unknown without a verdict, of $TOTAL."
if [ "$fails" = 0 ] && [ "$unknown" = 0 ]; then echo "ALL PASS"; else exit 1; fi
