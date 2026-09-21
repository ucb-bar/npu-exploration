// One delta from mxl15: N=32, so J=2 and the output carries ONE E8M0 block per row
// instead of two. Convicts J, and separately tests whether GN=1 is handled at all.
//
// Third-tier rung: added after mxl7 FAILED with `0/2048 codes, 36/64 scales` -- the requantizer
// computes the right FP8 values and writes the wrong E8M0 bytes. The chain test PASSES on the same
// bitstream and checks those scales the same way, so what matters is the tile counts, not the
// layout. gen/gen_mx_ladder.py carries the table.
//
// Generated data: data/mx_ladder_mxl18.h. Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl18.h"
#define LADDER_KIND   KIND_REQUANT
#include "mx_ladder.c"
