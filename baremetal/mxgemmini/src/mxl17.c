// One delta from mxl15: K=32, so Kt=2 while I and J stay 4. Convicts Kt alone.
//
// Third-tier rung: added after mxl7 FAILED with `0/2048 codes, 36/64 scales` -- the requantizer
// computes the right FP8 values and writes the wrong E8M0 bytes. The chain test PASSES on the same
// bitstream and checks those scales the same way, so what matters is the tile counts, not the
// layout. gen/gen_mx_ladder.py carries the table.
//
// Generated data: data/mx_ladder_mxl17.h. Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl17.h"
#define LADDER_KIND   KIND_REQUANT
#include "mx_ladder.c"
