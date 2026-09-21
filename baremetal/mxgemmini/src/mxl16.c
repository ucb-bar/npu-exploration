// One delta from mxl15: M=32, so I=2 while J and Kt stay 4. Convicts I alone.
//
// Third-tier rung: added after mxl7 FAILED with `0/2048 codes, 36/64 scales` -- the requantizer
// computes the right FP8 values and writes the wrong E8M0 bytes. The chain test PASSES on the same
// bitstream and checks those scales the same way, so what matters is the tile counts, not the
// layout. gen/gen_mx_ladder.py carries the table.
//
// Generated data: data/mx_ladder_mxl16.h. Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl16.h"
#define LADDER_KIND   KIND_REQUANT
#include "mx_ladder.c"
