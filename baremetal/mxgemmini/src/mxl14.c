// 32 groups like the passing mxl3, but M=64 gives it mxl4's 2048-byte A-side scale
// window. Separates 'the A window is too big' from 'there are too many groups' -- the same
// statement at M=32, since the window is groups*M bytes.
//
// Second-tier rung: added after mxl0..mxl3 PASSED and mxl4 FAILED on MxGemminiRocketConfig, which
// put the cliff between K=1024 and K=2048 without saying which of the four things that change
// across it is responsible. gen/gen_mx_ladder.py carries the truth table.
//
// Generated data: data/mx_ladder_mxl14.h. Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl14.h"
#define LADDER_KIND   KIND_PLAIN
#include "mx_ladder.c"
