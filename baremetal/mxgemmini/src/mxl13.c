// 32 groups like the PASSING mxl3, but N=128 gives it mxl4's 4096-byte B-side scale
// window. The only rung that can convict the B window on its own.
//
// Second-tier rung: added after mxl0..mxl3 PASSED and mxl4 FAILED on MxGemminiRocketConfig, which
// put the cliff between K=1024 and K=2048 without saying which of the four things that change
// across it is responsible. gen/gen_mx_ladder.py carries the truth table.
//
// Generated data: data/mx_ladder_mxl13.h. Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl13.h"
#define LADDER_KIND   KIND_PLAIN
#include "mx_ladder.c"
