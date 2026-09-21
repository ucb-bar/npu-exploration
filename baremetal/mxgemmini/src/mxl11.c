// K=1536 = 48 groups, midway between the passing 32 (mxl3) and the failing 64 (mxl4).
// Brackets the cliff with mxl10 whichever way that one goes.
//
// Second-tier rung: added after mxl0..mxl3 PASSED and mxl4 FAILED on MxGemminiRocketConfig, which
// put the cliff between K=1024 and K=2048 without saying which of the four things that change
// across it is responsible. gen/gen_mx_ladder.py carries the truth table.
//
// Generated data: data/mx_ladder_mxl11.h. Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl11.h"
#define LADDER_KIND   KIND_PLAIN
#include "mx_ladder.c"
