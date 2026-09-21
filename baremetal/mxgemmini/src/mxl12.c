// 64 groups exactly like the FAILING mxl4, but N=32 halves the B-side scale window to
// 1024 bytes. PASSES if mxl4's failure is the B window; fails just like mxl4 if it is the group
// count or the A window.
//
// Second-tier rung: added after mxl0..mxl3 PASSED and mxl4 FAILED on MxGemminiRocketConfig, which
// put the cliff between K=1024 and K=2048 without saying which of the four things that change
// across it is responsible. gen/gen_mx_ladder.py carries the truth table.
//
// Generated data: data/mx_ladder_mxl12.h. Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl12.h"
#define LADDER_KIND   KIND_PLAIN
#include "mx_ladder.c"
