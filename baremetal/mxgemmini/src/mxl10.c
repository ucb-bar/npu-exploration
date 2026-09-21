// K=1088 = 34 E8M0 groups -- ONE group past mxl3's 32, which is the last passing rung.
// If the cliff is a 5-bit group index this is the first to fail, and it fails by a hair rather than
// by half the reduction, so the MAGNITUDE of the error here is itself evidence.
//
// Second-tier rung: added after mxl0..mxl3 PASSED and mxl4 FAILED on MxGemminiRocketConfig, which
// put the cliff between K=1024 and K=2048 without saying which of the four things that change
// across it is responsible. gen/gen_mx_ladder.py carries the truth table.
//
// Generated data: data/mx_ladder_mxl10.h. Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl10.h"
#define LADDER_KIND   KIND_PLAIN
#include "mx_ladder.c"
