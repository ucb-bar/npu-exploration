// One delta from the PASSING mxl4: N=16, so J=1 (a single output column tile) at TK=128.
//
// Added after llama_attention_full's 8192-row plan (proj N=16, I=2 J=1 TK=128) got nearly all of Q
// wrong on the FPGA while its 16384-row plan (N=64, mxl4's shape) was bit-exact, and the N=16 plan
// was bit-exact on spike. No earlier rung has J=1. gen/gen_mx_ladder.py carries the rationale.
//
// Generated data: data/mx_ladder_mxl19.h. Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl19.h"
#define LADDER_KIND   KIND_PLAIN
#include "mx_ladder.c"
