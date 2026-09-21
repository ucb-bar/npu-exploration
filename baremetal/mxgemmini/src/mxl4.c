// llama's first projection, EXACTLY: Q = Xn @ Wq at M=32 D=2048 H=64, i.e. I=2,
// J=4, TK=128, a 2048-byte A scale window and a 4096-byte B one. This is the rung that matters
// most -- if it fails and mxl3 passes, the cliff is between TK=64 and TK=128 or between a 2 KB and
// a 4 KB scale window, and both are one more bisection away.
//
// Generated data: data/mx_ladder_mxl4.h (gen/gen_mx_ladder.py). Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl4.h"
#define LADDER_KIND   KIND_PLAIN
#include "mx_ladder.c"
