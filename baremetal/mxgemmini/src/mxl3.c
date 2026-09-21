// One delta from mxl2: K=1024 -- TK=64, and a 2048-byte B-side scale window.
// The passing ISA tests load 128 bytes of B scales; this is the first rung well past them.
//
// Generated data: data/mx_ladder_mxl3.h (gen/gen_mx_ladder.py). Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl3.h"
#define LADDER_KIND   KIND_PLAIN
#include "mx_ladder.c"
