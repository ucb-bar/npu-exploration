// One delta from mxl0: M=32, so the tile grid is NON-SQUARE (I=2, J=4). Every
// ISA test that passes on this bitstream is square; every llama matmul is not.
//
// Generated data: data/mx_ladder_mxl1.h (gen/gen_mx_ladder.py). Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl1.h"
#define LADDER_KIND   KIND_PLAIN
#include "mx_ladder.c"
