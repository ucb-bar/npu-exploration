// mxl3's matmul with A as a column slice of a [32][2048] array, so the mvin DMA
// walks a 2048-byte row pitch to gather each 16-byte tile row. This is how a K-tile of Xn moves in
// without being copied out first, and it is what a BANK_ROWS=2048 build does on every projection.
//
// Generated data: data/mx_ladder_mxl8.h (gen/gen_mx_ladder.py). Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl8.h"
#define LADDER_KIND   KIND_STRIDED
#include "mx_ladder.c"
