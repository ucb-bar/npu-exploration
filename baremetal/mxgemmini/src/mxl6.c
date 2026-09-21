// Output-region REUSE: two different matmuls into one C_spad, both asking to
// overwrite. The MX path used to accumulate unconditionally (gemmini.cc's `smem[idx] =
// bf16_accum_add(...)`), and mvout frees the spad rows but never touches that shadow accumulator.
// o_proj's N-chunk loop depends on the region being cleared.
//
// Generated data: data/mx_ladder_mxl6.h (gen/gen_mx_ladder.py). Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl6.h"
#define LADDER_KIND   KIND_REUSE
#include "mx_ladder.c"
