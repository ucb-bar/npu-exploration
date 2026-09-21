// The control. 64x64x64 -- the shape matmul_tiled_fp8_64x64 already passes on the
// FPGA -- but issued through llama_attention's own helpers (include/mx_mesh.h). A failure HERE is
// in the calling convention, not in any of llama's shapes, and every rung below it is moot.
//
// Generated data: data/mx_ladder_mxl0.h (gen/gen_mx_ladder.py). Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl0.h"
#define LADDER_KIND   KIND_PLAIN
#include "mx_ladder.c"
