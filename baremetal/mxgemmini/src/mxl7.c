// The resident seam: O = A @ B requantized to FP8 straight into the scratchpad in
// the operand-A tiled layout, its E8M0 bytes written into the act-scale window, then read IN PLACE
// as the next matmul's A with no mvin and no scale reload. This is llama_attention's P@V -> o_proj,
// and the closest relative of matmul_tiled_fp8_64x64_chain, which passes -- so a failure here is
// about the SHAPE (M=32, K=32, N=64, non-square) rather than the seam itself.
//
// Generated data: data/mx_ladder_mxl7.h (gen/gen_mx_ladder.py). Driver: src/mx_ladder.c.
#define LADDER_HEADER "mx_ladder_mxl7.h"
#define LADDER_KIND   KIND_REQUANT
#include "mx_ladder.c"
