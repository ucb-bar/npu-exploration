// chained native E4M3 quad 128x128 on MxDim32E4M3QuadGemminiRocketConfig (2x2 output tiles, 4 K passes per matmul).
#define QUAD_HDR "include/matmul_data_e4m3_quad_chain_128x128_dim32.h"
#include "../../../../software/gemmini-rocc-tests/bareMetalC/matmul_tiled_fp8_e4m3_quad_chain_dim32.c"
