// native E4M3 quad 128x128x128 E4M3 requant to spad on MxDim32E4M3QuadGemminiRocketConfig (2x2 output tiles).
#define QUAD_HDR "include/matmul_data_e4m3_quad_128x128_dim32.h"
#include "../../../../software/gemmini-rocc-tests/bareMetalC/matmul_tiled_fp8_e4m3_quad_requant_dim32.c"
