// FP4 quad DRAM loop 128x128x128, BF16 out, on MxDim32AllGemminiRocketConfig (2x2 tiles).
#define FP4_HDR "include/matmul_fp4_128x128_dim32.h"
#include "../../../../software/gemmini-rocc-tests/bareMetalC/matmul_tiled_fp4_dramloop_dim32.c"
