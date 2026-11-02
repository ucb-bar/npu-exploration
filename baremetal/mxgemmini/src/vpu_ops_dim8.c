// vpu_ops on the 8x8 mesh (MxDim8Fp4E3M2E4M3VpuGemminiRocketConfig): one 8-byte spad row = 4 BF16 VPU lanes.
#define VPU_LANES 4
#include "../../../../software/gemmini-rocc-tests/bareMetalC/vpu_ops.c"
