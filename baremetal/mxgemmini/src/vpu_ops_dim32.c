// vpu_ops on the 32x32 mesh (MxDim32{E4M3,E3M2,Fp4}VpuGemminiRocketConfig): one 32-byte spad row = 16 BF16 VPU lanes.
#define VPU_LANES 16
#include "../../../../software/gemmini-rocc-tests/bareMetalC/vpu_ops.c"
