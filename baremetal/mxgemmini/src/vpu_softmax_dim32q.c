// vpu_softmax on the native E4M3 quad 32x32 mesh (MxDim32E4M3QuadGemminiRocketConfig): one 64-byte spad row = 32 BF16 VPU lanes.
#define VPU_LANES 32
#include "../../../../software/gemmini-rocc-tests/bareMetalC/vpu_softmax.c"
