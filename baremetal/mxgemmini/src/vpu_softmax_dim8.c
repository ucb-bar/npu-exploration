// vpu_softmax on the 8x8 mesh (MxDim8E3M2VpuGemminiRocketConfig): one 8-byte spad row = 4 BF16 VPU lanes.
#define VPU_LANES 4
#include "../../../../software/gemmini-rocc-tests/bareMetalC/vpu_softmax.c"
