// vpu_softmax on the 4x4 mesh (MxDim4{Fp4,E3M2}VpuGemminiRocketConfig): one 4-byte spad row = 2 BF16 VPU lanes.
#define VPU_LANES 2
#include "../../../../software/gemmini-rocc-tests/bareMetalC/vpu_softmax.c"
