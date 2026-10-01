// attn_vpu.c at Sq 64, Sk 128, d 64 -- between the tiny block and the flash shape, sized for RTL sim.
#define ATTN_HEADER "attn_vpu_mid.h"
#define ATTN_EXPECT "attn_vpu_mid_expect.h"
#define ATTN_HOST_SOFTMAX 0
#define ATTN_ACC_ROWS 16
#include "attn_vpu.c"
