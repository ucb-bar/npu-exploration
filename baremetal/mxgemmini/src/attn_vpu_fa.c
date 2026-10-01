// attn_vpu.c at the flash kernel's shape (Sq 64, Sk 256, d 128) as one dense block. No Rocket softmax
// (~2M cycles here, the bulk of an RTL run).
#define ATTN_HEADER "attn_vpu_fa.h"
#define ATTN_EXPECT "attn_vpu_fa_expect.h"
#define ATTN_HOST_SOFTMAX 0
#define ATTN_ACC_ROWS 16
#include "attn_vpu.c"
