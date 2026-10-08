// attn_flash_llama7b_fused with Q, K, V and P in FP4 (E2M1, quad mesh: 4x the E4M3 rate): a throughput experiment, the
// same shape and schedule. Data: gen_attn_fp4.py --d 128 --scale-in-q --tag _llama7b_fp4 (golden hash from
// golden_attn_flash.py, P through the FP4 SPAD_REQUANT).
#define ATTN_EXPSUB 1
#define ATTN_EXPSUM 1
#define ATTN_ST_BANK2 1
#define ATTN_HEADER "attn_vpu_llama7b_fp4.h"
#define ATTN_EXPECT "attn_flash_llama7b_fp4_expect.h"
#ifndef BK
#define BK 128
#endif
#ifndef ATTN_V3
#define ATTN_V3 0
#endif
#ifndef BK_FIRST
#define BK_FIRST 32
#endif
#ifndef BK_LAST
#define BK_LAST 96
#endif
#include "attn_flash.c"
