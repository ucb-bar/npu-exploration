// attn_flash_llama_2h_fused with Q, K, V and P in FP4 (E2M1, quad mesh: 4x the E4M3 rate): a throughput experiment, the
// same heads / tokens / schedule. Data: gen_attn_fp4.py --capture h0 h1 --tag _llama_2h_fp4 (golden hash from
// golden_attn_flash.py, P through the FP4 SPAD_REQUANT).
#define ATTN_HEADER "attn_vpu_llama_2h_fp4.h"
#define ATTN_EXPECT "attn_flash_llama_2h_fp4_expect.h"
#define BK 128
#define ATTN_EXPSUB 1
#define ATTN_EXPSUM 1
#define ATTN_ST_BANK2 1
#include "attn_flash.c"
