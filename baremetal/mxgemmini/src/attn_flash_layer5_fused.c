// attn_flash_layer5 with the optional fused VPU ops (needs VpuParams(expSub = true, expSum = true)) and the softmax stats
// in bank 2, as attn_flash_llama_2h_fused. Bit-identical to attn_flash_layer5: the same expected O hashes.
#define ATTN_EXPSUB 1
#define ATTN_EXPSUM 1
#define ATTN_ST_BANK2 1
#include "attn_flash_layer.c"
