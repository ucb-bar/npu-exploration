// attn_flash_llama_vb with the optional fused VPU ops (needs VpuParams(expSub = true, expSum = true)): EXPSUM
// computes exp(S - m) and its row sums in one pass; the softmax stats live in bank 2. Bit-identical to
// attn_flash_llama_vb, so it checks against the same expected O hash.
#define ATTN_EXPSUB 1
#define ATTN_EXPSUM 1
#define ATTN_ST_BANK2 1   // stats in bank 2: banks 0-1 only hold the mesh operands
#include "attn_flash_llama_vb.c"
