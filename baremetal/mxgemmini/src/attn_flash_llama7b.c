// attn_flash.c at Llama-2-7B's attention shape: head_dim 128, one head per pass (no GQA, so no head packing), 64 query
// tokens over a 2048-token KV cache, non-causal. Key blocks 32 + 15 x 128 + 96: a 128-key block at d 128 takes the
// scratchpad rows a 256-key block takes at d 64, so the TinyLlama single-head bank layout and schedule carry over.
// Tuned with the spike perf model (firesim preset): V double-buffered, 32-key first block (156,411 -> 152,229 cycles).
// Q carries 1/sqrt(128) (folded into Wq). Data: gen_attn_vpu.py --d 128 --scale-in-q --tag _llama7b (random N(0,1)).
#define ATTN_HEADER "attn_vpu_llama7b.h"
#define ATTN_EXPECT "attn_flash_llama7b_expect.h"
#ifndef BK
#define BK 128
#endif
#ifndef ATTN_V3
#define ATTN_V3 0   // V double-buffered: measured faster here than the triple buffer
#endif
#ifndef BK_FIRST
#define BK_FIRST 32
#endif
#ifndef BK_LAST
#define BK_LAST 96
#endif
#include "attn_flash.c"
