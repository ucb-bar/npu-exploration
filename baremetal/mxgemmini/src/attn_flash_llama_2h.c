// GQA head packing: TinyLlama layer 5 query heads 0 and 1 share kv head 0, so their queries stack as 128 rows of one
// flash attention (rows 0..63 head 0, 64..127 head 1; softmax is per row, so it is exact). Bk 128 keeps every buffer
// the single-head size. Data: gen_attn_vpu.py --capture h0.npz h1.npz --tag _llama_2h (attn_vpu_llama_2h.h).
#define ATTN_HEADER "attn_vpu_llama_2h.h"
#define ATTN_EXPECT "attn_flash_llama_2h_expect.h"
#define BK 128
#include "attn_flash.c"
