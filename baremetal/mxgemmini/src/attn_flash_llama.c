// attn_flash.c on a real TinyLlama head: 64 new tokens (positions 2048..2111) attending to a 2048-token KV cache,
// d 64, Bk 256 (8 key blocks), K/V streamed. Data: gen/capture_attn_qkv.py + gen_attn_vpu.py --capture (attn_vpu_llama.h).
#define ATTN_HEADER "attn_vpu_llama.h"
#define ATTN_EXPECT "attn_flash_llama_expect.h"
#define BK 256
#include "attn_flash.c"
