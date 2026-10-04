// attn_flash.c with a causal mask: TinyLlama layer 5 head 0, 64 queries = tokens 1984..2047 (a prefill chunk) over
// keys 0..2047, d 64, Bk 256; the chunk's own 64 keys are masked triangularly in the last key block.
// Data: gen_attn_vpu.py --capture ... --sq 64 --sk 2048 --tag _llama_causal --no-s-golden --causal.
#define ATTN_HEADER "attn_vpu_llama_causal.h"
#define ATTN_EXPECT "attn_flash_llama_causal_expect.h"
#define BK 256
#include "attn_flash.c"
