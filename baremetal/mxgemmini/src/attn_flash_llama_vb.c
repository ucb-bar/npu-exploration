// attn_flash_llama with variable key blocks: 64 + 7 x 256 + 192 = 2048 keys. The short first block fills the pipeline
// sooner, the short last block shortens the drain. Same data as attn_flash_llama; its own expected O hash (the online
// softmax rounds per block).
#define ATTN_HEADER "attn_vpu_llama.h"
#define ATTN_EXPECT "attn_flash_llama_vb_expect.h"
#define BK 256
#define BK_FIRST 64
#define BK_LAST 192
#include "attn_flash.c"
