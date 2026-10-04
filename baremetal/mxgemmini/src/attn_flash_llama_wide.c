// attn_flash_llama with ONLY the wide (4-tile) mvins: V stays double-buffered, so its load still waits on PV(j).
// Isolates the mvin change from the triple buffer (attn_flash_llama has both).
#define ATTN_HEADER "attn_vpu_llama.h"
#define ATTN_EXPECT "attn_flash_llama_expect.h"
#define BK 256
#define ATTN_V3 0
#include "attn_flash.c"
