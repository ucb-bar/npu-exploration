// attn_flash_llama_vb_fused plus the fenced serial pass on the same hardware: the diff report names which O tiles and
// which softmax stat buffers (m, a, 1/l, lt) of the pipelined pass differ from it. Diagnostic only (slow).
#define ATTN_SERIAL 1
#include "attn_flash_llama_vb_fused.c"
