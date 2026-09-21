// `llama_attention.c` at D = 64 -- one step below `llama_attention_small.c`'s 256, and the
// smallest hidden size the E8M0 block (32) and the mvin tile (DIM) allow.
//
// All six mesh matmuls are still here, including the resident P@V -> o_proj seam, but the
// projections collapse to one K-tile and o_proj to one chunk, and the header is 0.4 MB instead of
// 6 MB. Read alongside `llama_attention_small`: if the small one fails on hardware and this one
// passes, the fault scales with D or with the number of chunks; if both fail the same way, it is
// structural and the ladder (src/mx_ladder.c) will name it.
//
// Everything `llama_attention_small.c` says about what the slice costs applies here more strongly:
// RMSNorm over 64 features is not llama's, and Q/K/V are partial sums over 64 of 2048 inputs. The
// fp32 reference is recomputed on the slice, so the grade still describes the device.
//
//   cd gen && ../../../.venv/bin/python3 gen_llama_layer.py attn --d 64 --tag _tiny
#define LLAMA_ATTN_HEADER "llama_attn_tiny.h"
#include "llama_attention.c"
